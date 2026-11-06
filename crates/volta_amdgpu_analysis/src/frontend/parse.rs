use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

use super::{Global, Kernel, Module, SourceSpan, constant_expr, instr, metadata};

// Explicit bounded object-data envelope. Larger ELF objects are rejected before
// allocating host storage; their bytes must never be silently discarded.
const MAX_OBJECT_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub message: String,
    pub span: SourceSpan,
}

impl fmt::Display for ParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}:{}: {}",
            self.span.line, self.span.column, self.message
        )
    }
}

impl Error for ParseError {}

/// Parse native gfx942/gfx950/gfx1250 assembly without executing it.
///
/// Descriptors and AMDHSA metadata identify entry kernels. Explicitly declared,
/// bounded non-entry functions are retained separately. Every entry
/// instruction is checked, including instructions on unreachable paths. Without
/// entry declarations, all function bodies retain the strict handwritten dialect.
pub fn parse(source: &str) -> Result<Module, ParseError> {
    parse_module(source)
}

/// Raw instruction lines of one function body, after the structural directive
/// pass and before instruction parsing. Profiles use this to classify every
/// mnemonic of an entry, including forms the instruction parser would refuse.
pub fn function_source_lines(
    source: &str,
    name: &str,
) -> Result<Option<Vec<(String, SourceSpan)>>, ParseError> {
    let parser = structural_pass(source)?;
    Ok(parser
        .bodies
        .get(name)
        .map(|body| body.instructions.clone()))
}

pub fn parse_module(source: &str) -> Result<Module, ParseError> {
    finish_module(structural_pass(source)?)
}

fn structural_pass(source: &str) -> Result<Parser, ParseError> {
    let mut parser = Parser {
        module: Module {
            source: source.to_owned(),
            ..Module::default()
        },
        active: None,
        descriptor: None,
        functions: BTreeSet::new(),
        text_labels: BTreeSet::new(),
        bodies: BTreeMap::new(),
        closed_functions: BTreeSet::new(),
        overlapping_functions: Vec::new(),
        globals: BTreeSet::new(),
        text_section: true,
        metadata: None,
        metadata_start: SourceSpan::default(),
        metadata_blocks: Vec::new(),
        descriptors: BTreeSet::new(),
        in_block_comment: false,
        data_object: None,
        data_alignment: 1,
        data_read_only: false,
        data_section_supported: false,
        data_zero_fill: false,
        data_section_kinds: BTreeMap::from([
            (".data".into(), false),
            (".rodata".into(), false),
            (".bss".into(), true),
        ]),
        constant_definitions: BTreeMap::new(),
        constant_spans: BTreeMap::new(),
        trailing_owner: None,
    };
    parser.collect_function_declarations(source);
    let mut start = 0;
    for (line_index, raw) in source.split_inclusive('\n').enumerate() {
        let span = SourceSpan {
            line: line_index + 1,
            column: raw.len() - raw.trim_start().len() + 1,
            start: start + raw.len() - raw.trim_start().len(),
            end: start + raw.trim_end_matches(['\r', '\n']).len(),
        };
        start += raw.len();
        if let Some(buffer) = &mut parser.metadata {
            if raw.trim() == ".end_amdgpu_metadata" {
                let buffer = parser.metadata.take().unwrap();
                parser.metadata_blocks.push((buffer, parser.metadata_start));
            } else {
                buffer.push_str(raw);
            }
            continue;
        }
        let line = parser.strip_comments(raw);
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        parser.line(line, span)?;
    }
    if parser.in_block_comment {
        return Err(ParseError {
            message: "unterminated block comment".into(),
            span: SourceSpan {
                line: source.lines().count(),
                ..SourceSpan::default()
            },
        });
    }
    if parser.metadata.is_some() {
        return Err(ParseError {
            message: "unterminated .amdgpu_metadata".into(),
            span: parser.metadata_start,
        });
    }
    if let Some(index) = parser.descriptor {
        return Err(ParseError {
            message: format!(
                "unterminated .amdhsa_kernel {}",
                parser.module.kernels[index].name
            ),
            span: SourceSpan::default(),
        });
    }
    if let Some(index) = parser.data_object {
        return Err(ParseError {
            message: format!(
                "module object `{}` has no explicit .size",
                parser.module.globals[index].symbol
            ),
            span: parser.module.globals[index].span,
        });
    }
    Ok(parser)
}

fn finish_module(mut parser: Parser) -> Result<Module, ParseError> {
    if parser.metadata_blocks.len() > 1 {
        return Err(ParseError {
            message: "duplicate .amdgpu_metadata".into(),
            span: parser.metadata_blocks[1].1,
        });
    }
    parser.module.constants =
        constant_expr::resolve_all(&parser.constant_definitions).map_err(|(name, message)| {
            ParseError {
                message,
                span: parser.constant_spans[&name],
            }
        })?;
    let mut entries = parser.descriptors.clone();
    for (buffer, span) in std::mem::take(&mut parser.metadata_blocks) {
        entries.extend(metadata::apply_metadata(&mut parser.module, &buffer, span)?);
    }
    // Instruction parsing follows entry classification: compiler modules can
    // contain unrelated functions using instructions outside our kernel model.
    // Structural directives, metadata and globals have already been checked.
    let skipped = parser
        .module
        .kernels
        .iter()
        .filter(|kernel| !entries.is_empty() && !entries.contains(&kernel.name))
        .map(|kernel| kernel.name.clone())
        .collect::<BTreeSet<_>>();
    for name in &skipped {
        if !parser.functions.contains(name) || !parser.closed_functions.contains(name) {
            return Err(ParseError {
                message: format!(
                    "non-entry function `{name}` requires explicit .type and matching .size"
                ),
                span: parser.bodies[name].span,
            });
        }
    }
    for (previous, next, span) in &parser.overlapping_functions {
        if skipped.contains(previous) || skipped.contains(next) {
            return Err(ParseError {
                message: format!("function `{next}` begins before explicit .size for `{previous}`"),
                span: *span,
            });
        }
    }
    let architecture = crate::target::Architecture::from_target(
        parser.module.target.as_deref().unwrap_or("gfx950"),
    )
    .ok_or_else(|| ParseError {
        message: "unsupported AMD target".into(),
        span: SourceSpan::default(),
    })?;
    let mut entries = Vec::new();
    for mut kernel in std::mem::take(&mut parser.module.kernels) {
        let name = kernel.name.clone();
        let parsed = parse_body(
            &mut kernel,
            parser.bodies.get(&name),
            &parser.module.constants,
            architecture,
        );
        if skipped.contains(&kernel.name) {
            let refusal = parsed.err().map(|error| (error.message, error.span));
            // A failed prefix is never an executable helper body.
            if refusal.is_some() {
                kernel.instructions.clear();
            }
            let span = parser.bodies[&kernel.name].span;
            parser.module.functions.push(super::ast::Function {
                name: kernel.name,
                instructions: kernel.instructions,
                labels: kernel.labels,
                span,
                refusal,
            });
            continue;
        }
        parsed?;
        if let Some(instruction) = kernel
            .instructions
            .iter()
            .find(|instruction| instruction.mnemonic == "s_setpc_b64")
        {
            return Err(ParseError {
                message: "unsupported s_setpc_b64 return in an entry kernel without a caller"
                    .into(),
                span: instruction.span,
            });
        }
        // Do not turn a physical fallthrough into an implicit return when
        // excluding another function's bytes. Branch destinations are still
        // resolved strictly within this kernel by lowering.
        if !skipped.is_empty()
            && kernel.instructions.last().is_some_and(|instruction| {
                !matches!(instruction.mnemonic.as_str(), "s_endpgm" | "s_branch")
            })
        {
            return Err(ParseError {
                message: format!(
                    "entry kernel `{}` has unsupported function fallthrough",
                    kernel.name
                ),
                span: kernel.instructions.last().unwrap().span,
            });
        }
        entries.push(kernel);
    }
    parser.module.kernels = entries;
    crate::target::validate(&parser.module)?;
    if parser.module.kernels.is_empty() {
        return Err(ParseError {
            message: "no executable kernels found".into(),
            span: SourceSpan::default(),
        });
    }
    for kernel in &parser.module.kernels {
        if kernel.instructions.is_empty() {
            return Err(ParseError {
                message: format!("kernel `{}` has no instructions", kernel.name),
                span: SourceSpan::default(),
            });
        }
        validate_descriptor(kernel)?;
    }
    Ok(parser.module)
}

fn parse_body(
    kernel: &mut Kernel,
    body: Option<&FunctionBody>,
    constants: &BTreeMap<String, i64>,
    architecture: crate::target::Architecture,
) -> Result<(), ParseError> {
    if let Some(body) = body {
        for (line, span) in &body.instructions {
            let normalized =
                instr::resolve_register_indices(line, constants).map_err(|message| ParseError {
                    message,
                    span: *span,
                })?;
            kernel
                .instructions
                .push(instr::parse_instruction(&normalized, *span, architecture)?);
        }
    }
    for instruction in &mut kernel.instructions {
        if instruction.opcode == super::Opcode::Branch {
            continue;
        }
        for operand in &mut instruction.operands {
            if let super::Operand::Symbol(name) = operand
                && let Some(value) = constants.get(name)
            {
                *operand = super::Operand::Integer(*value);
            }
        }
        instr::validate_control(instruction)?;
        instr::normalize_f64_sources(instruction)?;
        instr::validate_scalar_sources(instruction, architecture)?;
    }
    Ok(())
}

struct Parser {
    module: Module,
    active: Option<usize>,
    descriptor: Option<usize>,
    functions: BTreeSet<String>,
    text_labels: BTreeSet<String>,
    bodies: BTreeMap<String, FunctionBody>,
    closed_functions: BTreeSet<String>,
    overlapping_functions: Vec<(String, String, SourceSpan)>,
    globals: BTreeSet<String>,
    text_section: bool,
    metadata: Option<String>,
    metadata_start: SourceSpan,
    metadata_blocks: Vec<(String, SourceSpan)>,
    descriptors: BTreeSet<String>,
    in_block_comment: bool,
    data_object: Option<usize>,
    data_alignment: u64,
    data_read_only: bool,
    data_section_supported: bool,
    data_zero_fill: bool,
    data_section_kinds: BTreeMap<String, bool>,
    constant_definitions: BTreeMap<String, constant_expr::Expr>,
    constant_spans: BTreeMap<String, SourceSpan>,
    trailing_owner: Option<usize>,
}

struct FunctionBody {
    span: SourceSpan,
    instructions: Vec<(String, SourceSpan)>,
}

impl Parser {
    // A function's .type may follow its label. Discover declarations before
    // assigning labels to bodies, using exactly the same comment lexer. The
    // normal pass validates directive syntax, symbol kinds and metadata.
    fn collect_function_declarations(&mut self, source: &str) {
        let mut in_metadata = false;
        for raw in source.split_inclusive('\n') {
            if in_metadata {
                in_metadata = raw.trim() != ".end_amdgpu_metadata";
                continue;
            }
            let line = self.strip_comments(raw);
            let mut line = line.trim();
            while let Some((label, rest)) = line.split_once(':')
                && valid_symbol(label)
            {
                line = rest.trim();
            }
            if line == ".amdgpu_metadata" {
                in_metadata = true;
                continue;
            }
            let (directive, args) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
            if directive == ".type"
                && let Ok(parts) = instr::split_top_level(args.trim(), ',')
                && parts.len() == 2
                && valid_symbol(parts[0])
                && matches!(parts[1], "@function" | "%function")
            {
                self.functions.insert(parts[0].into());
            }
        }
        self.in_block_comment = false;
    }

    fn strip_comments(&mut self, raw: &str) -> String {
        let mut result = String::new();
        let mut chars = raw.chars().peekable();
        let mut quoted = false;
        let mut escaped = false;
        while let Some(ch) = chars.next() {
            if self.in_block_comment {
                if ch == '*' && chars.peek() == Some(&'/') {
                    chars.next();
                    self.in_block_comment = false;
                }
                continue;
            }
            if quoted {
                result.push(ch);
                if escaped {
                    escaped = false;
                } else if ch == '\\' {
                    escaped = true;
                } else if ch == '"' {
                    quoted = false;
                }
                continue;
            }
            if ch == '"' {
                quoted = true;
                result.push(ch);
            } else if ch == ';' || ch == '#' || ch == '/' && chars.peek() == Some(&'/') {
                break;
            } else if ch == '/' && chars.peek() == Some(&'*') {
                chars.next();
                self.in_block_comment = true;
            } else {
                result.push(ch);
            }
        }
        result
    }

    fn line(&mut self, line: &str, span: SourceSpan) -> Result<(), ParseError> {
        let fail = |message| ParseError { message, span };
        // Labels are recognized before directives because local assembler labels
        // begin with a period. Do not confuse `offset:4` with a leading label.
        if let Some((label, rest)) = line.split_once(':')
            && !label.contains(char::is_whitespace)
            && !label.contains('[')
        {
            if !valid_symbol(label) {
                return Err(fail(format!("malformed label `{label}`")));
            }
            if self.text_section {
                if !self.text_labels.insert(label.into()) {
                    return Err(fail(format!("duplicate assembly label `{label}`")));
                }
                if self.functions.contains(label)
                    || !label.starts_with('.') && self.active.is_none()
                {
                    if self.bodies.contains_key(label) {
                        return Err(fail(format!("duplicate function `{label}`")));
                    }
                    if let Some(index) = self.active {
                        self.overlapping_functions.push((
                            self.module.kernels[index].name.clone(),
                            label.into(),
                            span,
                        ));
                    }
                    let index = if let Some(index) = self
                        .module
                        .kernels
                        .iter()
                        .position(|kernel| kernel.name == label)
                    {
                        if !self.module.kernels[index].instructions.is_empty()
                            || !self.module.kernels[index].labels.is_empty()
                        {
                            return Err(fail(format!("duplicate kernel `{label}`")));
                        }
                        index
                    } else {
                        let index = self.module.kernels.len();
                        self.module.kernels.push(Kernel {
                            name: label.into(),
                            ..Kernel::default()
                        });
                        index
                    };
                    self.bodies.insert(
                        label.into(),
                        FunctionBody {
                            span,
                            instructions: Vec::new(),
                        },
                    );
                    self.active = Some(index);
                    self.trailing_owner = None;
                }
                if let Some(index) = self.active {
                    let kernel = &mut self.module.kernels[index];
                    if kernel
                        .labels
                        .insert(label.into(), self.bodies[&kernel.name].instructions.len())
                        .is_some()
                    {
                        return Err(fail(format!("duplicate label `{label}`")));
                    }
                } else if !label.starts_with(".Lfunc_end") {
                    return Err(fail(format!("label `{label}` outside executable function")));
                }
            } else if self.globals.contains(label) {
                if self.data_object.is_some() {
                    return Err(fail(
                        "a module object must have .size before the next object".into(),
                    ));
                }
                if !self.data_section_supported {
                    return Err(fail(
                        "module object is outside a supported data/rodata/bss section".into(),
                    ));
                }
                if self
                    .module
                    .globals
                    .iter()
                    .any(|object| object.symbol == label)
                {
                    return Err(fail(format!("duplicate module object `{label}`")));
                }
                self.data_object = Some(self.module.globals.len());
                self.module.globals.push(Global {
                    symbol: label.into(),
                    size: 0,
                    alignment: self.data_alignment,
                    initial_bytes: Vec::new(),
                    read_only: self.data_read_only,
                    span,
                });
                self.data_alignment = 1;
            }
            if !rest.trim().is_empty() {
                self.line(rest.trim(), span)?;
            }
            return Ok(());
        }
        if line.starts_with('.') {
            return self.directive(line, span);
        }
        if !self.text_section {
            return Err(fail("executable instruction outside .text section".into()));
        }
        if self.descriptor.is_some() {
            return Err(fail("instruction inside kernel descriptor".into()));
        }
        let index = self
            .active
            .ok_or_else(|| fail("instruction outside executable function".into()))?;
        self.bodies
            .get_mut(&self.module.kernels[index].name)
            .unwrap()
            .instructions
            .push((line.into(), span));
        Ok(())
    }

    fn directive(&mut self, line: &str, span: SourceSpan) -> Result<(), ParseError> {
        let fail = |message| ParseError { message, span };
        let (name, args) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
        let args = args.trim();
        if let Some(field) = name.strip_prefix(".amdhsa_")
            && field != "kernel"
            && field != "code_object_version"
        {
            let index = self
                .descriptor
                .ok_or_else(|| fail(format!("`{name}` outside .amdhsa_kernel")))?;
            if !descriptor_fields().contains(&field) {
                return Err(fail(format!(
                    "unsupported kernel descriptor directive `{name}`"
                )));
            }
            let value = instr::integer(args)
                .filter(|value| *value >= 0)
                .ok_or_else(|| fail(format!("`{name}` requires a nonnegative integer")))?
                as u64;
            if self.module.kernels[index]
                .descriptor
                .fields
                .insert(field.into(), value)
                .is_some()
            {
                return Err(fail(format!("duplicate descriptor field `{name}`")));
            }
            return Ok(());
        }
        match name {
            ".amdgcn_target" => {
                let target = unquote(args);
                self.module
                    .target_directive
                    .get_or_insert_with(|| target.into());
                set_target(&mut self.module, target, span)?;
            }
            ".amdhsa_code_object_version" => {
                let version = instr::integer(args)
                    .filter(|value| (4..=6).contains(value))
                    .ok_or_else(|| {
                        fail("only AMDHSA code-object versions 4, 5 and 6 are supported".into())
                    })? as u32;
                if self.module.code_object_version.replace(version).is_some() {
                    return Err(fail("duplicate code-object version".into()));
                }
            }
            ".amdhsa_kernel" => {
                if self.descriptor.is_some() {
                    return Err(fail("nested .amdhsa_kernel".into()));
                }
                if !valid_symbol(args) {
                    return Err(fail("malformed kernel descriptor name".into()));
                }
                let index = if let Some(index) = self
                    .module
                    .kernels
                    .iter()
                    .position(|kernel| kernel.name == args)
                {
                    index
                } else {
                    let index = self.module.kernels.len();
                    self.module.kernels.push(Kernel {
                        name: args.into(),
                        ..Kernel::default()
                    });
                    index
                };
                if !self.descriptors.insert(args.into()) {
                    return Err(fail(format!("duplicate descriptor for `{args}`")));
                }
                self.descriptor = Some(index);
            }
            ".end_amdhsa_kernel" => {
                if !args.is_empty() {
                    return Err(fail(".end_amdhsa_kernel does not take operands".into()));
                }
                if self.descriptor.take().is_none() {
                    return Err(fail("unmatched .end_amdhsa_kernel".into()));
                }
            }
            ".amdgpu_metadata" => {
                if !args.is_empty() {
                    return Err(fail(".amdgpu_metadata does not take operands".into()));
                }
                if self.module.metadata_version.is_some() {
                    return Err(fail("duplicate .amdgpu_metadata".into()));
                }
                self.metadata = Some(String::new());
                self.metadata_start = span;
            }
            ".text" => {
                self.finish_section(span)?;
                self.text_section = true;
            }
            ".data" | ".rodata" | ".bss" => {
                self.finish_section(span)?;
                self.text_section = false;
                self.data_read_only = name == ".rodata";
                self.data_section_supported = true;
                self.data_zero_fill = name == ".bss";
            }
            ".section" | ".pushsection" => {
                if name == ".pushsection" {
                    return Err(fail(
                        ".pushsection requires a section stack and is unsupported".into(),
                    ));
                }
                let mut parts = args.split(',');
                let section = parts.next().unwrap_or_default().trim_matches('"');
                self.finish_section(span)?;
                self.text_section = section == ".text" || section.starts_with(".text.");
                let flags = parts.next().unwrap_or_default().trim().trim_matches('"');
                if flags.contains('x') && !self.text_section {
                    return Err(fail("unsupported executable section".into()));
                }
                if (section == ".rodata" || section.starts_with(".rodata.")) && flags.contains('w')
                {
                    return Err(fail(
                        "writable flags on a read-only module data section are unsupported".into(),
                    ));
                }
                self.data_read_only = !flags.contains('w');
                self.data_section_supported = [".data", ".rodata", ".bss"].iter().any(|prefix| {
                    section == *prefix
                        || section
                            .strip_prefix(prefix)
                            .is_some_and(|rest| rest.starts_with('.'))
                });
                if self.data_section_supported {
                    let zero_fill = match parts.next().map(str::trim) {
                        None => self
                            .data_section_kinds
                            .get(section)
                            .copied()
                            .unwrap_or(section.starts_with(".bss.")),
                        Some("@nobits" | "%nobits") => true,
                        Some("@progbits" | "%progbits") => false,
                        _ => {
                            return Err(fail(
                                "unsupported module data section storage type".into(),
                            ));
                        }
                    };
                    if self
                        .data_section_kinds
                        .get(section)
                        .is_some_and(|previous| *previous != zero_fill)
                    {
                        return Err(fail(
                            "module data section storage type cannot change".into(),
                        ));
                    }
                    self.data_section_kinds.insert(section.into(), zero_fill);
                    self.data_zero_fill = zero_fill;
                }
            }
            ".type" => {
                let parts = instr::split_top_level(args, ',').map_err(fail)?;
                if parts.len() != 2 || !valid_symbol(parts[0]) {
                    return Err(fail("malformed .type directive".into()));
                }
                match parts[1] {
                    "@function" | "%function" => {
                        if self.globals.contains(parts[0]) {
                            return Err(fail("symbol is both function and object".into()));
                        }
                        self.functions.insert(parts[0].into());
                    }
                    "@object" | "%object" => {
                        if self.functions.contains(parts[0]) {
                            return Err(fail("symbol is both function and object".into()));
                        }
                        self.globals.insert(parts[0].into());
                    }
                    _ => return Err(fail(format!("unsupported .type kind `{}`", parts[1]))),
                }
            }
            ".size" => {
                let parts = instr::split_top_level(args, ',').map_err(fail)?;
                if parts.len() != 2 || !valid_symbol(parts[0]) || parts[1].is_empty() {
                    return Err(fail("malformed .size directive".into()));
                }
                if let Some(index) = self
                    .active
                    .filter(|index| self.module.kernels[*index].name == parts[0])
                {
                    self.closed_functions.insert(parts[0].into());
                    self.active = None;
                    self.module.kernels[index].size_directive = Some(super::SizeDirective {
                        expression: parts[1].into(),
                        span,
                    });
                    self.trailing_owner = Some(index);
                }
                if self.globals.contains(parts[0]) {
                    let index = self
                        .data_object
                        .ok_or_else(|| fail("object .size has no active definition".into()))?;
                    let object = &mut self.module.globals[index];
                    let size = instr::integer(parts[1])
                        .and_then(|value| u64::try_from(value).ok())
                        .filter(|size| *size > 0 && *size <= MAX_OBJECT_BYTES)
                        .ok_or_else(|| {
                            fail(
                                "module object size must be a positive integer at most16MiB".into(),
                            )
                        })?;
                    if object.symbol != parts[0] || object.initial_bytes.len() as u64 != size {
                        return Err(fail(
                            "module object .size must match its exact initialized bytes".into(),
                        ));
                    }
                    object.size = size;
                    self.data_object = None;
                }
            }
            ".set" | ".equ" | ".equiv" => {
                let parts = instr::split_top_level(args, ',').map_err(fail)?;
                if parts.len() != 2 || !valid_symbol(parts[0]) {
                    return Err(fail("malformed constant definition".into()));
                }
                // Unique definitions have immutable values, so forward references
                // can be resolved before instruction normalization. LLVM permits
                // .set/.equ reassignment with use-site snapshot semantics; that
                // requires a different representation and remains unsupported.
                // .equiv is nonredefinable even in LLVM. Never use the final value
                // of a reassigned symbol for its earlier instruction uses.
                if self.constant_definitions.contains_key(parts[0]) {
                    return Err(fail(format!(
                        "duplicate or reassigned assembly constant `{}` is unsupported",
                        parts[0]
                    )));
                }
                let expression = constant_expr::parse(parts[1]).map_err(fail)?;
                if let Ok(value) = expression.known_value(&self.module.constants) {
                    self.module.constants.insert(parts[0].into(), value);
                }
                self.constant_definitions
                    .insert(parts[0].into(), expression);
                self.constant_spans.insert(parts[0].into(), span);
            }
            // Linkage, debugging, unwind and producer annotations have no value,
            // control-flow, register, or memory effect in this analysis model.
            ".globl" | ".global" | ".protected" | ".hidden" | ".local" | ".weak"
            | ".addrsig_sym" => {
                if !valid_symbol(args) {
                    return Err(fail(format!("malformed `{name}` symbol")));
                }
            }
            ".file"
            | ".loc"
            | ".ident"
            | ".addrsig"
            | ".cfi_startproc"
            | ".cfi_endproc"
            | ".cfi_sections"
            | ".cfi_def_cfa"
            | ".cfi_offset"
            | ".cfi_def_cfa_offset"
            | ".cfi_def_cfa_register" => {}
            ".p2align" | ".p2alignl" | ".balign" | ".align" => {
                let parts = instr::split_top_level(args, ',').map_err(fail)?;
                if parts.is_empty()
                    || parts.len() > 3
                    || parts.iter().any(|part| instr::integer(part).is_none())
                {
                    return Err(fail(format!("malformed `{name}` alignment")));
                }
                self.record_trailing(name, args, span);
                if !self.text_section {
                    if parts.len() == 3 {
                        return Err(fail(
                            "conditional max-skip module-object alignment is unsupported".into(),
                        ));
                    }
                    if name == ".align" {
                        return Err(fail(
                            "module data requires explicit .p2align or .balign".into(),
                        ));
                    }
                    if self.data_object.is_some() {
                        return Err(fail(
                            "internal module-object alignment padding is unsupported".into(),
                        ));
                    }
                    let value = instr::integer(parts[0]).unwrap();
                    let alignment = if name.starts_with(".p2align") {
                        u32::try_from(value)
                            .ok()
                            .and_then(|shift| 1u64.checked_shl(shift))
                    } else {
                        u64::try_from(value).ok()
                    };
                    self.data_alignment = alignment
                        .filter(|alignment| {
                            alignment.is_power_of_two() && *alignment <= MAX_OBJECT_BYTES
                        })
                        .ok_or_else(|| fail("unsupported module-object alignment".into()))?;
                }
            }
            ".byte" | ".short" | ".hword" | ".long" | ".word" | ".quad" | ".zero" | ".space"
            | ".fill" | ".ascii" | ".asciz" | ".string" => {
                if self.text_section && self.active.is_some() {
                    return Err(fail(format!(
                        "raw data directive `{name}` inside executable kernel"
                    )));
                }
                if args.is_empty() {
                    return Err(fail(format!("missing data for `{name}`")));
                }
                self.record_trailing(name, args, span);
                if self.data_object.is_some() {
                    self.object_data(name, args, span)?;
                } else if name == ".asciz" {
                    return Err(fail(
                        "module .asciz requires an active declared object".into(),
                    ));
                }
            }
            _ => return Err(fail(format!("unsupported assembly directive `{name}`"))),
        }
        Ok(())
    }

    // Directives between a closing `.size` and the next function/section are
    // retained for callers that account for padding; they never execute.
    fn record_trailing(&mut self, name: &str, args: &str, span: SourceSpan) {
        if self.text_section
            && self.active.is_none()
            && let Some(index) = self.trailing_owner
        {
            self.module.kernels[index]
                .trailing_text_directives
                .push(super::TextDirective {
                    name: name.into(),
                    arguments: args.into(),
                    span,
                });
        }
    }

    fn finish_section(&mut self, span: SourceSpan) -> Result<(), ParseError> {
        self.trailing_owner = None;
        if self.data_object.is_some() {
            return Err(ParseError {
                message: "module object changes section before explicit .size".into(),
                span,
            });
        }
        self.data_alignment = 1;
        self.data_zero_fill = false;
        Ok(())
    }

    fn object_data(
        &mut self,
        directive: &str,
        args: &str,
        span: SourceSpan,
    ) -> Result<(), ParseError> {
        let fail = |message: &str| ParseError {
            message: message.into(),
            span,
        };
        if directive == ".asciz" {
            let bytes = nul_terminated_ascii(args).map_err(fail)?;
            return self.append_object_bytes(&bytes, span);
        }
        if directive == ".ascii" {
            // LLVM emits printable lookup-table bytes as .ascii. Keep this
            // profile exact: one quoted literal, no escapes or terminator.
            let literal = args
                .strip_prefix('"')
                .and_then(|text| text.strip_suffix('"'))
                .filter(|text| {
                    text.bytes()
                        .all(|byte| (0x20..=0x7e).contains(&byte) && !matches!(byte, b'\\' | b'"'))
                })
                .ok_or_else(|| {
                    fail("module .ascii requires one plain printable ASCII string without escapes")
                })?;
            return self.append_object_bytes(literal.as_bytes(), span);
        }
        let parts = instr::split_top_level(args, ',').map_err(|_| fail("malformed module data"))?;
        let numbers = parts
            .iter()
            .map(|part| instr::constant_expression(part, &self.module.constants))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                fail(
                    "module data requires exact integer bytes; relocations/strings are unsupported",
                )
            })?;
        let mut bytes = Vec::new();
        let width = match directive {
            ".byte" => Some(1),
            ".short" => Some(2),
            ".long" => Some(4),
            ".quad" => Some(8),
            _ => None,
        };
        if let Some(width) = width {
            for value in numbers {
                bytes.extend_from_slice(&(value as u64).to_le_bytes()[..width]);
            }
        } else if matches!(directive, ".zero" | ".space") && (1..=2).contains(&numbers.len()) {
            let count = u64::try_from(numbers[0])
                .ok()
                .filter(|count| *count <= MAX_OBJECT_BYTES)
                .ok_or_else(|| fail("module zero/space data exceeds bounded object envelope"))?;
            let fill = numbers.get(1).copied().unwrap_or(0);
            if !(0..=255).contains(&fill) {
                return Err(fail("module data fill must be one byte"));
            }
            bytes.resize(count as usize, fill as u8);
        } else if directive == ".fill" && (1..=3).contains(&numbers.len()) {
            let count = u64::try_from(numbers[0])
                .ok()
                .ok_or_else(|| fail("negative module fill count"))?;
            let width = numbers.get(1).copied().unwrap_or(1);
            if ![1, 2, 4].contains(&width) {
                return Err(fail("module fill size must be1,2or4"));
            }
            let size = count
                .checked_mul(width as u64)
                .filter(|size| *size <= MAX_OBJECT_BYTES)
                .ok_or_else(|| fail("module fill data exceeds bounded object envelope"))?;
            let value = numbers.get(2).copied().unwrap_or(0) as u64;
            bytes.reserve(size as usize);
            for _ in 0..count {
                bytes.extend_from_slice(&value.to_le_bytes()[..width as usize]);
            }
        } else {
            return Err(fail("unsupported module object data directive"));
        }
        self.append_object_bytes(&bytes, span)
    }

    fn append_object_bytes(&mut self, bytes: &[u8], span: SourceSpan) -> Result<(), ParseError> {
        if self.data_zero_fill && bytes.iter().any(|byte| *byte != 0) {
            return Err(ParseError {
                message: "NOBITS module data requires zero-filled bytes".into(),
                span,
            });
        }
        let object = &mut self.module.globals[self.data_object.unwrap()];
        if (object.initial_bytes.len() as u64)
            .checked_add(bytes.len() as u64)
            .is_none_or(|size| size > MAX_OBJECT_BYTES)
        {
            return Err(ParseError {
                message: "module object exceeds bounded16MiB data envelope".into(),
                span,
            });
        }
        object.initial_bytes.extend_from_slice(bytes);
        Ok(())
    }
}

/// One bounded assembler literal, including its exact terminating NUL. This
/// deliberately excludes the rest of LLVM's string/escape grammar rather than
/// treating an unknown escape or a second literal as bytes.
fn nul_terminated_ascii(args: &str) -> Result<Vec<u8>, &'static str> {
    if args.len() as u64 > MAX_OBJECT_BYTES + 1 {
        return Err("module .asciz literal exceeds bounded16MiB envelope");
    }
    let literal = args
        .strip_prefix('"')
        .and_then(|text| text.strip_suffix('"'))
        .ok_or("module .asciz requires one terminated quoted ASCII literal")?;
    let mut bytes = Vec::with_capacity(literal.len() + 1);
    let mut source = literal.bytes();
    while let Some(byte) = source.next() {
        let value = match byte {
            b'\\' => match source.next() {
                Some(b'\\') => b'\\',
                Some(b'"') => b'"',
                Some(b'n') => b'\n',
                Some(b'r') => b'\r',
                Some(b't') => b'\t',
                _ => return Err("unsupported or unterminated module .asciz escape"),
            },
            b'"' => return Err("module .asciz requires exactly one quoted literal"),
            0x20..=0x7e => byte,
            _ => return Err("module .asciz requires printable ASCII or supported escapes"),
        };
        bytes.push(value);
    }
    bytes.push(0);
    Ok(bytes)
}

pub(super) fn unquote(value: &str) -> &str {
    value
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .unwrap_or(value)
}

fn valid_symbol(symbol: &str) -> bool {
    !symbol.is_empty()
        && !symbol.starts_with(|c: char| c.is_ascii_digit())
        && symbol
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || "_.$@".contains(character))
}

pub(super) fn set_target(
    module: &mut Module,
    target: &str,
    span: SourceSpan,
) -> Result<(), ParseError> {
    // The `unknown` vendor-environment spelling is accepted only for gfx1250,
    // whose assembler emits it; other targets keep the empty-environment form.
    let base = target
        .strip_prefix("amdgcn-amd-amdhsa--")
        .or_else(|| {
            target
                .strip_prefix("amdgcn-amd-amdhsa-unknown-")
                .filter(|rest| rest.split(':').next() == Some("gfx1250"))
        })
        .unwrap_or(target);
    let gpu = base.split(':').next().unwrap_or(base);
    if !["gfx942", "gfx950", "gfx1250"].contains(&gpu) {
        return Err(ParseError {
            message: format!(
                "unsupported AMD target `{target}`; expected gfx942, gfx950 or gfx1250"
            ),
            span,
        });
    }
    let mut xnack = None;
    let mut sramecc = None;
    for feature in base.split(':').skip(1) {
        if !["xnack+", "xnack-", "sramecc+", "sramecc-"].contains(&feature) {
            return Err(ParseError {
                message: format!("unsupported target feature `{feature}`"),
                span,
            });
        }
        let selected = if feature.starts_with("xnack") {
            &mut xnack
        } else {
            &mut sramecc
        };
        if let Some(previous) = selected {
            let kind = if *previous == feature {
                "duplicate"
            } else {
                "conflicting"
            };
            return Err(ParseError {
                message: format!("{kind} target features `{previous}` and `{feature}`"),
                span,
            });
        }
        *selected = Some(feature);
    }
    if let Some(previous) = &module.target
        && previous != base
    {
        return Err(ParseError {
            message: format!("conflicting assembly and metadata targets `{previous}` and `{base}`"),
            span,
        });
    }
    module.target = Some(base.into());
    Ok(())
}

fn descriptor_fields() -> &'static [&'static str] {
    &[
        "group_segment_fixed_size",
        "private_segment_fixed_size",
        "kernarg_size",
        "user_sgpr_count",
        "user_sgpr_private_segment_buffer",
        "user_sgpr_dispatch_ptr",
        "user_sgpr_queue_ptr",
        "user_sgpr_kernarg_segment_ptr",
        "user_sgpr_dispatch_id",
        "user_sgpr_flat_scratch_init",
        "user_sgpr_private_segment_size",
        "user_sgpr_kernarg_preload_length",
        "user_sgpr_kernarg_preload_offset",
        "uses_dynamic_stack",
        "enable_private_segment",
        "system_sgpr_private_segment_wavefront_offset",
        "system_sgpr_workgroup_id_x",
        "system_sgpr_workgroup_id_y",
        "system_sgpr_workgroup_id_z",
        "system_sgpr_workgroup_info",
        "system_vgpr_workitem_id",
        "next_free_vgpr",
        "next_free_sgpr",
        "accum_offset",
        "reserve_vcc",
        "reserve_flat_scratch",
        "reserve_xnack_mask",
        "float_round_mode_32",
        "float_round_mode_16_64",
        "float_denorm_mode_32",
        "float_denorm_mode_16_64",
        "dx10_clamp",
        "ieee_mode",
        "fp16_overflow",
        "tg_split",
        "wavefront_size32",
        "named_barrier_count",
        "memory_ordered",
        "forward_progress",
        "inst_pref_size",
        "round_robin_scheduling",
        "exception_fp_ieee_invalid_op",
        "exception_fp_denorm_src",
        "exception_fp_ieee_div_zero",
        "exception_fp_ieee_overflow",
        "exception_fp_ieee_underflow",
        "exception_fp_ieee_inexact",
        "exception_int_div_zero",
    ]
}

pub(super) fn validate_descriptor(kernel: &Kernel) -> Result<(), ParseError> {
    let fail = |message| ParseError {
        message: format!("kernel `{}`: {message}", kernel.name),
        span: SourceSpan::default(),
    };
    for (key, value) in &kernel.descriptor.fields {
        if (key.starts_with("system_sgpr_")
            || key.starts_with("reserve_")
            || key.starts_with("exception_")
            || key.starts_with("user_sgpr_")
                && ![
                    "user_sgpr_count",
                    "user_sgpr_kernarg_preload_length",
                    "user_sgpr_kernarg_preload_offset",
                ]
                .contains(&key.as_str())
            || [
                "uses_dynamic_stack",
                "enable_private_segment",
                "dx10_clamp",
                "ieee_mode",
                "fp16_overflow",
                "tg_split",
                "wavefront_size32",
            ]
            .contains(&key.as_str()))
            && *value > 1
        {
            return Err(fail(format!("descriptor `{key}` must be zero or one")));
        }
        if key == "system_vgpr_workitem_id" && *value > 2 {
            return Err(fail("system_vgpr_workitem_id must be 0, 1 or 2".into()));
        }
        if (key.starts_with("float_round_mode_") || key.starts_with("float_denorm_mode_"))
            && *value > 3
        {
            return Err(fail(format!("descriptor `{key}` must fit in two bits")));
        }
        // Operand namespaces stop at255, but this descriptor counts the
        // unified VGPR+AGPR physical allocation on both CDNA3 and CDNA4.
        if key == "next_free_vgpr" && *value > 1024 {
            return Err(fail(
                "next_free_vgpr exceeds the unified native AMD allocation limit".into(),
            ));
        }
        if key == "next_free_sgpr" && *value > 106 {
            return Err(fail(
                "next_free_sgpr exceeds the native AMD register range".into(),
            ));
        }
    }
    fn register(operand: &super::Operand) -> Option<super::Register> {
        match operand {
            super::Operand::Register(register) => Some(*register),
            super::Operand::Modified { value, .. } => register(value),
            _ => None,
        }
    }
    let registers = kernel
        .instructions
        .iter()
        .flat_map(|instruction| instruction.operands.iter())
        .filter_map(register)
        .collect::<Vec<_>>();
    let has_accumulators = registers
        .iter()
        .any(|register| register.class == super::RegisterClass::Accumulator);
    let accum_offset = if has_accumulators {
        let offset = kernel.descriptor.get("accum_offset").ok_or_else(|| fail("explicit AGPRs require an accum_offset descriptor to establish unified-register aliases".into()))?;
        if offset == 0 || offset > 256 || !offset.is_multiple_of(4) {
            return Err(fail(
                "accum_offset must be a multiple of four in 4..=256".into(),
            ));
        }
        Some(offset)
    } else {
        None
    };
    for register in registers {
        let end = u64::from(register.index) + u64::from(register.count);
        match register.class {
            super::RegisterClass::Scalar => {
                if kernel
                    .descriptor
                    .get("next_free_sgpr")
                    .is_some_and(|count| end > count)
                {
                    return Err(fail(format!(
                        "SGPR operand s{}..{} exceeds next_free_sgpr",
                        register.index,
                        end - 1
                    )));
                }
            }
            super::RegisterClass::Vector => {
                if kernel
                    .descriptor
                    .get("next_free_vgpr")
                    .is_some_and(|count| end > count)
                {
                    return Err(fail(format!(
                        "VGPR operand v{}..{} exceeds next_free_vgpr",
                        register.index,
                        end - 1
                    )));
                }
                if accum_offset.is_some_and(|offset| end > offset) {
                    return Err(fail("explicit VGPR aliases an AGPR physical allocation; overlapping unified register names are unsupported".into()));
                }
            }
            super::RegisterClass::Accumulator => {
                if let (Some(offset), Some(count)) =
                    (accum_offset, kernel.descriptor.get("next_free_vgpr"))
                    && offset + end > count
                {
                    return Err(fail(
                        "AGPR operand exceeds unified next_free_vgpr allocation".into(),
                    ));
                }
            }
            super::RegisterClass::Special => {}
            super::RegisterClass::Temporary => {}
        }
    }
    if kernel.descriptor.flag("tg_split")
        && kernel.instructions.iter().any(|instruction| {
            matches!(
                instruction.opcode,
                super::Opcode::Lds | super::Opcode::Barrier
            )
        })
    {
        return Err(fail(
            "tg_split kernels cannot use LDS or workgroup barriers".into(),
        ));
    }
    Ok(())
}

/// Used by metadata consistency checks without exposing parser implementation.
pub(super) fn merge_descriptor_fields(
    kernel: &mut Kernel,
    fields: BTreeMap<String, u64>,
    span: SourceSpan,
) -> Result<(), ParseError> {
    let private_sources = kernel
        .descriptor
        .get("private_segment_fixed_size")
        .zip(fields.get("private_segment_fixed_size").copied());
    for (name, value) in fields {
        if kernel.descriptor.get(&name).is_some_and(|old| old != value) {
            return Err(ParseError {
                message: format!(
                    "metadata disagrees with `{}` descriptor field `{name}`",
                    kernel.name
                ),
                span,
            });
        }
        kernel.descriptor.fields.entry(name).or_insert(value);
    }
    kernel.descriptor.private_segment_fixed_size_sources = private_sources;
    Ok(())
}
