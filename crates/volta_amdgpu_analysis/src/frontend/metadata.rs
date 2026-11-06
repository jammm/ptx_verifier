//! LLVM AMDHSA YAML metadata reader. YAML is parsed by a real parser; unknown
//! semantic keys, overlapping argument slots and target conflicts are errors.

use std::collections::{BTreeMap, BTreeSet};

use serde_yaml::{Mapping, Value};

use super::{KernelArg, Module, ParseError, SourceSpan, parse};

fn untag(value: &Value) -> &Value {
    match value {
        Value::Tagged(tagged) => untag(&tagged.value),
        _ => value,
    }
}

fn get<'a>(mapping: &'a Mapping, name: &str) -> Option<&'a Value> {
    mapping.get(Value::String(name.into())).map(untag)
}

fn string(value: &Value) -> Option<String> {
    match untag(value) {
        Value::String(value) => Some(value.clone()),
        _ => None,
    }
}

fn number(value: &Value) -> Option<u64> {
    untag(value).as_u64()
}

pub(super) fn apply_metadata(
    module: &mut Module,
    yaml: &str,
    span: SourceSpan,
) -> Result<BTreeSet<String>, ParseError> {
    let fail = |message| ParseError { message, span };
    let document: Value = serde_yaml::from_str(yaml).map_err(|error| {
        let location = error.location();
        ParseError {
            message: format!("malformed AMDGPU metadata: {error}"),
            span: SourceSpan {
                line: span.line + location.as_ref().map_or(0, |location| location.line()),
                column: location.map_or(span.column, |location| location.column()),
                ..span
            },
        }
    })?;
    let root = document
        .as_mapping()
        .ok_or_else(|| fail("AMDGPU metadata must be a mapping".into()))?;
    for key in root.keys() {
        let key = key
            .as_str()
            .ok_or_else(|| fail("metadata key must be a string".into()))?;
        if ![
            "amdhsa.kernels",
            "amdhsa.target",
            "amdhsa.version",
            "amdhsa.printf",
        ]
        .contains(&key)
        {
            return Err(fail(format!("unsupported AMDGPU metadata key `{key}`")));
        }
    }
    let version = get(root, "amdhsa.version")
        .and_then(Value::as_sequence)
        .ok_or_else(|| fail("missing amdhsa.version metadata".into()))?;
    let version = version
        .iter()
        .map(|entry| number(entry).and_then(|number| u32::try_from(number).ok()))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| fail("invalid metadata version".into()))?;
    if version.len() != 2 || version[0] != 1 || version[1] > 2 {
        return Err(fail(format!(
            "unsupported AMDGPU metadata version {version:?}"
        )));
    }
    module.metadata_version = Some(version);
    if let Some(target) = get(root, "amdhsa.target") {
        let target = string(target).ok_or_else(|| fail("amdhsa.target must be a string".into()))?;
        module.metadata_target = Some(target.clone());
        parse::set_target(module, &target, span)?;
    }
    let wave_size = module
        .target
        .as_deref()
        .and_then(crate::target::Architecture::from_target)
        .map_or(64, |target| target.wave_size() as u64);
    let kernels = get(root, "amdhsa.kernels")
        .and_then(Value::as_sequence)
        .ok_or_else(|| fail("missing amdhsa.kernels metadata".into()))?;
    let mut seen_kernels = BTreeSet::new();
    for entry in kernels {
        let entry = entry
            .as_mapping()
            .ok_or_else(|| fail("kernel metadata must be a mapping".into()))?;
        for key in entry.keys() {
            let key = key
                .as_str()
                .ok_or_else(|| fail("kernel metadata key must be a string".into()))?;
            if !kernel_keys().contains(&key) {
                return Err(fail(format!("unsupported kernel metadata key `{key}`")));
            }
        }
        let name = get(entry, ".name")
            .and_then(string)
            .ok_or_else(|| fail("kernel metadata requires .name".into()))?;
        if !seen_kernels.insert(name.clone()) {
            return Err(fail(format!("duplicate metadata for kernel `{name}`")));
        }
        let kernel = module
            .kernels
            .iter_mut()
            .find(|kernel| kernel.name == name)
            .ok_or_else(|| fail(format!("metadata refers to undefined kernel `{name}`")))?;
        if let Some(wavefront_size) = get(entry, ".wavefront_size")
            && number(wavefront_size) != Some(wave_size)
        {
            return Err(fail(format!(
                "kernel `{name}` must use wavefront_size {wave_size}"
            )));
        }
        if let Some(symbol) = get(entry, ".symbol")
            && string(symbol).as_deref() != Some(&format!("{name}.kd"))
        {
            return Err(fail(format!(
                "kernel `{name}` has a mismatched descriptor symbol"
            )));
        }
        let mut fields = BTreeMap::new();
        for (key, descriptor_key) in [
            (".group_segment_fixed_size", "group_segment_fixed_size"),
            (".private_segment_fixed_size", "private_segment_fixed_size"),
            (".kernarg_segment_size", "kernarg_size"),
        ] {
            if let Some(value) = get(entry, key) {
                fields.insert(
                    descriptor_key.into(),
                    number(value)
                        .ok_or_else(|| fail(format!("`{key}` must be a nonnegative integer")))?,
                );
            }
        }
        if let Some(alignment) = get(entry, ".kernarg_segment_align") {
            let alignment = number(alignment)
                .filter(|value| value.is_power_of_two())
                .ok_or_else(|| {
                    fail("kernarg_segment_align must be a nonzero power of two".into())
                })?;
            fields.insert("kernarg_segment_align".into(), alignment);
        }
        if let Some(size) = get(entry, ".max_flat_workgroup_size") {
            fields.insert(
                "max_flat_workgroup_size".into(),
                number(size)
                    .filter(|value| (1..=1024).contains(value))
                    .ok_or_else(|| fail("max_flat_workgroup_size must be 1..=1024".into()))?,
            );
        }
        if let Some(required) = get(entry, ".reqd_workgroup_size") {
            let required = required
                .as_sequence()
                .filter(|required| required.len() == 3)
                .ok_or_else(|| fail("reqd_workgroup_size requires three dimensions".into()))?;
            let mut total = 1_u64;
            for (axis, value) in ["x", "y", "z"].iter().zip(required) {
                let value = number(value)
                    .filter(|value| (1..=1024).contains(value))
                    .ok_or_else(|| fail("required workgroup dimension must be 1..=1024".into()))?;
                total = total
                    .checked_mul(value)
                    .ok_or_else(|| fail("required workgroup size overflows".into()))?;
                fields.insert(format!("reqd_workgroup_size_{axis}"), value);
            }
            if total > 1024 {
                return Err(fail(
                    "required workgroup size exceeds the native AMD limit".into(),
                ));
            }
        }
        if let Some(uniform) = get(entry, ".uniform_work_group_size") {
            let uniform = number(uniform)
                .filter(|value| *value <= 1)
                .ok_or_else(|| fail("uniform_work_group_size must be zero or one".into()))?;
            fields.insert("uniform_work_group_size".into(), uniform);
        }
        if let Some(dynamic_stack) = get(entry, ".uses_dynamic_stack") {
            fields.insert(
                "uses_dynamic_stack".into(),
                u64::from(
                    untag(dynamic_stack)
                        .as_bool()
                        .ok_or_else(|| fail("uses_dynamic_stack must be a boolean".into()))?,
                ),
            );
        }
        parse::merge_descriptor_fields(kernel, fields, span)?;
        if let Some(args) = get(entry, ".args") {
            let args = args
                .as_sequence()
                .ok_or_else(|| fail(".args metadata must be a list".into()))?;
            let mut parsed_args = Vec::new();
            for arg in args {
                parsed_args.push(parse_arg(arg, span)?);
            }
            parsed_args.sort_by_key(|arg| arg.offset);
            for pair in parsed_args.windows(2) {
                if pair[0].offset + pair[0].size > pair[1].offset {
                    return Err(fail(format!(
                        "overlapping argument slots in kernel `{name}`"
                    )));
                }
            }
            if let Some(size) = kernel.descriptor.get("kernarg_size")
                && parsed_args.iter().any(|arg| {
                    arg.offset
                        .checked_add(arg.size)
                        .is_none_or(|end| end > size)
                })
            {
                return Err(fail(format!(
                    "argument outside kernarg segment in kernel `{name}`"
                )));
            }
            kernel.args = parsed_args;
        }
    }
    Ok(seen_kernels)
}

fn parse_arg(value: &Value, span: SourceSpan) -> Result<KernelArg, ParseError> {
    let fail = |message| ParseError { message, span };
    let arg = value
        .as_mapping()
        .ok_or_else(|| fail("argument metadata must be a mapping".into()))?;
    for key in arg.keys() {
        let key = key
            .as_str()
            .ok_or_else(|| fail("argument metadata key must be a string".into()))?;
        if ![
            ".name",
            ".offset",
            ".size",
            ".value_kind",
            ".address_space",
            ".value_type",
            ".type_name",
            ".access",
            ".actual_access",
            ".is_const",
            ".is_restrict",
            ".is_volatile",
            ".is_pipe",
            ".pointee_align",
        ]
        .contains(&key)
        {
            return Err(fail(format!("unsupported argument metadata key `{key}`")));
        }
    }
    let offset = get(arg, ".offset")
        .and_then(number)
        .ok_or_else(|| fail("argument metadata requires integer .offset".into()))?;
    let size = get(arg, ".size")
        .and_then(number)
        .filter(|size| *size != 0)
        .ok_or_else(|| fail("argument metadata requires nonzero .size".into()))?;
    offset
        .checked_add(size)
        .ok_or_else(|| fail("argument slot overflows address space".into()))?;
    let value_kind = get(arg, ".value_kind")
        .and_then(string)
        .ok_or_else(|| fail("argument metadata requires .value_kind".into()))?;
    if !value_kinds().contains(&value_kind.as_str()) {
        return Err(fail(format!(
            "unsupported argument value_kind `{value_kind}`"
        )));
    }
    let optional_string = |key: &str| -> Result<Option<String>, ParseError> {
        get(arg, key)
            .map(|value| {
                string(value).ok_or_else(|| fail(format!("argument `{key}` must be a string")))
            })
            .transpose()
    };
    let address_space = optional_string(".address_space")?;
    if address_space.as_ref().is_some_and(|space| {
        ![
            "private", "global", "constant", "local", "generic", "region",
        ]
        .contains(&space.as_str())
    }) {
        return Err(fail("unknown argument address_space".into()));
    }
    if let Some(alignment) = get(arg, ".pointee_align")
        && !number(alignment).is_some_and(u64::is_power_of_two)
    {
        return Err(fail("pointee_align must be a nonzero power of two".into()));
    }
    for key in [".is_const", ".is_restrict", ".is_volatile", ".is_pipe"] {
        if let Some(value) = get(arg, key)
            && untag(value).as_bool().is_none()
        {
            return Err(fail(format!("argument `{key}` must be a boolean")));
        }
    }
    Ok(KernelArg {
        name: optional_string(".name")?,
        offset,
        size,
        value_kind,
        address_space,
        value_type: optional_string(".value_type")?,
    })
}

fn kernel_keys() -> &'static [&'static str] {
    &[
        ".name",
        ".symbol",
        ".args",
        ".agpr_count",
        ".group_segment_fixed_size",
        ".kernarg_segment_align",
        ".kernarg_segment_size",
        ".language",
        ".language_version",
        ".max_flat_workgroup_size",
        ".private_segment_fixed_size",
        ".sgpr_count",
        ".sgpr_spill_count",
        ".uniform_work_group_size",
        ".uses_dynamic_stack",
        ".vgpr_count",
        ".vgpr_spill_count",
        ".wavefront_size",
        ".reqd_workgroup_size",
        ".workgroup_size_hint",
        ".vec_type_hint",
        ".device_enqueue_symbol",
        ".workgroup_processor_mode",
        ".memory_ordered",
        ".forward_progress",
    ]
}

fn value_kinds() -> &'static [&'static str] {
    &[
        "by_value",
        "global_buffer",
        "dynamic_shared_pointer",
        "sampler",
        "image",
        "pipe",
        "queue",
        "hidden_global_offset_x",
        "hidden_global_offset_y",
        "hidden_global_offset_z",
        "hidden_none",
        "hidden_printf_buffer",
        "hidden_default_queue",
        "hidden_completion_action",
        "hidden_multigrid_sync_arg",
        "hidden_hostcall_buffer",
        "hidden_heap_v1",
        "hidden_block_count_x",
        "hidden_block_count_y",
        "hidden_block_count_z",
        "hidden_group_size_x",
        "hidden_group_size_y",
        "hidden_group_size_z",
        "hidden_remainder_x",
        "hidden_remainder_y",
        "hidden_remainder_z",
        "hidden_grid_dims",
        "hidden_dynamic_lds_size",
        "hidden_private_base",
        "hidden_shared_base",
        "hidden_queue_ptr",
        "hidden_runtime_spare",
    ]
}
