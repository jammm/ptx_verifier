//! Exact binary32 bits of a WMMA F32 result element in the exact-representable
//! regime: `c + sum(a_k * b_k)` over normal (or zero) inputs, where every
//! partial sum in any order and grouping is an exactly representable normal
//! binary32 value. Then accumulation order, rounding mode, denormal handling
//! and any adder width of at least binary32 cannot change the result bits.
use rug::{Integer, Rational};

/// Smallest normal binary32 exponent.
const MIN_NORMAL_EXP: i64 = -126;
const SIGNIFICAND_BITS: i64 = 24;

/// Exponent of the lowest set bit of a nonzero dyadic rational, or `None`
/// when the denominator is not a power of two.
fn lowest_bit_exponent(value: &Rational) -> Option<i64> {
    let denominator = value.denom();
    if !denominator.is_power_of_two() {
        return None;
    }
    let denominator_exp = i64::from(denominator.significant_bits()) - 1;
    let numerator = Integer::from(value.numer().abs_ref());
    let numerator_exp = i64::from(numerator.find_one(0)?);
    Some(numerator_exp - denominator_exp)
}

/// `2^exp` as an exact rational.
fn power_of_two(exp: i64) -> Rational {
    let mut value = Rational::from(1);
    if exp >= 0 {
        value <<= exp as u32;
    } else {
        value >>= (-exp) as u32;
    }
    value
}

/// True when a nonzero input is at least the smallest normal binary32
/// magnitude (BF16 and F32 share the binary32 exponent range).
pub(super) fn normal_or_zero(value: &Rational) -> bool {
    normal_or_zero_from(value, MIN_NORMAL_EXP)
}

/// True when a nonzero input is at least `2^min_exp`, the smallest normal
/// magnitude of its own format (F16 -14, FP8 E4M3 -6).
pub(super) fn normal_or_zero_from(value: &Rational, min_exp: i64) -> bool {
    *value == 0 || Rational::from(value.abs_ref()) >= power_of_two(min_exp)
}

/// The exact result bits of `accumulator + sum(products)`, or `None` when
/// the element lies outside the exact-representable regime: a non-dyadic or
/// too small term, a magnitude sum of at least `2^24` times the common
/// quantum, or a zero result (whose sign depends on rounding and order).
pub(super) fn exact_binary32(accumulator: &Rational, products: &[Rational]) -> Option<u32> {
    let terms = || std::iter::once(accumulator).chain(products);
    let mut quantum: Option<i64> = None;
    for term in terms().filter(|term| **term != 0) {
        let exp = lowest_bit_exponent(term)?;
        quantum = Some(quantum.map_or(exp, |q| q.min(exp)));
    }
    let quantum = quantum?;
    if quantum < MIN_NORMAL_EXP {
        return None;
    }
    let mut magnitude = Rational::new();
    let mut sum = Rational::new();
    for term in terms() {
        magnitude += Rational::from(term.abs_ref());
        sum += term;
    }
    if magnitude >= power_of_two(quantum + SIGNIFICAND_BITS) || sum == 0 {
        return None;
    }
    // |sum| = m * 2^quantum with 0 < m < 2^24: exact in f64 and in f32.
    let value = sum.to_f64();
    let narrowed = value as f32;
    (f64::from(narrowed) == value && Rational::from_f64(value).as_ref() == Some(&sum))
        .then(|| narrowed.to_bits())
}

/// The exact F16 bits of `accumulator + sum(products)` when every partial
/// sum in any order and grouping is exactly representable in F16 (11-bit
/// significand, normal or zero): then any accumulator of at least F16
/// precision gives these bits. `None` otherwise, or for a zero result.
pub(super) fn exact_binary16(accumulator: &Rational, products: &[Rational]) -> Option<u16> {
    let terms = || std::iter::once(accumulator).chain(products);
    let mut quantum: Option<i64> = None;
    for term in terms().filter(|term| **term != 0) {
        let exp = lowest_bit_exponent(term)?;
        quantum = Some(quantum.map_or(exp, |q| q.min(exp)));
    }
    let quantum = quantum?;
    if quantum < -14 {
        return None;
    }
    let mut magnitude = Rational::new();
    let mut sum = Rational::new();
    for term in terms() {
        magnitude += Rational::from(term.abs_ref());
        sum += term;
    }
    if magnitude >= power_of_two(quantum + 11) || sum == 0 {
        return None;
    }
    f16_bits(sum.to_f64())
}

/// F16 bits of a value exactly representable as a normal F16, else `None`.
pub(super) fn f16_bits(value: f64) -> Option<u16> {
    if value == 0.0 || !value.is_finite() {
        return (value == 0.0).then_some(if value.is_sign_negative() { 0x8000 } else { 0 });
    }
    let sign = if value < 0.0 { 0x8000u16 } else { 0 };
    let magnitude = value.abs();
    let exponent = magnitude.log2().floor() as i32;
    if !(-14..=15).contains(&exponent) {
        return None;
    }
    let mantissa = magnitude / 2f64.powi(exponent) - 1.0;
    let scaled = mantissa * 1024.0;
    (scaled.fract() == 0.0 && scaled < 1024.0)
        .then(|| sign | (((exponent + 15) as u16) << 10) | scaled as u16)
}

/// The value of F16 bits; `None` for Inf/NaN.
pub(super) fn f16_value(bits: u16) -> Option<(f64, bool)> {
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = f64::from(bits & 0x3ff);
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    match exponent {
        0x1f => None,
        0 => Some((sign * mantissa * 2f64.powi(-24), mantissa == 0.0)),
        e => Some((
            sign * (1.0 + mantissa / 1024.0) * 2f64.powi(i32::from(e) - 15),
            true,
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(value: f64) -> Rational {
        Rational::from_f64(value).unwrap()
    }

    #[test]
    fn integers_below_two_to_the_twenty_four_are_exact() {
        let c = r(8_388_608.0 + 5.0);
        let products = vec![r(2.0); 32];
        assert_eq!(
            exact_binary32(&c, &products),
            Some((8_388_608.0f32 + 69.0).to_bits())
        );
    }

    #[test]
    fn magnitude_at_two_to_the_twenty_four_is_refused() {
        let c = r(16_777_216.0 - 1.0);
        assert_eq!(exact_binary32(&c, &[r(1.0)]), None);
        // The result alone is representable, but a partial sum is not.
        assert_eq!(exact_binary32(&r(16_777_216.0), &[r(1.0), r(-1.0)]), None);
    }

    #[test]
    fn mixed_quanta_use_the_finest_bit() {
        // 2^23 + 0.5 needs 25 significant bits.
        assert_eq!(exact_binary32(&r(8_388_608.0), &[r(0.5)]), None);
        assert_eq!(
            exact_binary32(&r(1024.0), &[r(0.5), r(0.25)]),
            Some(1024.75f32.to_bits())
        );
    }

    #[test]
    fn zero_and_tiny_results_are_refused() {
        assert_eq!(exact_binary32(&r(1.0), &[r(-1.0)]), None);
        assert_eq!(exact_binary32(&r(0.0), &[r(0.0)]), None);
        let tiny = r(f64::from(f32::MIN_POSITIVE) / 2.0);
        assert_eq!(exact_binary32(&tiny, &[]), None);
        assert!(!normal_or_zero(&tiny));
        assert!(normal_or_zero(&r(0.0)) && normal_or_zero(&r(-1.0)));
    }

    #[test]
    fn format_normal_bounds() {
        // F16 normal from 2^-14, FP8 E4M3 normal from 2^-6.
        assert!(normal_or_zero_from(&r(1.0 / 16384.0), -14));
        assert!(!normal_or_zero_from(&r(1.0 / 32768.0), -14));
        assert!(normal_or_zero_from(&r(1.0 / 64.0), -6));
        assert!(!normal_or_zero_from(&r(1.0 / 128.0), -6));
        assert!(normal_or_zero_from(&r(0.0), -6));
        // 128 FP8 products of 2 on C = 2^23 + 3 are exact.
        assert_eq!(
            exact_binary32(&r(8_388_611.0), &vec![r(2.0); 128]),
            Some((8_388_611.0f32 + 256.0).to_bits())
        );
    }

    #[test]
    fn binary16_regime() {
        // 512 + 64 products of 2: 640, exact; 1501 + 600 = 2101 needs 12 bits.
        assert_eq!(exact_binary16(&r(512.0), &vec![r(2.0); 64]), Some(0x6100));
        assert_eq!(exact_binary16(&r(1501.0), &[r(600.0)]), None);
        assert_eq!(exact_binary16(&r(512.0), &[r(0.5)]), Some(0x6001));
        assert_eq!(exact_binary16(&r(1024.0), &[r(0.5)]), None);
        assert_eq!(f16_bits(1023.0), Some(0x63fe));
        assert_eq!(f16_bits(1.5), Some(0x3e00));
        assert_eq!(f16_bits(1.0 / 3.0), None);
        assert_eq!(f16_value(0x6000), Some((512.0, true)));
        assert_eq!(f16_value(0x0001).map(|v| v.1), Some(false));
        assert_eq!(f16_value(0x7c00), None);
    }

    #[test]
    fn negative_results_keep_their_sign() {
        assert_eq!(
            exact_binary32(&r(-8.0), &[r(2.0), r(-4.0)]),
            Some((-10.0f32).to_bits())
        );
    }
}
