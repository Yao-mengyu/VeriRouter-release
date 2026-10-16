use plonky2::iop::target::{BoolTarget, Target};
use plonky2::plonk::circuit_builder::CircuitBuilder;

use crate::util::{D, F};

/// Constrain a field element to the canonical unsigned `bit_width` range.
///
/// Widths are deliberately capped below 63 bits. In the Goldilocks field,
/// accepting a full 64-bit range makes a negative difference look like a
/// large positive field element. All VeriRouter order domains fit in at most
/// 62 bits.
pub fn constrain_bits_le(builder: &mut CircuitBuilder<F, D>, value: Target, bit_width: usize) {
    assert!(
        (1..=62).contains(&bit_width),
        "order range width must be between 1 and 62 bits"
    );
    builder.range_check(value, bit_width);
}

pub(crate) fn constrain_bits_if(
    builder: &mut CircuitBuilder<F, D>,
    gate: BoolTarget,
    value: Target,
    bit_width: usize,
) {
    let gated_value = builder.mul(value, gate.target);
    constrain_bits_le(builder, gated_value, bit_width);
}

pub fn assert_nonnegative_diff_if(
    builder: &mut CircuitBuilder<F, D>,
    gate: BoolTarget,
    lhs: Target,
    rhs: Target,
    bit_width: usize,
) -> Target {
    constrain_bits_if(builder, gate, lhs, bit_width);
    constrain_bits_if(builder, gate, rhs, bit_width);
    let diff = builder.sub(lhs, rhs);
    let gated_diff = builder.mul(diff, gate.target);
    constrain_bits_le(builder, gated_diff, bit_width);
    gated_diff
}

/// Assert `lhs >= rhs + offset` for a Boolean offset while range-checking the
/// original operands. Checking `rhs + offset` instead would be unsafe because
/// a maximum field element could wrap to zero before its range check.
pub fn assert_nonnegative_diff_with_offset_if(
    builder: &mut CircuitBuilder<F, D>,
    gate: BoolTarget,
    lhs: Target,
    rhs: Target,
    offset: BoolTarget,
    bit_width: usize,
) -> Target {
    constrain_bits_if(builder, gate, lhs, bit_width);
    constrain_bits_if(builder, gate, rhs, bit_width);
    let diff = builder.sub(lhs, rhs);
    let diff_with_offset = builder.sub(diff, offset.target);
    let gated_diff = builder.mul(diff_with_offset, gate.target);
    constrain_bits_le(builder, gated_diff, bit_width);
    gated_diff
}

pub fn assert_positive_diff_if(
    builder: &mut CircuitBuilder<F, D>,
    gate: BoolTarget,
    lhs: Target,
    rhs: Target,
    bit_width: usize,
) -> Target {
    constrain_bits_if(builder, gate, lhs, bit_width);
    constrain_bits_if(builder, gate, rhs, bit_width);
    let diff = builder.sub(lhs, rhs);
    let one = builder.one();
    let diff_minus_one = builder.sub(diff, one);
    let gated_diff = builder.mul(diff_minus_one, gate.target);
    constrain_bits_le(builder, gated_diff, bit_width);
    gated_diff
}

/// Assert `lhs > rhs` while reusing operand range constraints established
/// elsewhere in the same circuit.
///
/// This helper constrains only the gated difference. It is sound only when the
/// caller has already established `gate => lhs,rhs < 2^bit_width`; callers
/// should prefer [`assert_positive_diff_if`] unless that proof obligation is
/// explicit at the call site.
pub(crate) fn assert_positive_diff_if_bounded_operands(
    builder: &mut CircuitBuilder<F, D>,
    gate: BoolTarget,
    lhs: Target,
    rhs: Target,
    bit_width: usize,
) -> Target {
    let diff = builder.sub(lhs, rhs);
    let one = builder.one();
    let diff_minus_one = builder.sub(diff, one);
    let gated_diff = builder.mul(diff_minus_one, gate.target);
    constrain_bits_le(builder, gated_diff, bit_width);
    gated_diff
}

/// Assert `lhs >= rhs` while reusing operand range constraints established
/// elsewhere. Same proof obligation as
/// [`assert_positive_diff_if_bounded_operands`].
pub(crate) fn assert_nonnegative_diff_if_bounded_operands(
    builder: &mut CircuitBuilder<F, D>,
    gate: BoolTarget,
    lhs: Target,
    rhs: Target,
    bit_width: usize,
) -> Target {
    let diff = builder.sub(lhs, rhs);
    let gated_diff = builder.mul(diff, gate.target);
    constrain_bits_le(builder, gated_diff, bit_width);
    gated_diff
}

/// Assert `lhs >= rhs + offset` for a Boolean offset, reusing operand range
/// constraints established elsewhere (`gate => lhs,rhs < 2^bit_width`).
/// With bounded operands the integer `lhs - rhs - offset` lies in
/// `[-2^bit_width, 2^bit_width)`, and a negative value is a field element
/// above `p - 2^bit_width > 2^bit_width`, so the gated range check rejects it.
pub(crate) fn assert_nonnegative_diff_with_offset_if_bounded_operands(
    builder: &mut CircuitBuilder<F, D>,
    gate: BoolTarget,
    lhs: Target,
    rhs: Target,
    offset: BoolTarget,
    bit_width: usize,
) -> Target {
    let diff = builder.sub(lhs, rhs);
    let diff_with_offset = builder.sub(diff, offset.target);
    let gated_diff = builder.mul(diff_with_offset, gate.target);
    constrain_bits_le(builder, gated_diff, bit_width);
    gated_diff
}
