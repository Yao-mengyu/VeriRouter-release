use anyhow::Result;
use plonky2::field::extension::Extendable;
use plonky2::field::packed::PackedField;
use plonky2::gates::gate::Gate;
use plonky2::gates::packed_util::PackedEvaluableBase;
use plonky2::gates::util::StridedConstraintConsumer;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use plonky2::iop::target::Target;
use plonky2::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{CircuitConfig, CommonCircuitData};
use plonky2::plonk::vars::{
    EvaluationTargets, EvaluationVars, EvaluationVarsBase, EvaluationVarsBaseBatch,
    EvaluationVarsBasePacked,
};
use plonky2::util::serialization::{Buffer, IoResult, Read, Write};

#[derive(Clone, Debug)]
pub struct DotProductGate {
    pub num_lanes: usize,
    pub products_per_lane: usize,
}

impl DotProductGate {
    pub const fn new(num_lanes: usize, products_per_lane: usize) -> Self {
        Self {
            num_lanes,
            products_per_lane,
        }
    }

    pub const fn new_from_config(config: &CircuitConfig) -> Self {
        let num_lanes = 4;
        let products_per_lane = (config.num_routed_wires - 2 * num_lanes) / (num_lanes + 1);
        Self::new(num_lanes, products_per_lane)
    }

    const fn wire_acc_in(&self, lane: usize) -> usize {
        lane
    }

    const fn wire_acc_out(&self, lane: usize) -> usize {
        self.num_lanes + lane
    }

    const fn start_shared(&self) -> usize {
        2 * self.num_lanes
    }

    const fn wire_shared(&self, i: usize) -> usize {
        self.start_shared() + i
    }

    const fn start_right(&self) -> usize {
        self.start_shared() + self.products_per_lane
    }

    const fn wire_right(&self, lane: usize, i: usize) -> usize {
        self.start_right() + lane * self.products_per_lane + i
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for DotProductGate {
    fn id(&self) -> String {
        format!("{self:?}")
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.num_lanes)?;
        dst.write_usize(self.products_per_lane)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        Ok(Self::new(src.read_usize()?, src.read_usize()?))
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let mut constraints = Vec::with_capacity(self.num_lanes);
        for lane in 0..self.num_lanes {
            let mut sum = vars.local_wires[self.wire_acc_in(lane)];
            for i in 0..self.products_per_lane {
                sum += vars.local_wires[self.wire_shared(i)]
                    * vars.local_wires[self.wire_right(lane, i)];
            }
            constraints.push(vars.local_wires[self.wire_acc_out(lane)] - sum);
        }
        constraints
    }

    fn eval_unfiltered_base_one(
        &self,
        _vars: EvaluationVarsBase<F>,
        _yield_constr: StridedConstraintConsumer<F>,
    ) {
        panic!("use eval_unfiltered_base_batch");
    }

    fn eval_unfiltered_base_batch(&self, vars_base: EvaluationVarsBaseBatch<F>) -> Vec<F> {
        self.eval_unfiltered_base_batch_packed(vars_base)
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let mut constraints = Vec::with_capacity(self.num_lanes);
        for lane in 0..self.num_lanes {
            let mut sum = vars.local_wires[self.wire_acc_in(lane)];
            for i in 0..self.products_per_lane {
                sum = builder.mul_add_extension(
                    vars.local_wires[self.wire_shared(i)],
                    vars.local_wires[self.wire_right(lane, i)],
                    sum,
                );
            }
            constraints.push(builder.sub_extension(vars.local_wires[self.wire_acc_out(lane)], sum));
        }
        constraints
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        (0..self.num_lanes)
            .map(|lane| {
                WitnessGeneratorRef::new(
                    DotProductGenerator {
                        row,
                        num_lanes: self.num_lanes,
                        products_per_lane: self.products_per_lane,
                        lane,
                    }
                    .adapter(),
                )
            })
            .collect()
    }

    fn num_wires(&self) -> usize {
        2 * self.num_lanes + (self.num_lanes + 1) * self.products_per_lane
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        2
    }

    fn num_constraints(&self) -> usize {
        self.num_lanes
    }
}

impl<F: RichField + Extendable<D>, const D: usize> PackedEvaluableBase<F, D> for DotProductGate {
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        vars: EvaluationVarsBasePacked<P>,
        mut yield_constr: StridedConstraintConsumer<P>,
    ) {
        for lane in 0..self.num_lanes {
            let mut sum = vars.local_wires[self.wire_acc_in(lane)];
            for i in 0..self.products_per_lane {
                sum += vars.local_wires[self.wire_shared(i)]
                    * vars.local_wires[self.wire_right(lane, i)];
            }
            yield_constr.one(vars.local_wires[self.wire_acc_out(lane)] - sum);
        }
    }
}

#[derive(Clone, Debug, Default)]
struct DotProductGenerator {
    row: usize,
    num_lanes: usize,
    products_per_lane: usize,
    lane: usize,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D> for DotProductGenerator {
    fn id(&self) -> String {
        "VeriRouterDotProductGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        let gate = DotProductGate::new(self.num_lanes, self.products_per_lane);
        let mut dependencies = Vec::with_capacity(1 + 2 * self.products_per_lane);
        dependencies.push(Target::wire(self.row, gate.wire_acc_in(self.lane)));
        for i in 0..self.products_per_lane {
            dependencies.push(Target::wire(self.row, gate.wire_shared(i)));
            dependencies.push(Target::wire(self.row, gate.wire_right(self.lane, i)));
        }
        dependencies
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let gate = DotProductGate::new(self.num_lanes, self.products_per_lane);
        let mut sum = witness.get_target(Target::wire(self.row, gate.wire_acc_in(self.lane)));
        for i in 0..self.products_per_lane {
            let left = witness.get_target(Target::wire(self.row, gate.wire_shared(i)));
            let right = witness.get_target(Target::wire(self.row, gate.wire_right(self.lane, i)));
            sum += left * right;
        }
        out_buffer.set_target(Target::wire(self.row, gate.wire_acc_out(self.lane)), sum)
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.row)?;
        dst.write_usize(self.num_lanes)?;
        dst.write_usize(self.products_per_lane)?;
        dst.write_usize(self.lane)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        Ok(Self {
            row: src.read_usize()?,
            num_lanes: src.read_usize()?,
            products_per_lane: src.read_usize()?,
            lane: src.read_usize()?,
        })
    }
}

pub fn batch_dot_product_accumulate<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    initial: &[Target],
    shared: &[Target],
    right: &[&[Target]],
) -> Vec<Target> {
    assert_eq!(initial.len(), right.len());
    assert!(!right.is_empty());
    assert!(right.iter().all(|values| values.len() == shared.len()));
    let gate = DotProductGate::new_from_config(&builder.config);
    assert!(right.len() <= gate.num_lanes);
    assert!(gate.products_per_lane > 0);
    let zero = builder.zero();
    let mut accumulators = initial.to_vec();
    accumulators.resize(gate.num_lanes, zero);
    for offset in (0..shared.len()).step_by(gate.products_per_lane) {
        let row = builder.add_gate(gate.clone(), vec![]);
        for (lane, &acc) in accumulators.iter().enumerate() {
            builder.connect(acc, Target::wire(row, gate.wire_acc_in(lane)));
        }
        for i in 0..gate.products_per_lane {
            let source_idx = offset + i;
            let shared_target = shared.get(source_idx).copied().unwrap_or(zero);
            builder.connect(shared_target, Target::wire(row, gate.wire_shared(i)));
            for lane in 0..gate.num_lanes {
                let right_target = right
                    .get(lane)
                    .and_then(|values| values.get(source_idx))
                    .copied()
                    .unwrap_or(zero);
                builder.connect(right_target, Target::wire(row, gate.wire_right(lane, i)));
            }
        }
        for (lane, acc) in accumulators.iter_mut().enumerate() {
            *acc = Target::wire(row, gate.wire_acc_out(lane));
        }
    }
    accumulators.truncate(right.len());
    accumulators
}
