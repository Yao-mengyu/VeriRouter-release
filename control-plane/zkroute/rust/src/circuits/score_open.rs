use crate::util::ExactZip;
use plonky2::iop::target::{BoolTarget, Target};
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitData;

use crate::circuits::common::{
    add_merkle_path, assert_bool_implies_eq, assert_bool_implies_zero, connect_hashes,
    constrain_node_id_with_mode, hash_query, hash_score_table, verify_node_openings,
    MerklePathTargets, NodeIdRangeMode,
};
use crate::commit_check::NodeMerkleData;
use crate::gadgets::dot_product::{batch_dot_product_accumulate, DotProductGate};
use crate::gadgets::order::assert_positive_diff_if;
use crate::util::{
    canonical_node_id, recursion_config, u32_to_f, u64_to_f, C, D, F, NODE_ID_BITS,
    NODE_ID_SENTINEL, SCORE_BIAS,
};
use crate::witness::ProofInput;

#[derive(Clone)]
pub struct ScoreOpenTargets {
    pub node_root_pi: plonky2::hash::hash_types::HashOutTarget,
    pub q_hash_pi: plonky2::hash::hash_types::HashOutTarget,
    pub score_table_hash_pi: plonky2::hash::hash_types::HashOutTarget,
    pub zero_point: Target,
    pub query_u16: Vec<Target>,
    pub node_ids: Vec<Target>,
    pub model_ids: Vec<Target>,
    pub vectors_u16: Vec<Vec<Target>>,
    pub score_encs: Vec<Target>,
    pub node_paths: Vec<MerklePathTargets>,
}

#[derive(Clone, Copy, Debug)]
pub struct ScoreOpenBuildStats {
    pub pre_padding_gate_count: usize,
    pub node_merkle_opening_gate_count: usize,
    pub node_merkle_leaf_hash_gate_count: usize,
    pub node_merkle_path_gate_count: usize,
    pub score_formula_gate_count: usize,
    pub score_table_hash_gate_count: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScoreDotFormula {
    Original,
    CenteredQuery,
}

fn score_query_terms(
    builder: &mut CircuitBuilder<F, D>,
    zero_point: Target,
    query_u16: &[Target],
) -> (Vec<Target>, Target) {
    let mut sum_q = builder.zero();
    let mut centered_query = Vec::with_capacity(query_u16.len());
    for &q in query_u16 {
        sum_q = builder.add(sum_q, q);
        centered_query.push(builder.sub(q, zero_point));
    }
    let z_sum_q = builder.mul(zero_point, sum_q);
    let z_sq = builder.mul(zero_point, zero_point);
    let dim_t = builder.constant(u64_to_f(query_u16.len() as u64));
    let dim_z_sq = builder.mul(dim_t, z_sq);
    let bias = builder.constant(u64_to_f(SCORE_BIAS));
    let offset = builder.sub(dim_z_sq, z_sum_q);
    let offset = builder.add(offset, bias);
    (centered_query, offset)
}

fn original_score_query_terms(
    builder: &mut CircuitBuilder<F, D>,
    zero_point: Target,
    query_u16: &[Target],
) -> (Target, Target, Target) {
    let mut sum_q = builder.zero();
    for &q in query_u16 {
        sum_q = builder.add(sum_q, q);
    }
    let z_sum_q = builder.mul(zero_point, sum_q);
    let z_sq = builder.mul(zero_point, zero_point);
    let dim_t = builder.constant(u64_to_f(query_u16.len() as u64));
    let dim_z_sq = builder.mul(dim_t, z_sq);
    let bias = builder.constant(u64_to_f(SCORE_BIAS));
    (z_sum_q, dim_z_sq, bias)
}

fn encoded_score_original(
    builder: &mut CircuitBuilder<F, D>,
    zero_point: Target,
    query_u16: &[Target],
    z_sum_q: Target,
    dim_z_sq: Target,
    bias: Target,
    vector_u16: &[Target],
) -> Target {
    let mut dot = builder.zero();
    let mut sum_v = builder.zero();
    for (&q, &v) in query_u16.iter().zip(vector_u16.iter()) {
        let prod = builder.mul(q, v);
        dot = builder.add(dot, prod);
        sum_v = builder.add(sum_v, v);
    }
    let z_sum_v = builder.mul(zero_point, sum_v);
    let score = builder.sub(dot, z_sum_q);
    let score = builder.sub(score, z_sum_v);
    let score = builder.add(score, dim_z_sq);
    builder.add(score, bias)
}

fn encoded_scores_from_centered_dot(
    builder: &mut CircuitBuilder<F, D>,
    centered_query: &[Target],
    score_offset: Target,
    vectors_u16: &[Vec<Target>],
) -> Vec<Target> {
    let lane_count = DotProductGate::new_from_config(&builder.config).num_lanes;
    let mut scores = Vec::with_capacity(vectors_u16.len());
    for vectors in vectors_u16.chunks(lane_count) {
        let initial = vec![score_offset; vectors.len()];
        let vector_slices = vectors.iter().map(Vec::as_slice).collect::<Vec<_>>();
        scores.extend(batch_dot_product_accumulate(
            builder,
            &initial,
            centered_query,
            &vector_slices,
        ));
    }
    scores
}

pub fn build_score_open_circuit(
    use_zk: bool,
    node_count: usize,
    dimension: usize,
    node_tree_height: usize,
) -> (CircuitData<F, C, D>, ScoreOpenTargets) {
    let (data, targets, _) = build_score_open_circuit_with_options(
        use_zk,
        node_count,
        dimension,
        node_tree_height,
        NodeIdRangeMode::Wide56,
        ScoreDotFormula::CenteredQuery,
    );
    (data, targets)
}

pub fn build_score_open_circuit_with_options(
    use_zk: bool,
    node_count: usize,
    dimension: usize,
    node_tree_height: usize,
    node_id_mode: NodeIdRangeMode,
    score_formula: ScoreDotFormula,
) -> (CircuitData<F, C, D>, ScoreOpenTargets, ScoreOpenBuildStats) {
    assert!(node_count > 0, "score node count must be positive");
    // Conservative integer bound, independent of the witness zero point.
    assert!(
        dimension > 0 && (dimension as u128) * 65535u128.pow(2) < SCORE_BIAS as u128,
        "dimension must keep centered scores strictly within the 48-bit biased domain"
    );
    let config = recursion_config(use_zk);
    let mut builder = CircuitBuilder::<F, D>::new(config);

    let node_root_pi = builder.add_virtual_hash_public_input();
    let q_hash_pi = builder.add_virtual_hash_public_input();
    let score_table_hash_pi = builder.add_virtual_hash_public_input();

    let zero_point = builder.add_virtual_target();

    let query_u16 = builder.add_virtual_targets(dimension);
    builder.range_check(zero_point, 16);
    for &q in &query_u16 {
        builder.range_check(q, 16);
    }

    let node_ids = builder.add_virtual_targets(node_count);
    let model_ids = builder.add_virtual_targets(node_count);
    let mut vectors_u16 = Vec::with_capacity(node_count);
    let mut score_encs = Vec::with_capacity(node_count);
    let mut node_paths = Vec::with_capacity(node_count);
    for _ in 0..node_count {
        let vec = builder.add_virtual_targets(dimension);
        // Active vectors are authenticated against the endorsed reference table.
        // Offline approval checks its complete, fixed-dimension u16 encoding.
        // Joint extraction plus commitment/Merkle binding transfers that range
        // to these coordinates; repeating it per lookup is unnecessary. A
        // standalone leaf with an arbitrary, unapproved root has no such claim.
        vectors_u16.push(vec);
        let score = builder.add_virtual_target();
        builder.range_check(score, crate::util::SCORE_BITS);
        score_encs.push(score);
        node_paths.push(add_merkle_path(&mut builder, node_tree_height));
    }

    let q_hash_calc = hash_query(&mut builder, zero_point, &query_u16);
    connect_hashes(&mut builder, &q_hash_pi, &q_hash_calc);

    let mut enabled = Vec::<BoolTarget>::with_capacity(node_count);
    let mut seen_dummy = builder._false();
    for i in 0..node_count {
        let is_dummy = constrain_node_id_with_mode(&mut builder, node_ids[i], node_id_mode);
        let is_active = builder.not(is_dummy);
        assert_bool_implies_zero(&mut builder, seen_dummy, is_active.target);
        seen_dummy = builder.or(seen_dummy, is_dummy);
        assert_bool_implies_zero(&mut builder, is_dummy, model_ids[i]);
        assert_bool_implies_zero(&mut builder, is_dummy, score_encs[i]);
        if i > 0 {
            assert_positive_diff_if(
                &mut builder,
                is_active,
                node_ids[i],
                node_ids[i - 1],
                NODE_ID_BITS,
            );
        }
        enabled.push(is_active);
    }
    let node_merkle_start = builder.num_gates();
    let node_merkle_stats = verify_node_openings(
        &mut builder,
        &node_root_pi,
        &node_ids,
        &model_ids,
        &vectors_u16,
        &node_paths,
        &enabled,
    );
    let node_merkle_opening_gate_count = builder.num_gates() - node_merkle_start;

    let score_formula_start = builder.num_gates();
    match score_formula {
        ScoreDotFormula::CenteredQuery => {
            let (centered_query, score_offset) =
                score_query_terms(&mut builder, zero_point, &query_u16);
            let expected_scores = encoded_scores_from_centered_dot(
                &mut builder,
                &centered_query,
                score_offset,
                &vectors_u16,
            );
            for (i, expected) in expected_scores.into_iter().enumerate() {
                assert_bool_implies_eq(&mut builder, enabled[i], expected, score_encs[i]);
            }
        }
        ScoreDotFormula::Original => {
            let (z_sum_q, dim_z_sq, bias) =
                original_score_query_terms(&mut builder, zero_point, &query_u16);
            for i in 0..node_count {
                let expected = encoded_score_original(
                    &mut builder,
                    zero_point,
                    &query_u16,
                    z_sum_q,
                    dim_z_sq,
                    bias,
                    &vectors_u16[i],
                );
                assert_bool_implies_eq(&mut builder, enabled[i], expected, score_encs[i]);
            }
        }
    }
    let score_formula_gate_count = builder.num_gates() - score_formula_start;

    let score_table_hash_start = builder.num_gates();
    let score_table_hash_calc = hash_score_table(&mut builder, &node_ids, &score_encs, &model_ids);
    connect_hashes(&mut builder, &score_table_hash_pi, &score_table_hash_calc);
    let score_table_hash_gate_count = builder.num_gates() - score_table_hash_start;

    let stats = ScoreOpenBuildStats {
        pre_padding_gate_count: builder.num_gates(),
        node_merkle_opening_gate_count,
        node_merkle_leaf_hash_gate_count: node_merkle_stats.leaf_hash_gate_count,
        node_merkle_path_gate_count: node_merkle_stats.path_gate_count,
        score_formula_gate_count,
        score_table_hash_gate_count,
    };
    let data = builder.build::<C>();
    let targets = ScoreOpenTargets {
        node_root_pi,
        q_hash_pi,
        score_table_hash_pi,
        zero_point,
        query_u16,
        node_ids,
        model_ids,
        vectors_u16,
        score_encs,
        node_paths,
    };
    (data, targets, stats)
}

pub fn fill_score_open_witness(
    pw: &mut PartialWitness<F>,
    targets: &ScoreOpenTargets,
    input: &ProofInput,
    node_merkle: &NodeMerkleData,
    node_root: &[u64; 4],
    q_hash: &[u64; 4],
    score_table_hash: &[u64; 4],
) {
    pw.set_target(targets.zero_point, u32_to_f(input.quantizer.zero_point))
        .expect("set zero_point");
    for (target, &value) in targets
        .query_u16
        .iter()
        .zip_exact(input.query.vector_u16.iter())
    {
        pw.set_target(*target, u32_to_f(u32::from(value)))
            .expect("set query value");
    }
    for (i, node) in (0..targets.node_ids.len()).zip_exact(&input.nodes) {
        pw.set_target(
            targets.node_ids[i],
            u64_to_f(canonical_node_id(node.node_id)),
        )
        .expect("set node_id");
        pw.set_target(targets.model_ids[i], u32_to_f(node.model_id))
            .expect("set model_id");
        for (target, &value) in targets.vectors_u16[i]
            .iter()
            .zip_exact(node.vector_u16.iter())
        {
            pw.set_target(*target, u32_to_f(u32::from(value)))
                .expect("set vector value");
        }
        pw.set_target(targets.score_encs[i], u64_to_f(node.score_enc))
            .expect("set score_enc");
        if canonical_node_id(node.node_id) == NODE_ID_SENTINEL {
            for bit_t in &targets.node_paths[i].index_bits {
                pw.set_bool_target(*bit_t, false)
                    .expect("set dummy node path bit");
            }
            for sib_t in &targets.node_paths[i].siblings {
                pw.set_hash_target(*sib_t, crate::commitment::digest_to_hashout([0, 0, 0, 0]))
                    .expect("set dummy node path sibling");
            }
        } else {
            let path = node_merkle
                .open_by_id(canonical_node_id(node.node_id))
                .expect("global node opening");
            for (bit_t, &bit) in targets.node_paths[i]
                .index_bits
                .iter()
                .zip_exact(path.index_bits.iter())
            {
                pw.set_bool_target(*bit_t, bit).expect("set node path bit");
            }
            for (sib_t, sib) in targets.node_paths[i]
                .siblings
                .iter()
                .zip_exact(path.siblings.iter())
            {
                pw.set_hash_target(*sib_t, crate::commitment::digest_to_hashout(*sib))
                    .expect("set node path sibling");
            }
        }
    }
    pw.set_hash_target(
        targets.node_root_pi,
        crate::commitment::digest_to_hashout(*node_root),
    )
    .expect("set node_root");
    pw.set_hash_target(
        targets.q_hash_pi,
        crate::commitment::digest_to_hashout(*q_hash),
    )
    .expect("set q_hash");
    pw.set_hash_target(
        targets.score_table_hash_pi,
        crate::commitment::digest_to_hashout(*score_table_hash),
    )
    .expect("set score table hash");
}
