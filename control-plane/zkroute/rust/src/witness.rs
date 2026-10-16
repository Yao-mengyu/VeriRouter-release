//! Route input data and witness preparation.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::util::{
    canonical_node_id, is_node_id_sentinel, read_text_or_zlib, NodeId, NODE_ID_SENTINEL, SCORE_BITS,
};

fn is_zero_u64(value: &u64) -> bool {
    *value == 0
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CommitmentRandomness {
    pub reference: [u64; 4],
    pub graph: [u64; 4],
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PositionSchema {
    PaperWorkupdV1,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofInput {
    pub version: String,
    pub position_schema: PositionSchema,
    pub commitment_randomness: CommitmentRandomness,
    pub metric: String,
    pub query_index: u32,
    pub score_bits: usize,
    pub entry_point: NodeId,
    /// Backend selected for serving. Legacy traces omit this and select rank one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_model_id: Option<u32>,
    pub global_artifacts: GlobalArtifacts,
    pub quantizer: AffineQuantizer,
    pub query: QuantizedQuery,
    pub nodes: Vec<ProofNode>,
    pub expand_rows: Vec<ExpandRow>,
    pub upper_trace: Vec<UpperTraceStep>,
    pub ef_search: EfSearchTrace,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<ProofInputSource>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalArtifacts {
    pub node_table_path: String,
    pub graph_rows_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffineQuantizer {
    pub scheme: String,
    pub scale: f64,
    pub zero_point: u32,
    pub projected_dim: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedQuery {
    pub dimension: usize,
    pub vector_u16: Vec<u16>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofNode {
    pub node_id: NodeId,
    pub model_id: u32,
    pub vector_u16: Vec<u16>,
    pub score_enc: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProofInputSource {
    #[serde(default)]
    pub score_node_count_cap: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistRef {
    pub node_id: NodeId,
    pub score_enc: u64,
    pub model_id: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpandRow {
    pub layer: u32,
    pub src: DistRef,
    pub neighbors: Vec<DistRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpperTraceStep {
    pub selected: DistRef,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeighborVisit {
    pub node_id: NodeId,
    pub is_new: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfExpansion {
    pub pop_id: NodeId,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalTopkRef {
    pub node_id: NodeId,
    pub score_enc: u64,
    pub model_id: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PqTrace {
    pub del_src_times: Vec<u32>,
    pub del_refs: Vec<DistRef>,
    pub del_dummy_ordinals: Vec<u32>,
    pub del_tie_positions: Vec<u32>,
    pub final_src_times: Vec<u32>,
    pub final_refs: Vec<DistRef>,
    pub final_dummy_ordinals: Vec<u32>,
    pub final_tie_positions: Vec<u32>,
}

/// One physical row of the paper's WorkUpd schedule.
///
/// `item` is serialized as `ref` to match the language used by the trace
/// exporter. Row zero is the entry; row `1+t` is raw level-0 slot `t`.
/// Inactive rows are canonical holes. The loader deliberately does not
/// "repair" semantic witness values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkUpdate {
    pub active: bool,
    #[serde(rename = "ref")]
    pub item: DistRef,
    pub assigned_c_position: u32,
    pub w_push_position: u32,
    pub w_pop_position: u32,
    pub admitted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrevLowerEvent {
    pub active: bool,
    pub is_push: bool,
    pub cur_index: u32,
    pub cur_key: u64,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub cur_tie: u64,
    pub cur_time: u32,
    pub top_before_index: u32,
    pub top_before_key: u64,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub top_before_tie: u64,
    pub top_before_time: u32,
    pub top_after_index: u32,
    pub top_after_key: u64,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub top_after_tie: u64,
    pub top_after_time: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrevLowerFinalNode {
    pub node_index: u32,
    pub node_key: u64,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub node_tie: u64,
    pub node_time: u32,
    pub parent_index: u32,
    pub parent_key: u64,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub parent_tie: u64,
    pub parent_time: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrevLowerTrace {
    pub prev_lower: Vec<u32>,
    pub prev_lower_times: Vec<u32>,
    pub events: Vec<PrevLowerEvent>,
    pub final_nodes: Vec<PrevLowerFinalNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfSearchTrace {
    pub ef_search: u32,
    pub k: u32,
    pub expansions: Vec<EfExpansion>,
    pub neighbor_visits: Vec<NeighborVisit>,
    pub work_updates: Vec<WorkUpdate>,
    pub final_topk: Vec<FinalTopkRef>,
    pub c_pq: PqTrace,
    pub w_pq: PqTrace,
    pub c_prev_lower: PrevLowerTrace,
    pub w_prev_lower: PrevLowerTrace,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ScoreTableCompaction {
    pub original_node_count: usize,
    pub compact_node_count: usize,
    pub target_node_count: usize,
    pub final_node_count: usize,
    pub removed_duplicate_nodes: usize,
    pub padded_dummy_nodes: usize,
}

fn same_node_payload(a: &ProofNode, b: &ProofNode) -> bool {
    a.model_id == b.model_id && a.score_enc == b.score_enc && a.vector_u16 == b.vector_u16
}

fn canonicalize_proof_input_node_ids(input: &mut ProofInput) {
    input.entry_point = canonical_node_id(input.entry_point);
    for node in &mut input.nodes {
        node.node_id = canonical_node_id(node.node_id);
    }
    for row in &mut input.expand_rows {
        row.src.node_id = canonical_node_id(row.src.node_id);
        for neighbor in &mut row.neighbors {
            neighbor.node_id = canonical_node_id(neighbor.node_id);
        }
    }
    for step in &mut input.upper_trace {
        step.selected.node_id = canonical_node_id(step.selected.node_id);
    }
    for expansion in &mut input.ef_search.expansions {
        expansion.pop_id = canonical_node_id(expansion.pop_id);
    }
    for visit in &mut input.ef_search.neighbor_visits {
        visit.node_id = canonical_node_id(visit.node_id);
    }
    for value in &mut input.ef_search.final_topk {
        value.node_id = canonical_node_id(value.node_id);
    }
    for update in &mut input.ef_search.work_updates {
        update.item.node_id = canonical_node_id(update.item.node_id);
    }
    for value in &mut input.ef_search.c_pq.del_refs {
        value.node_id = canonical_node_id(value.node_id);
    }
    for value in &mut input.ef_search.c_pq.final_refs {
        value.node_id = canonical_node_id(value.node_id);
    }
    for value in &mut input.ef_search.w_pq.del_refs {
        value.node_id = canonical_node_id(value.node_id);
    }
    for value in &mut input.ef_search.w_pq.final_refs {
        value.node_id = canonical_node_id(value.node_id);
    }
}

pub fn compact_score_table(
    input: &mut ProofInput,
    score_node_count_cap_override: Option<usize>,
) -> Result<ScoreTableCompaction> {
    canonicalize_proof_input_node_ids(input);
    let sent_id = NODE_ID_SENTINEL;
    let original_node_count = input.nodes.len();
    let target_node_count = score_node_count_cap_override
        .or_else(|| {
            input
                .source
                .as_ref()
                .and_then(|source| source.score_node_count_cap)
        })
        .unwrap_or(original_node_count);
    if target_node_count == 0 {
        bail!("score_node_count_cap must be positive");
    }
    let mut first_by_id = BTreeMap::<NodeId, usize>::new();
    let mut compact_nodes = Vec::<ProofNode>::new();

    for node in &input.nodes {
        if is_node_id_sentinel(node.node_id) {
            continue;
        }
        if let Some(&idx) = first_by_id.get(&node.node_id) {
            if !same_node_payload(&compact_nodes[idx], node) {
                bail!(
                    "duplicate score-table node {} has inconsistent payload",
                    node.node_id
                );
            }
            continue;
        }
        first_by_id.insert(node.node_id, compact_nodes.len());
        compact_nodes.push(node.clone());
    }

    if compact_nodes.is_empty() {
        bail!("score table is empty after compaction");
    }
    if compact_nodes.len() > target_node_count {
        bail!(
            "score_node_count_cap {} is smaller than unique score node count {}",
            target_node_count,
            compact_nodes.len()
        );
    }
    compact_nodes.sort_by_key(|node| node.node_id);

    let compact_node_count = compact_nodes.len();
    let dummy_node = ProofNode {
        node_id: sent_id,
        model_id: 0,
        vector_u16: vec![0; input.query.dimension],
        score_enc: 0,
    };
    while compact_nodes.len() < target_node_count {
        compact_nodes.push(dummy_node.clone());
    }

    input.nodes = compact_nodes;
    let final_node_count = input.nodes.len();
    Ok(ScoreTableCompaction {
        original_node_count,
        compact_node_count,
        target_node_count,
        final_node_count,
        removed_duplicate_nodes: original_node_count.saturating_sub(compact_node_count),
        padded_dummy_nodes: final_node_count.saturating_sub(compact_node_count),
    })
}

/// Fixed circuit shape. Padded traces of one bucket share it, so the circuits and
/// verifier key do not depend on the query's search path.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RouteShape {
    pub node_count: usize,
    pub max_upper_steps: usize,
    pub max_rounds: usize,
    pub upper_neighbor_cap: usize,
    pub round_neighbor_cap: usize,
    pub visit_len: usize,
    pub c_final_len: usize,
    pub w_final_len: usize,
}

impl RouteShape {
    /// Reads the shape of a trace and rejects traces that are not padded to it.
    pub fn of(input: &ProofInput) -> Result<Self> {
        let upper = input.upper_trace.len();
        let rows = &input.expand_rows;
        let cap = |rows: &[ExpandRow]| rows.first().map_or(0, |row| row.neighbors.len());
        let (upper_rows, round_rows) = rows.split_at(upper.min(rows.len()));
        let shape = Self {
            node_count: input.nodes.len(),
            max_upper_steps: upper,
            max_rounds: input.ef_search.expansions.len(),
            upper_neighbor_cap: cap(upper_rows),
            round_neighbor_cap: cap(round_rows),
            visit_len: input.ef_search.neighbor_visits.len(),
            c_final_len: input.ef_search.c_pq.final_refs.len(),
            w_final_len: input.ef_search.w_pq.final_refs.len(),
        };
        if rows.len() != upper + shape.max_rounds
            || upper_rows
                .iter()
                .any(|row| row.neighbors.len() != shape.upper_neighbor_cap)
            || round_rows
                .iter()
                .any(|row| row.neighbors.len() != shape.round_neighbor_cap)
            || shape.visit_len != shape.max_rounds * shape.round_neighbor_cap
        {
            bail!("trace is not padded to a fixed route shape; prepare it with `verirouter.py experiment`");
        }
        Ok(shape)
    }

    pub fn row_neighbor_lens(&self) -> Vec<usize> {
        let upper = std::iter::repeat(self.upper_neighbor_cap).take(self.max_upper_steps);
        let rounds = std::iter::repeat(self.round_neighbor_cap).take(self.max_rounds);
        upper.chain(rounds).collect()
    }
}

/// Last real upper-layer selection; padded steps select the sentinel.
pub fn upper_end(input: &ProofInput) -> NodeId {
    input
        .upper_trace
        .iter()
        .rev()
        .map(|step| canonical_node_id(step.selected.node_id))
        .find(|&node_id| node_id != NODE_ID_SENTINEL)
        .unwrap_or_else(|| canonical_node_id(input.entry_point))
}

pub fn final_output_model_id(input: &ProofInput) -> Result<u32> {
    if let Some(model_id) = input.selected_model_id {
        return Ok(model_id);
    }
    let top1 = input
        .ef_search
        .final_topk
        .first()
        .context("ef_search.final_topk is empty")?;
    Ok(top1.model_id)
}

pub fn ef_search_unused_ref_len(input: &ProofInput) -> usize {
    let mut used = BTreeSet::<NodeId>::new();
    for row in &input.expand_rows {
        if canonical_node_id(row.src.node_id) != NODE_ID_SENTINEL {
            used.insert(canonical_node_id(row.src.node_id));
        }
        for nb in &row.neighbors {
            if canonical_node_id(nb.node_id) != NODE_ID_SENTINEL {
                used.insert(canonical_node_id(nb.node_id));
            }
        }
    }
    for value in &input.ef_search.c_pq.final_refs {
        if canonical_node_id(value.node_id) != NODE_ID_SENTINEL {
            used.insert(canonical_node_id(value.node_id));
        }
    }
    for value in &input.ef_search.w_pq.del_refs {
        if canonical_node_id(value.node_id) != NODE_ID_SENTINEL {
            used.insert(canonical_node_id(value.node_id));
        }
    }
    for value in &input.ef_search.w_pq.final_refs {
        if canonical_node_id(value.node_id) != NODE_ID_SENTINEL {
            used.insert(canonical_node_id(value.node_id));
        }
    }
    let active_nodes = input
        .nodes
        .iter()
        .filter(|node| canonical_node_id(node.node_id) != NODE_ID_SENTINEL)
        .count();
    active_nodes.saturating_sub(used.len())
}

/// Loads a trace and reads its fixed route shape.
pub(crate) fn load_input(path: &Path) -> Result<(ProofInput, ScoreTableCompaction, RouteShape)> {
    let payload =
        read_text_or_zlib(path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut input: ProofInput = serde_json::from_str(&payload)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    let score_table_compaction = compact_score_table(&mut input, None)?;
    if input.metric != "dot" {
        bail!("only dot metric is supported, got {}", input.metric);
    }
    if input.score_bits != SCORE_BITS {
        bail!(
            "score_bits mismatch: expected {}, got {}",
            SCORE_BITS,
            input.score_bits
        );
    }
    let shape = RouteShape::of(&input)?;
    Ok((input, score_table_compaction, shape))
}
