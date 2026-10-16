//! Check the full reference and graph data before endorsing their commitments.
//! The checked tables also provide Merkle roots and witness openings.

use anyhow::{bail, Context, Result};
use plonky2::field::types::PrimeField64;
use plonky2::hash::hash_types::HashOut;
use plonky2::hash::poseidon::PoseidonHash;
use plonky2::plonk::config::Hasher;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::util::{
    canonical_node_id, u32_to_f, u64_to_f, NodeId, F, GRAPH_NEIGHBOR_LIST_HASH_DOMAIN,
    MAX_REAL_NODE_ID, NODE_ID_SENTINEL, NODE_VECTOR_HASH_DOMAIN,
};

#[derive(Clone)]
pub struct NodeMerklePath {
    pub index_bits: Vec<bool>,
    pub siblings: Vec<[u64; 4]>,
}

#[derive(Clone)]
pub struct GraphMerklePath {
    pub index_bits: Vec<bool>,
    pub siblings: Vec<[u64; 4]>,
}

#[derive(Clone)]
pub struct NodeMerkleData {
    levels: Vec<Vec<HashOut<F>>>,
    by_id: BTreeMap<NodeId, usize>,
    dimension: usize,
}

#[derive(Clone)]
pub struct GraphMerkleData {
    levels: Vec<Vec<HashOut<F>>>,
    by_key: BTreeMap<(NodeId, u32), usize>,
    source_ids: BTreeSet<NodeId>,
    neighbor_ids: BTreeSet<NodeId>,
    max_layer: u32,
}

#[derive(Debug, Clone, Deserialize)]
struct GlobalNodeRecord {
    node_id: NodeId,
    model_id: u32,
    vector_u16: Vec<u16>,
}

#[derive(Debug, Clone, Deserialize)]
struct GlobalGraphRowRecord {
    layer: u32,
    src: NodeId,
    neighbors: Vec<NodeId>,
}

fn hash_fields(fields: &[F]) -> HashOut<F> {
    PoseidonHash::hash_no_pad(fields)
}

fn hashout_to_u64s(hash: HashOut<F>) -> [u64; 4] {
    [
        hash.elements[0].to_canonical_u64(),
        hash.elements[1].to_canonical_u64(),
        hash.elements[2].to_canonical_u64(),
        hash.elements[3].to_canonical_u64(),
    ]
}

fn build_levels(mut leaves: Vec<HashOut<F>>) -> Vec<Vec<HashOut<F>>> {
    assert!(!leaves.is_empty(), "empty merkle tree");
    let mut levels = vec![std::mem::take(&mut leaves)];
    while levels.last().expect("merkle leaves").len() > 1 {
        let cur = levels.last().expect("merkle level");
        let mut next = Vec::with_capacity((cur.len() + 1) / 2);
        for pair in cur.chunks(2) {
            let left = pair[0];
            let right = if pair.len() == 2 { pair[1] } else { pair[0] };
            let mut fields = Vec::with_capacity(8);
            fields.extend_from_slice(&left.elements);
            fields.extend_from_slice(&right.elements);
            next.push(hash_fields(&fields));
        }
        levels.push(next);
    }
    levels
}

fn open_index(levels: &[Vec<HashOut<F>>], mut idx: usize) -> (Vec<bool>, Vec<[u64; 4]>) {
    let tree_height = levels.len().saturating_sub(1);
    let mut index_bits = Vec::with_capacity(tree_height);
    let mut siblings = Vec::with_capacity(tree_height);
    for level in &levels[..tree_height] {
        let is_right = idx % 2 == 1;
        let sib_idx = if is_right {
            idx - 1
        } else if idx + 1 < level.len() {
            idx + 1
        } else {
            idx
        };
        index_bits.push(is_right);
        siblings.push(hashout_to_u64s(level[sib_idx]));
        idx /= 2;
    }
    (index_bits, siblings)
}

fn hash_node_leaf_record(node: &GlobalNodeRecord) -> HashOut<F> {
    let mut vector_fields = Vec::with_capacity(2 + node.vector_u16.len());
    vector_fields.push(u64_to_f(NODE_VECTOR_HASH_DOMAIN));
    vector_fields.push(u64_to_f(node.vector_u16.len() as u64));
    vector_fields.extend(node.vector_u16.iter().map(|&v| u32_to_f(u32::from(v))));
    let vec_hash = hash_fields(&vector_fields);
    hash_fields(&[
        u64_to_f(node.node_id),
        u32_to_f(node.model_id),
        vec_hash.elements[0],
        vec_hash.elements[1],
        vec_hash.elements[2],
        vec_hash.elements[3],
    ])
}

fn hash_graph_row_leaf_record(row: &GlobalGraphRowRecord) -> HashOut<F> {
    let mut neighbor_fields = Vec::with_capacity(2 + row.neighbors.len());
    neighbor_fields.push(u64_to_f(GRAPH_NEIGHBOR_LIST_HASH_DOMAIN));
    neighbor_fields.push(u64_to_f(row.neighbors.len() as u64));
    neighbor_fields.extend(
        row.neighbors
            .iter()
            .map(|&v| u64_to_f(canonical_node_id(v))),
    );
    let nbr_hash = hash_fields(&neighbor_fields);
    hash_fields(&[
        u32_to_f(row.layer),
        u64_to_f(canonical_node_id(row.src)),
        nbr_hash.elements[0],
        nbr_hash.elements[1],
        nbr_hash.elements[2],
        nbr_hash.elements[3],
    ])
}

fn checked_real_artifact_id(value: NodeId, label: &str) -> Result<NodeId> {
    let canonical = canonical_node_id(value);
    if canonical > MAX_REAL_NODE_ID {
        bail!("{label} {value} is not a canonical real node id");
    }
    Ok(canonical)
}

fn checked_graph_neighbor_id(value: NodeId, label: &str) -> Result<NodeId> {
    let canonical = canonical_node_id(value);
    if canonical == NODE_ID_SENTINEL {
        if value != NODE_ID_SENTINEL {
            bail!("{label} uses a non-canonical sentinel spelling");
        }
        return Ok(canonical);
    }
    if canonical > MAX_REAL_NODE_ID {
        bail!("{label} {value} exceeds the real node-id domain");
    }
    Ok(canonical)
}

fn open_jsonl(path: &Path) -> Result<Box<dyn BufRead>> {
    let file = File::open(path).with_context(|| format!("failed to read {}", path.display()))?;
    if path.extension().and_then(|value| value.to_str()) == Some("bin") {
        Ok(Box::new(BufReader::new(flate2::read::ZlibDecoder::new(
            file,
        ))))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

fn for_each_jsonl<T, Visit>(path: &Path, mut visit: Visit) -> Result<usize>
where
    T: for<'de> Deserialize<'de>,
    Visit: FnMut(T, usize) -> Result<()>,
{
    let reader = open_jsonl(path)?;
    let mut count = 0usize;
    for (line_no, line) in reader.lines().enumerate() {
        let line = line
            .with_context(|| format!("failed to read {} line {}", path.display(), line_no + 1))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value = serde_json::from_str::<T>(line).with_context(|| {
            format!(
                "failed to parse {} line {} as json",
                path.display(),
                line_no + 1
            )
        })?;
        visit(value, count)?;
        count += 1;
    }
    if count == 0 {
        bail!("{} produced an empty jsonl set", path.display());
    }
    Ok(count)
}

/// Check all rows against the requested profile before computing commitments.
/// The caller is responsible for approving and authenticating the resulting commitments.
pub fn check_commit_data(
    node_table: &Path,
    graph_rows: &Path,
    dimension: usize,
) -> Result<(NodeMerkleData, GraphMerkleData)> {
    let nodes = NodeMerkleData::from_jsonl(node_table)
        .with_context(|| format!("failed to load reference table {}", node_table.display()))?;
    nodes
        .validate_dimension(dimension)
        .context("reference table does not match the profile dimension")?;
    let graph = GraphMerkleData::from_jsonl(graph_rows)
        .with_context(|| format!("failed to load graph rows {}", graph_rows.display()))?;
    graph
        .validate_node_domain(&nodes)
        .context("graph rows reference nodes outside the reference table")?;
    Ok((nodes, graph))
}

impl NodeMerkleData {
    pub fn from_jsonl(path: &Path) -> Result<Self> {
        let mut by_id = BTreeMap::new();
        let mut leaves = Vec::new();
        let mut dimension = None;
        for_each_jsonl::<GlobalNodeRecord, _>(path, |mut node, idx| {
            // Deserialization checks every coordinate as u16, including rows
            // not visited by this query. Endorsement must use this full table.
            let row_dimension = node.vector_u16.len();
            if row_dimension == 0 {
                bail!("node_table has an empty vector at row {idx}");
            }
            if let Some(expected) = dimension {
                if row_dimension != expected {
                    bail!("node_table vector dimension {row_dimension} at row {idx} differs from {expected}");
                }
            } else {
                dimension = Some(row_dimension);
            }
            let node_id = checked_real_artifact_id(node.node_id, "node_table.node_id")?;
            node.node_id = node_id;
            if by_id.insert(node_id, idx).is_some() {
                bail!("duplicate node_id {} in {}", node_id, path.display());
            }
            leaves.push(hash_node_leaf_record(&node));
            Ok(())
        })?;
        let levels = build_levels(leaves);
        Ok(Self {
            levels,
            by_id,
            dimension: dimension.expect("nonempty node table"),
        })
    }

    /// Offline approval must call this with the independently approved profile
    /// dimension before endorsing the root/commitment. Router-side validation
    /// alone is not an endorsement or an external client's trust boundary.
    pub fn validate_dimension(&self, expected: usize) -> Result<()> {
        if expected == 0 || self.dimension != expected {
            bail!(
                "node_table dimension {} differs from profile dimension {expected}",
                self.dimension
            );
        }
        Ok(())
    }

    pub fn tree_height(&self) -> usize {
        self.levels.len().saturating_sub(1)
    }

    pub fn root_u64s(&self) -> [u64; 4] {
        hashout_to_u64s(
            *self
                .levels
                .last()
                .expect("node root")
                .first()
                .expect("node root leaf"),
        )
    }

    pub fn open_by_id(&self, node_id: NodeId) -> Result<NodeMerklePath> {
        let node_id = canonical_node_id(node_id);
        let idx = self
            .by_id
            .get(&node_id)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("missing node leaf for node_id {}", node_id))?;
        let (index_bits, siblings) = open_index(&self.levels, idx);
        Ok(NodeMerklePath {
            index_bits,
            siblings,
        })
    }
}

impl GraphMerkleData {
    pub fn from_jsonl(path: &Path) -> Result<Self> {
        let mut by_key = BTreeMap::new();
        let mut source_ids = BTreeSet::new();
        let mut neighbor_ids = BTreeSet::new();
        let mut leaves = Vec::new();
        let mut max_layer = 0u32;
        for_each_jsonl::<GlobalGraphRowRecord, _>(path, |mut row, idx| {
            row.src = checked_real_artifact_id(row.src, "graph_rows.src")?;
            source_ids.insert(row.src);
            for (neighbor_idx, neighbor) in row.neighbors.iter_mut().enumerate() {
                *neighbor = checked_graph_neighbor_id(
                    *neighbor,
                    &format!("graph_rows.neighbors[{neighbor_idx}]"),
                )?;
                if *neighbor != NODE_ID_SENTINEL {
                    neighbor_ids.insert(*neighbor);
                }
            }
            let key = (row.src, row.layer);
            if by_key.insert(key, idx).is_some() {
                bail!(
                    "duplicate graph row for src={} layer={} in {}",
                    key.0,
                    key.1,
                    path.display()
                );
            }
            max_layer = max_layer.max(row.layer);
            leaves.push(hash_graph_row_leaf_record(&row));
            Ok(())
        })?;
        let levels = build_levels(leaves);
        Ok(Self {
            levels,
            by_key,
            source_ids,
            neighbor_ids,
            max_layer,
        })
    }

    /// Validate the cross-table setup invariant before any route witness is used:
    /// every real graph key and neighbor must belong to the committed node domain.
    /// Canonical sentinel neighbors are padding and are excluded in `from_jsonl`.
    pub fn validate_node_domain(&self, nodes: &NodeMerkleData) -> Result<()> {
        for &src in &self.source_ids {
            if !nodes.by_id.contains_key(&src) {
                bail!("graph source node_id {src} is missing from the node table");
            }
        }
        for &neighbor in &self.neighbor_ids {
            if !nodes.by_id.contains_key(&neighbor) {
                bail!("graph neighbor node_id {neighbor} is missing from the node table");
            }
        }
        Ok(())
    }

    pub fn tree_height(&self) -> usize {
        self.levels.len().saturating_sub(1)
    }

    pub fn max_layer(&self) -> u32 {
        self.max_layer
    }

    pub fn root_u64s(&self) -> [u64; 4] {
        hashout_to_u64s(
            *self
                .levels
                .last()
                .expect("graph root")
                .first()
                .expect("graph root leaf"),
        )
    }

    pub fn open_by_key(&self, src: NodeId, layer: u32) -> Result<GraphMerklePath> {
        let src = canonical_node_id(src);
        let idx =
            self.by_key.get(&(src, layer)).copied().ok_or_else(|| {
                anyhow::anyhow!("missing graph row for src={} layer={}", src, layer)
            })?;
        let (index_bits, siblings) = open_index(&self.levels, idx);
        Ok(GraphMerklePath {
            index_bits,
            siblings,
        })
    }
}
