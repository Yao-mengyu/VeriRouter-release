//! Rewrite input data, private witnesses, and native transcript construction.

use crate::commit_check::RewriteBanks;
use crate::commitment;
use crate::util::build_dedup_proof_from_paths;
use anyhow::Result;
use serde::Deserialize;

use crate::circuits::batch::RewriteParams;
use crate::commitment::{
    absorb_chunk_bytes, hash_bytes_init, hash_msg, merkle_root, transcript_step,
};
use crate::util::{
    dedup_proof_map, digest_to_hashout, expand_path_from_dedup_map, f_u64, hashout_to_digest,
    DedupMerkleProofWitness, Digest, DIGEST_WIDTH, F,
};
use plonky2::field::types::Field64;
use plonky2::hash::hash_types::HashOut;

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct InputParams {
    pub batch: usize,
    pub chunk_bytes: usize,
    #[serde(default)]
    pub max_chunks: usize,
    #[serde(default)]
    pub max_orig_chunks: usize,
    #[serde(default)]
    pub max_msg_chunks: usize,
    #[serde(default)]
    pub max_frag_chunks: usize,
    pub h_msg: usize,
    pub h_frag: usize,
    pub max_frags: usize,
    pub max_gap_msgs: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) struct InputFile {
    #[serde(default)]
    pub(crate) params: Option<InputParams>,
    #[serde(default)]
    pub(crate) commit: Option<InputCommit>,
    pub(crate) messages: Vec<InputMessage>,
    /// Messages inserted after the final original message (`G_n`). The host
    /// always encodes these in one canonical terminal `present=false` row.
    #[serde(default)]
    pub(crate) trailing_gap: Vec<MsgItem>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) struct InputCommit {
    /// Committed list for template / inserted messages (msg_root).
    ///
    /// Each entry commits to `HashMsg(role_id, total_len, HashBytes(content))` as a Merkle leaf.
    #[serde(default)]
    pub(crate) msg_list: Vec<InputInsertedMsg>,
    /// Committed list for prefix/suffix fragments (frag_root).
    ///
    /// Each entry commits to `HashBytes(fragment)` as a Merkle leaf.
    #[serde(default)]
    pub(crate) frag_list: Vec<String>,
    /// Secret randomizers for the two top-level bank roots and the final
    /// rewritten request digest. They are consumed as witnesses and omitted
    /// from every report/public statement.
    pub(crate) msg_randomness: Digest,
    pub(crate) frag_randomness: Digest,
    pub(crate) request_randomness: Digest,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) struct InputMessage {
    /// "system" | "user" | "assistant" or a numeric role id.
    pub(crate) role: serde_json::Value,
    pub(crate) content: String,
    /// Whether this original message is present in the transcript.
    ///
    /// - default: true (backward compatible)
    /// - if false: this message contributes nothing to orig/mod roots, but its `gap_before`
    ///   insertions are still allowed (subject to prev_present gating).
    #[serde(default = "default_true")]
    pub(crate) present: bool,
    #[serde(default)]
    pub(crate) prefixes: Vec<FragItem>,
    #[serde(default)]
    pub(crate) suffixes: Vec<FragItem>,
    /// Messages inserted before this original message.
    #[serde(default)]
    pub(crate) gap_before: Vec<MsgItem>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) struct InputInsertedMsg {
    pub(crate) role: serde_json::Value,
    pub(crate) content: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum MsgItem {
    /// Inline inserted message `{ role, content }` (backward compatible).
    Inline(InputInsertedMsg),
    /// Reference into `commit.msg_list`.
    Ref { r#ref: usize },
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum FragItem {
    /// Inline fragment string (backward compatible).
    Inline(String),
    /// Reference into `commit.frag_list`.
    Ref { r#ref: usize },
}

fn default_true() -> bool {
    true
}

#[derive(Clone, Debug)]
pub struct ChunkWitness {
    pub enabled: bool,
    pub len: u64,
    /// Packed little-endian bytes (already computed off-circuit).
    pub packed_le: u64,
}

#[derive(Clone, Debug)]
pub struct FragWitness {
    pub enabled: bool,
    pub total_len: u64,
    pub chunks: Vec<ChunkWitness>,
    /// Optional full Merkle authentication path (legacy).
    /// If empty and `RewriteTreeBatchWitness.frag_dedup_proof` is provided,
    /// the prover/verifier can expand this from the dedup proof.
    pub path: Vec<Digest>,
    /// Leaf index bits (LSB-first), length = tree height.
    pub index_bits: Vec<bool>,
}

#[derive(Clone, Debug)]
pub struct MsgInsertionWitness {
    pub enabled: bool,
    /// Leaf value: HashMsg(template_msg), computed off-circuit.
    pub leaf: Digest,
    /// Optional full Merkle authentication path (legacy).
    /// If empty and `RewriteTreeBatchWitness.msg_dedup_proof` is provided,
    /// the prover/verifier can expand this from the dedup proof.
    pub path: Vec<Digest>,
    /// Leaf index bits (LSB-first), length = tree height.
    pub index_bits: Vec<bool>,
}

#[derive(Clone, Debug)]
pub struct OrigMsgWitness {
    pub present: bool,
    pub role_id: u64,
    pub total_len: u64,
    pub chunks: Vec<ChunkWitness>,
}

#[derive(Clone, Debug)]
pub struct RewriteTreeBatchWitness {
    /// Raw Merkle roots. These stay private to each leaf proof; only their
    /// top-level salted commitments are exposed to recursion.
    pub msg_root: Digest,
    pub frag_root: Digest,
    /// One top-level randomizer per committed object. The same values must be
    /// used by every leaf in a recursively aggregated request proof.
    pub msg_randomness: Digest,
    pub frag_randomness: Digest,
    pub request_randomness: Digest,
    /// Present bit of the message right before this batch (cross-batch gap gating).
    /// For the first batch, set this to `true`.
    pub prev_present: bool,
    /// Original-conversation transcript accumulator entering this batch.
    pub user_acc_in: Digest,
    /// Transcript accumulator entering this batch.
    pub mod_acc_in: Digest,
    pub start_idx: u64,
    pub end_idx: u64,
    /// Rewritten/model-received request message-count transition. This counts
    /// every enabled complete gap message and every present wrapped original.
    pub rewritten_count_in: u64,
    pub rewritten_count_out: u64,

    /// Optional deduplicated Merkle proof data for all `msg_root` membership checks
    /// (gap insertions). If present, individual `MsgInsertionWitness.path` may be empty.
    pub msg_dedup_proof: Option<DedupMerkleProofWitness>,
    /// Optional deduplicated Merkle proof data for all `frag_root` membership checks
    /// (prefix/suffix fragments). If present, individual `FragWitness.path` may be empty.
    pub frag_dedup_proof: Option<DedupMerkleProofWitness>,

    pub orig: Vec<OrigMsgWitness>,
    pub gap_insertions: Vec<Vec<MsgInsertionWitness>>,
    pub prefixes: Vec<Vec<FragWitness>>,
    pub suffixes: Vec<Vec<FragWitness>>,
}

/// Roles are canonical field elements; accepting larger integers aliases roles.
pub fn parse_role_id(value: &serde_json::Value) -> Result<u64> {
    if let Some(role) = value.as_u64() {
        anyhow::ensure!(
            role < F::ORDER,
            "numeric role must be a canonical Goldilocks field element"
        );
        return Ok(role);
    }
    match value.as_str() {
        Some("system") => Ok(0),
        Some("user") => Ok(1),
        Some("assistant") => Ok(2),
        _ => anyhow::bail!("invalid role: expected system|user|assistant or a canonical u64"),
    }
}

/// Compute the final presence bit and both ordered transcript accumulators.
pub fn compute_roots(
    params: &RewriteParams,
    w: &RewriteTreeBatchWitness,
) -> Result<(bool, Digest, Digest)> {
    anyhow::ensure!(w.orig.len() == params.batch, "orig len mismatch");
    anyhow::ensure!(w.gap_insertions.len() == params.batch, "gap len mismatch");
    anyhow::ensure!(w.prefixes.len() == params.batch, "prefixes len mismatch");
    anyhow::ensure!(w.suffixes.len() == params.batch, "suffixes len mismatch");

    let msg_root = digest_to_hashout(w.msg_root);
    let frag_root = digest_to_hashout(w.frag_root);

    let mut prev = w.prev_present;
    let mut last_present = w.prev_present;
    let mut present_count = 0u64;
    let mut rewritten_count = w.rewritten_count_in;

    let mut user_acc = digest_to_hashout(w.user_acc_in);
    let mut mod_acc = digest_to_hashout(w.mod_acc_in);

    // Pre-build dedup maps if present (faster than rebuilding per leaf).
    let msg_dedup = w
        .msg_dedup_proof
        .as_ref()
        .map(|p| (p.height, dedup_proof_map(p)));
    let frag_dedup = w
        .frag_dedup_proof
        .as_ref()
        .map(|p| (p.height, dedup_proof_map(p)));

    for i in 0..params.batch {
        for j in 0..params.max_gap_msgs {
            let ins = &w.gap_insertions[i][j];
            if ins.enabled {
                anyhow::ensure!(prev, "gap insertion enabled after prev_present=false");
                let leaf = digest_to_hashout(ins.leaf);
                let path_digest: Vec<Digest> = if ins.path.is_empty() && params.h_msg > 0 {
                    let (h, map) = msg_dedup.as_ref().ok_or_else(|| {
                        anyhow::anyhow!(
                            "gap path missing but msg_dedup_proof is None (i={i}, j={j})"
                        )
                    })?;
                    expand_path_from_dedup_map(*h, map, &ins.index_bits)?
                } else {
                    ins.path.clone()
                };
                let path: Vec<HashOut<F>> =
                    path_digest.iter().copied().map(digest_to_hashout).collect();
                let root_out = merkle_root(leaf, &path, &ins.index_bits);
                anyhow::ensure!(root_out == msg_root, "msg_root membership failed");
                mod_acc = transcript_step(mod_acc, leaf);
                rewritten_count = rewritten_count
                    .checked_add(1)
                    .ok_or_else(|| anyhow::anyhow!("rewritten message count overflow"))?;
            }
        }

        let om = &w.orig[i];
        if om.present {
            anyhow::ensure!(prev, "present after prev_present=false");
        }
        prev = om.present;
        last_present = om.present;
        if om.present {
            present_count = present_count
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("present message count overflow"))?;
            rewritten_count = rewritten_count
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("rewritten message count overflow"))?;
        }

        let role_id = f_u64(om.role_id);
        let total_len = f_u64(om.total_len);

        let mut rewritten_total: u64 = if om.present { om.total_len } else { 0 };
        for frag in w.prefixes[i]
            .iter()
            .chain(&w.suffixes[i])
            .filter(|frag| frag.enabled)
        {
            anyhow::ensure!(om.present, "frag enabled when msg not present");
            rewritten_total += frag.total_len;
        }

        // body_state over orig chunks
        let mut body_state = hash_bytes_init(total_len);
        let mut body_sum: u64 = 0;

        // rewritten bytes stream
        let mut bytes_state = hash_bytes_init(f_u64(rewritten_total));
        let mut sum_len: u64 = 0;

        let absorb_fragments = |fragments: &[FragWitness],
                                mut bytes_state: HashOut<F>|
         -> Result<(HashOut<F>, u64)> {
            let mut absorbed_len = 0;
            for (f, frag) in fragments.iter().enumerate() {
                if frag.enabled {
                    anyhow::ensure!(om.present, "frag enabled when msg not present");
                    let frag_total = frag.total_len;
                    let mut frag_state = hash_bytes_init(f_u64(frag_total));
                    let mut frag_sum: u64 = 0;
                    for k in 0..params.max_frag_chunks {
                        let ch = &frag.chunks[k];
                        if ch.enabled {
                            frag_sum += ch.len;
                            frag_state = absorb_chunk_bytes(
                                frag_state,
                                ch.len,
                                ch.packed_le,
                                params.chunk_bytes,
                            )?;
                            bytes_state = absorb_chunk_bytes(
                                bytes_state,
                                ch.len,
                                ch.packed_le,
                                params.chunk_bytes,
                            )?;
                            absorbed_len += ch.len;
                        }
                    }
                    anyhow::ensure!(frag_sum == frag_total, "frag len sum mismatch");
                    let path_digest: Vec<Digest> = if frag.path.is_empty() && params.h_frag > 0 {
                        let (h, map) = frag_dedup.as_ref().ok_or_else(|| {
                            anyhow::anyhow!(
                                "fragment path missing but frag_dedup_proof is None (i={i}, f={f})"
                            )
                        })?;
                        expand_path_from_dedup_map(*h, map, &frag.index_bits)?
                    } else {
                        frag.path.clone()
                    };
                    let path: Vec<HashOut<F>> =
                        path_digest.iter().copied().map(digest_to_hashout).collect();
                    let root_out = merkle_root(frag_state, &path, &frag.index_bits);
                    anyhow::ensure!(root_out == frag_root, "frag_root membership failed");
                }
            }
            Ok((bytes_state, absorbed_len))
        };
        let (state, prefix_len) = absorb_fragments(&w.prefixes[i], bytes_state)?;
        bytes_state = state;
        sum_len += prefix_len;

        // orig chunks
        for k in 0..params.max_orig_chunks {
            let ch = &om.chunks[k];
            if ch.enabled {
                anyhow::ensure!(om.present, "orig chunk enabled when msg not present");
                body_sum += ch.len;
                sum_len += ch.len;
                body_state =
                    absorb_chunk_bytes(body_state, ch.len, ch.packed_le, params.chunk_bytes)?;
                bytes_state =
                    absorb_chunk_bytes(bytes_state, ch.len, ch.packed_le, params.chunk_bytes)?;
            }
        }
        if om.present {
            anyhow::ensure!(body_sum == om.total_len, "orig len sum mismatch");
        }
        let (state, suffix_len) = absorb_fragments(&w.suffixes[i], bytes_state)?;
        bytes_state = state;
        sum_len += suffix_len;

        if om.present {
            anyhow::ensure!(rewritten_total == sum_len, "rewritten_total != sum_len");
        }

        let orig_leaf = hash_msg(role_id, total_len, body_state);
        let rewritten_leaf = hash_msg(role_id, f_u64(rewritten_total), bytes_state);

        if om.present {
            user_acc = transcript_step(user_acc, orig_leaf);
            mod_acc = transcript_step(mod_acc, rewritten_leaf);
        }
    }

    let expected_end_idx = w
        .start_idx
        .checked_add(present_count)
        .ok_or_else(|| anyhow::anyhow!("batch end index overflow"))?;
    anyhow::ensure!(
        w.end_idx == expected_end_idx,
        "batch interval does not match present message count"
    );
    anyhow::ensure!(
        w.rewritten_count_out == rewritten_count,
        "rewritten message-count transition mismatch"
    );
    anyhow::ensure!(w.start_idx <= u32::MAX as u64, "start_idx exceeds u32");
    anyhow::ensure!(w.end_idx <= u32::MAX as u64, "end_idx exceeds u32");
    anyhow::ensure!(
        w.rewritten_count_in <= u32::MAX as u64,
        "rewritten_count_in exceeds u32"
    );
    anyhow::ensure!(
        w.rewritten_count_out <= u32::MAX as u64,
        "rewritten_count_out exceeds u32"
    );

    Ok((
        last_present,
        hashout_to_digest(user_acc),
        hashout_to_digest(mod_acc),
    ))
}

fn zero_frag(max_chunks: usize, h_frag: usize) -> FragWitness {
    FragWitness {
        enabled: false,
        total_len: 0,
        chunks: (0..max_chunks)
            .map(|_| ChunkWitness {
                enabled: false,
                len: 0,
                packed_le: 0,
            })
            .collect(),
        path: vec![[0; DIGEST_WIDTH]; h_frag],
        index_bits: vec![false; h_frag],
    }
}

fn zero_gap(h_msg: usize) -> MsgInsertionWitness {
    MsgInsertionWitness {
        enabled: false,
        leaf: [0; DIGEST_WIDTH],
        path: vec![[0; DIGEST_WIDTH]; h_msg],
        index_bits: vec![false; h_msg],
    }
}

fn resolve_chunk_limit(specific: usize, legacy: usize, default: usize) -> usize {
    if specific > 0 {
        specific
    } else if legacy > 0 {
        legacy
    } else {
        default
    }
}

fn pack_le_u64(bytes: &[u8]) -> u64 {
    let mut out = 0u64;
    for (i, b) in bytes.iter().enumerate() {
        out |= (*b as u64) << (8 * i);
    }
    out
}

pub(crate) fn bytes_to_chunks(
    bytes: &[u8],
    chunk_bytes: usize,
    max_chunks: usize,
) -> anyhow::Result<(u64, Vec<ChunkWitness>)> {
    anyhow::ensure!(chunk_bytes > 0, "chunk_bytes must be > 0");
    anyhow::ensure!(
        chunk_bytes <= 8,
        "chunk_bytes must be <= 8 for this demo (packed_le is u64). got chunk_bytes={chunk_bytes}"
    );
    anyhow::ensure!(max_chunks > 0, "max_chunks must be > 0");
    anyhow::ensure!(
        bytes.len() <= chunk_bytes * max_chunks,
        "message too long: len={} > chunk_bytes*max_chunks={}",
        bytes.len(),
        chunk_bytes * max_chunks
    );

    let total_len = bytes.len() as u64;
    let mut chunks = Vec::with_capacity(max_chunks);
    for i in 0..max_chunks {
        let start = i * chunk_bytes;
        if start >= bytes.len() {
            chunks.push(ChunkWitness {
                enabled: false,
                len: 0,
                packed_le: 0,
            });
            continue;
        }
        let end = (start + chunk_bytes).min(bytes.len());
        let slice = &bytes[start..end];
        chunks.push(ChunkWitness {
            enabled: true,
            len: slice.len() as u64,
            packed_le: pack_le_u64(slice),
        });
    }
    Ok((total_len, chunks))
}

pub(crate) fn load_input(
    path: Option<&str>,
    defaults: InputParams,
) -> Result<(RewriteParams, Option<InputCommit>, Vec<InputMessage>)> {
    let input = if let Some(path) = path {
        serde_json::from_str::<InputFile>(&std::fs::read_to_string(path)?)?
    } else {
        InputFile {
            params: None,
            commit: None,
            trailing_gap: vec![],
            messages: [("user", "hello"), ("assistant", "world")]
                .into_iter()
                .map(|(role, content)| InputMessage {
                    role: serde_json::json!(role),
                    content: content.to_string(),
                    present: true,
                    prefixes: vec![],
                    suffixes: vec![],
                    gap_before: vec![],
                })
                .collect(),
        }
    };
    let pp = input.params.unwrap_or(defaults);
    let chunk_default = if path.is_some() { 4 } else { 1 };
    let params = RewriteParams {
        batch: pp.batch,
        chunk_bytes: pp.chunk_bytes,
        max_orig_chunks: resolve_chunk_limit(pp.max_orig_chunks, pp.max_chunks, chunk_default),
        max_msg_chunks: resolve_chunk_limit(pp.max_msg_chunks, pp.max_chunks, chunk_default),
        max_frag_chunks: resolve_chunk_limit(pp.max_frag_chunks, pp.max_chunks, chunk_default),
        msg_bank_len: 0,
        frag_bank_len: 0,
        h_msg: pp.h_msg,
        h_frag: pp.h_frag,
        max_frags: pp.max_frags,
        max_gap_msgs: pp.max_gap_msgs,
        fast_range_check: true,
    };
    let mut messages = input.messages;
    messages.push(InputMessage {
        role: serde_json::json!("system"),
        content: String::new(),
        present: false,
        prefixes: vec![],
        suffixes: vec![],
        gap_before: input.trailing_gap,
    });
    Ok((params, input.commit, messages))
}

pub(crate) fn resolve_message<'a>(
    item: &'a MsgItem,
    commit: Option<&'a InputCommit>,
) -> Result<&'a InputInsertedMsg> {
    match item {
        MsgItem::Inline(message) => Ok(message),
        MsgItem::Ref { r#ref } => commit
            .and_then(|bank| bank.msg_list.get(*r#ref))
            .ok_or_else(|| anyhow::anyhow!("missing committed message at index {}", r#ref)),
    }
}

pub(crate) fn resolve_fragment<'a>(
    item: &'a FragItem,
    commit: Option<&'a InputCommit>,
) -> Result<&'a str> {
    match item {
        FragItem::Inline(fragment) => Ok(fragment.as_str()),
        FragItem::Ref { r#ref } => commit
            .and_then(|bank| bank.frag_list.get(*r#ref))
            .map(String::as_str)
            .ok_or_else(|| anyhow::anyhow!("missing committed fragment at index {}", r#ref)),
    }
}

pub(crate) fn message_leaf(message: &InputInsertedMsg, params: &RewriteParams) -> Result<Digest> {
    let role = parse_role_id(&message.role)?;
    let bytes = message.content.as_bytes();
    bytes_to_chunks(bytes, params.chunk_bytes, params.max_msg_chunks)?;
    Ok(commitment::hash_msg_digest(
        role,
        bytes.len() as u64,
        commitment::hash_bytes_digest(bytes),
    ))
}

fn pad<T: Clone>(values: &mut Vec<T>, capacity: usize, padding: T) -> Result<()> {
    anyhow::ensure!(
        values.len() <= capacity,
        "witness capacity exceeded: {} > {capacity}",
        values.len()
    );
    values.resize(capacity, padding);
    Ok(())
}

fn absent_original(params: &RewriteParams) -> OrigMsgWitness {
    OrigMsgWitness {
        present: false,
        role_id: 0,
        total_len: 0,
        chunks: vec![
            ChunkWitness {
                enabled: false,
                len: 0,
                packed_le: 0
            };
            params.max_orig_chunks
        ],
    }
}

fn fragment_witnesses(
    items: &[FragItem],
    params: &RewriteParams,
    commit: Option<&InputCommit>,
    banks: &RewriteBanks,
    strict: bool,
    cursor: &mut usize,
) -> Result<Vec<FragWitness>> {
    let mut fragments = items
        .iter()
        .map(|item| {
            let text = resolve_fragment(item, commit)?;
            let (total_len, chunks) =
                bytes_to_chunks(text.as_bytes(), params.chunk_bytes, params.max_frag_chunks)?;
            let leaf = commitment::hash_bytes_digest(text.as_bytes());
            let reference = match item {
                FragItem::Ref { r#ref } => Some(*r#ref),
                FragItem::Inline(_) => None,
            };
            let (path, index_bits) =
                banks
                    .fragment
                    .opening(leaf, reference, commit.is_some(), strict, cursor)?;
            Ok(FragWitness {
                enabled: true,
                total_len,
                chunks,
                path,
                index_bits,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    pad(
        &mut fragments,
        params.max_frags,
        zero_frag(params.max_frag_chunks, params.h_frag),
    )?;
    Ok(fragments)
}

pub(crate) struct PreparedBatch {
    pub witness: RewriteTreeBatchWitness,
    pub last_present: bool,
    pub user_root: Digest,
    pub mod_root: Digest,
}

pub(crate) fn prepare_batches(
    params: &RewriteParams,
    commit: Option<&InputCommit>,
    messages: &[InputMessage],
    banks: &RewriteBanks,
    strict: bool,
    dedup: bool,
) -> Result<Vec<PreparedBatch>> {
    anyhow::ensure!(params.batch > 0, "batch must be positive");
    let mut gap_cursor = 0;
    let mut frag_cursor = 0;
    let mut prev_present = true;
    let mut user_acc_in = commitment::transcript_init();
    let mut mod_acc_in = commitment::transcript_init();
    let mut present_cursor = 0u64;
    let mut rewritten_cursor = 0u64;
    let mut batches = Vec::new();
    for messages in messages.chunks(params.batch) {
        let mut orig = Vec::new();
        let mut gap_insertions = Vec::new();
        let mut prefixes = Vec::new();
        let mut suffixes = Vec::new();
        for message in messages {
            anyhow::ensure!(
                message.present || (message.prefixes.is_empty() && message.suffixes.is_empty()),
                "absent message has fragments"
            );
            orig.push(if message.present {
                let (total_len, chunks) = bytes_to_chunks(
                    message.content.as_bytes(),
                    params.chunk_bytes,
                    params.max_orig_chunks,
                )?;
                OrigMsgWitness {
                    present: true,
                    role_id: parse_role_id(&message.role)?,
                    total_len,
                    chunks,
                }
            } else {
                absent_original(params)
            });
            let mut gaps = message
                .gap_before
                .iter()
                .map(|item| {
                    let leaf = message_leaf(resolve_message(item, commit)?, params)?;
                    let reference = match item {
                        MsgItem::Ref { r#ref } => Some(*r#ref),
                        MsgItem::Inline(_) => None,
                    };
                    let (path, index_bits) = banks.message.opening(
                        leaf,
                        reference,
                        commit.is_some(),
                        strict,
                        &mut gap_cursor,
                    )?;
                    Ok(MsgInsertionWitness {
                        enabled: true,
                        leaf,
                        path,
                        index_bits,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            pad(&mut gaps, params.max_gap_msgs, zero_gap(params.h_msg))?;
            gap_insertions.push(gaps);
            prefixes.push(fragment_witnesses(
                &message.prefixes,
                params,
                commit,
                banks,
                strict,
                &mut frag_cursor,
            )?);
            suffixes.push(fragment_witnesses(
                &message.suffixes,
                params,
                commit,
                banks,
                strict,
                &mut frag_cursor,
            )?);
        }
        pad(&mut orig, params.batch, absent_original(params))?;
        pad(
            &mut gap_insertions,
            params.batch,
            vec![zero_gap(params.h_msg); params.max_gap_msgs],
        )?;
        let empty_fragments =
            vec![zero_frag(params.max_frag_chunks, params.h_frag); params.max_frags];
        pad(&mut prefixes, params.batch, empty_fragments.clone())?;
        pad(&mut suffixes, params.batch, empty_fragments)?;
        let (msg_dedup_proof, frag_dedup_proof) = if dedup {
            let msg_paths = gap_insertions
                .iter()
                .flatten()
                .filter(|item| item.enabled)
                .map(|item| (item.path.clone(), item.index_bits.clone()))
                .collect::<Vec<_>>();
            let frag_paths = prefixes
                .iter()
                .chain(&suffixes)
                .flatten()
                .filter(|item| item.enabled)
                .map(|item| (item.path.clone(), item.index_bits.clone()))
                .collect::<Vec<_>>();
            let msg_proof = build_dedup_proof_from_paths(params.h_msg, &msg_paths)?;
            let frag_proof = build_dedup_proof_from_paths(params.h_frag, &frag_paths)?;
            for item in gap_insertions.iter_mut().flatten() {
                item.path.clear();
            }
            for item in prefixes.iter_mut().chain(&mut suffixes).flatten() {
                item.path.clear();
            }
            (Some(msg_proof), Some(frag_proof))
        } else {
            (None, None)
        };
        let present_count = orig.iter().filter(|item| item.present).count() as u64;
        let gap_count = gap_insertions
            .iter()
            .flatten()
            .filter(|item| item.enabled)
            .count() as u64;
        let end_idx = present_cursor
            .checked_add(present_count)
            .ok_or_else(|| anyhow::anyhow!("message count overflow"))?;
        let rewritten_count_out = rewritten_cursor
            .checked_add(present_count)
            .and_then(|count| count.checked_add(gap_count))
            .ok_or_else(|| anyhow::anyhow!("rewritten message count overflow"))?;
        let witness = RewriteTreeBatchWitness {
            msg_root: banks.message.root,
            frag_root: banks.fragment.root,
            msg_randomness: banks.msg_randomness,
            frag_randomness: banks.frag_randomness,
            request_randomness: banks.request_randomness,
            prev_present,
            user_acc_in,
            mod_acc_in,
            start_idx: present_cursor,
            end_idx,
            rewritten_count_in: rewritten_cursor,
            rewritten_count_out,
            msg_dedup_proof,
            frag_dedup_proof,
            orig,
            gap_insertions,
            prefixes,
            suffixes,
        };
        let (last_present, user_root, mod_root) = compute_roots(params, &witness)?;
        prev_present = last_present;
        user_acc_in = user_root;
        mod_acc_in = mod_root;
        present_cursor = end_idx;
        rewritten_cursor = rewritten_count_out;
        batches.push(PreparedBatch {
            witness,
            last_present,
            user_root,
            mod_root,
        });
    }
    Ok(batches)
}
