//! Check the message and fragment banks used by rewrite commitments.
//! The checked banks also provide Merkle roots and witness openings.

use crate::circuits::batch::RewriteParams;
use crate::commitment;
use crate::util::{Digest, DIGEST_WIDTH};
use crate::witness::{
    bytes_to_chunks, message_leaf, resolve_fragment, resolve_message, InputCommit, InputMessage,
};
use anyhow::Result;
use std::io::Read;

fn os_random_digest() -> Result<Digest> {
    let mut bytes = [0u8; 32];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(std::array::from_fn(|index| {
        let mut limb = [0u8; 8];
        limb.copy_from_slice(&bytes[index * 8..(index + 1) * 8]);
        u64::from_le_bytes(limb) & 0x7fff_ffff_ffff_ffff
    }))
}

fn zeros_path(height: usize) -> (Vec<Digest>, Vec<bool>) {
    (vec![[0; DIGEST_WIDTH]; height], vec![false; height])
}

fn merkle_root_and_paths(
    leaves: &[Digest],
    height: usize,
) -> anyhow::Result<(Digest, Vec<Vec<Digest>>, Vec<Vec<bool>>)> {
    // Build a full binary tree of 2^height leaves, padding missing leaves with 0.
    let n = 1usize
        .checked_shl(height as u32)
        .ok_or_else(|| anyhow::anyhow!("height too large"))?;
    anyhow::ensure!(
        leaves.len() <= n,
        "too many leaves={} for height={} (max={})",
        leaves.len(),
        height,
        n
    );

    // levels[0] = leaves, levels[height][0] = root
    let mut levels: Vec<Vec<Digest>> = Vec::with_capacity(height + 1);
    let mut level0 = vec![[0; DIGEST_WIDTH]; n];
    for (i, v) in leaves.iter().enumerate() {
        level0[i] = *v;
    }
    levels.push(level0);
    for l in 0..height {
        let prev = &levels[l];
        let mut next = Vec::with_capacity(prev.len() / 2);
        for pair in prev.chunks(2) {
            next.push(commitment::hash_node_digest(pair[0], pair[1]));
        }
        levels.push(next);
    }
    anyhow::ensure!(levels[height].len() == 1, "invalid merkle build");
    let root = levels[height][0];

    let mut paths: Vec<Vec<Digest>> = vec![vec![[0; DIGEST_WIDTH]; height]; leaves.len()];
    let mut bits: Vec<Vec<bool>> = vec![vec![false; height]; leaves.len()];
    for i in 0..leaves.len() {
        let mut idx = i;
        for l in 0..height {
            let sib = idx ^ 1;
            paths[i][l] = levels[l][sib];
            bits[i][l] = (idx & 1) == 1; // true => current is right child
            idx >>= 1;
        }
    }
    Ok((root, paths, bits))
}

pub(crate) struct MerkleBank {
    pub root: Digest,
    leaves: Vec<Digest>,
    paths: Vec<Vec<Digest>>,
    bits: Vec<Vec<bool>>,
    height: usize,
}

impl MerkleBank {
    fn new(leaves: Vec<Digest>, height: usize) -> Result<Self> {
        let (root, paths, bits) = if leaves.is_empty() {
            ([0; DIGEST_WIDTH], vec![], vec![])
        } else {
            merkle_root_and_paths(&leaves, height)?
        };
        Ok(Self {
            root,
            leaves,
            paths,
            bits,
            height,
        })
    }

    pub(crate) fn opening(
        &self,
        leaf: Digest,
        reference: Option<usize>,
        explicit_bank: bool,
        strict: bool,
        cursor: &mut usize,
    ) -> Result<(Vec<Digest>, Vec<bool>)> {
        let index = if explicit_bank {
            reference.or_else(|| self.leaves.iter().position(|&value| value == leaf))
        } else {
            let index = *cursor;
            *cursor += 1;
            Some(index)
        };
        if let Some(index) = index {
            if let (Some(path), Some(bits)) = (self.paths.get(index), self.bits.get(index)) {
                return Ok((path.clone(), bits.clone()));
            }
        }
        anyhow::ensure!(
            !strict && explicit_bank,
            "item has no opening in the committed bank"
        );
        Ok(zeros_path(self.height))
    }
}

pub(crate) struct RewriteBanks {
    pub message: MerkleBank,
    pub fragment: MerkleBank,
    pub msg_randomness: Digest,
    pub frag_randomness: Digest,
    pub request_randomness: Digest,
}

pub(crate) fn check_commit_data(
    params: &mut RewriteParams,
    commit: Option<&InputCommit>,
    messages: &[InputMessage],
) -> Result<RewriteBanks> {
    let (msg_randomness, frag_randomness, request_randomness) = if let Some(commit) = commit {
        (
            commit.msg_randomness,
            commit.frag_randomness,
            commit.request_randomness,
        )
    } else {
        (
            os_random_digest()?,
            os_random_digest()?,
            os_random_digest()?,
        )
    };
    let inserted = if let Some(commit) = commit {
        commit.msg_list.iter().collect::<Vec<_>>()
    } else {
        messages
            .iter()
            .flat_map(|message| &message.gap_before)
            .map(|item| resolve_message(item, None))
            .collect::<Result<Vec<_>>>()?
    };
    let fragments = if let Some(commit) = commit {
        commit
            .frag_list
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
    } else {
        messages
            .iter()
            .flat_map(|message| message.prefixes.iter().chain(&message.suffixes))
            .map(|item| resolve_fragment(item, None))
            .collect::<Result<Vec<_>>>()?
    };
    let message = MerkleBank::new(
        inserted
            .into_iter()
            .map(|item| message_leaf(item, params))
            .collect::<Result<Vec<_>>>()?,
        params.h_msg,
    )?;
    let fragment = MerkleBank::new(
        fragments
            .into_iter()
            .map(|text| {
                bytes_to_chunks(text.as_bytes(), params.chunk_bytes, params.max_frag_chunks)?;
                Ok(commitment::hash_bytes_digest(text.as_bytes()))
            })
            .collect::<Result<Vec<_>>>()?,
        params.h_frag,
    )?;
    params.msg_bank_len = message.leaves.len();
    params.frag_bank_len = fragment.leaves.len();
    Ok(RewriteBanks {
        message,
        fragment,
        msg_randomness,
        frag_randomness,
        request_randomness,
    })
}
