use anyhow::Result;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Field64;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use veriroute_rewrite::commitment;
use veriroute_rewrite::util::Digest;
use veriroute_rewrite::witness;

#[derive(Clone, Debug, Deserialize)]
struct ExecutedMessage {
    role: Value,
    content: String,
}

#[derive(Clone, Debug, Serialize)]
struct RootReport {
    root_type: String,
    version: String,
    chunk_bytes: usize,
    message_count: usize,
    /// `ReqHash = H(DS_REQ_FINAL || message_count || transcript_final_fold)`.
    /// This is the complete request digest that must be bound into the provider
    /// execution certificate.
    request_hash_hashout: Digest,
    /// `Commit(ReqHash; request_randomness)`.  The randomness remains local to
    /// the caller and is deliberately not serialized in this report.
    request_commitment: Digest,
}

fn parse_flag_string(argv: &[String], key: &str) -> Option<String> {
    let mut i = 0usize;
    while i < argv.len() {
        if argv[i] == key {
            return argv.get(i + 1).cloned();
        }
        i += 1;
    }
    None
}

fn parse_flag_usize(argv: &[String], key: &str, default: usize) -> usize {
    let mut i = 0usize;
    while i < argv.len() {
        if argv[i] == key {
            return argv
                .get(i + 1)
                .unwrap_or_else(|| panic!("missing value for {key}"))
                .parse()
                .unwrap_or_else(|_| panic!("invalid value for {key}"));
        }
        i += 1;
    }
    default
}

/// Parse the private randomness supplied by the request originator.  Accept
/// either a JSON array (for example, `[1,2,3,4]`) or a comma-separated list
/// (`1,2,3,4`) so that shell callers do not need to create a temporary file.
fn parse_request_randomness(raw: &str) -> Result<Digest> {
    let limbs: Vec<u64> = if raw.trim_start().starts_with('[') {
        serde_json::from_str(raw).map_err(|err| {
            anyhow::anyhow!(
                "invalid --request-randomness JSON array (expected [u64,u64,u64,u64]): {err}"
            )
        })?
    } else {
        raw.split(',')
            .map(|part| {
                part.trim().parse::<u64>().map_err(|err| {
                    anyhow::anyhow!(
                        "invalid --request-randomness limb {part:?} (expected u64): {err}"
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?
    };

    anyhow::ensure!(
        limbs.len() == 4,
        "--request-randomness must contain exactly four limbs"
    );
    for (index, limb) in limbs.iter().enumerate() {
        anyhow::ensure!(
            *limb < GoldilocksField::ORDER,
            "--request-randomness limb {index} is not a canonical Goldilocks field element"
        );
    }
    Ok([limbs[0], limbs[1], limbs[2], limbs[3]])
}

fn parse_messages(path: &str) -> Result<Vec<ExecutedMessage>> {
    let raw = std::fs::read_to_string(path)?;
    let value: Value = serde_json::from_str(&raw)?;
    if let Some(items) = value.as_array() {
        return Ok(serde_json::from_value(Value::Array(items.clone()))?);
    }
    if let Some(items) = value.get("messages") {
        return Ok(serde_json::from_value(items.clone())?);
    }
    anyhow::bail!("expected a JSON array of executed messages or an object containing messages[]");
}

fn main() -> Result<()> {
    let argv: Vec<String> = std::env::args().collect();
    let input_path =
        parse_flag_string(&argv, "--input").ok_or_else(|| anyhow::anyhow!("missing --input"))?;
    let request_randomness = parse_flag_string(&argv, "--request-randomness")
        .ok_or_else(|| anyhow::anyhow!("missing --request-randomness"))
        .and_then(|raw| parse_request_randomness(&raw))?;
    let out_path = parse_flag_string(&argv, "--out");
    let chunk_bytes = parse_flag_usize(&argv, "--chunk-bytes", 8);
    anyhow::ensure!(chunk_bytes > 0, "chunk_bytes must be > 0");
    anyhow::ensure!(
        chunk_bytes <= 8,
        "chunk_bytes must be <= 8 to match Rewrite packed_le semantics"
    );

    let messages = parse_messages(&input_path)?;
    anyhow::ensure!(
        messages.len() <= u32::MAX as usize,
        "received request message count exceeds the rewrite circuit's u32 capacity"
    );
    let mut mod_root = commitment::transcript_init();
    for message in &messages {
        let role_id = witness::parse_role_id(&message.role)?;
        let content_hash = commitment::hash_bytes_digest(message.content.as_bytes());
        let leaf = commitment::hash_msg_digest(
            role_id,
            message.content.as_bytes().len() as u64,
            content_hash,
        );
        // Transcript compression uses the same capacity-domain Poseidon relation
        // as the rewrite circuit.
        mod_root = commitment::transcript_step_digest(mod_root, leaf);
    }
    let request_hash = commitment::request_hash_digest(messages.len() as u64, mod_root);
    let request_commitment = commitment::commit_digest(request_hash, request_randomness);

    let report = RootReport {
        root_type: "verirouter_received_request_commitment".to_string(),
        version: "promptmod_exec_root_v5_request_commitment_hashout4".to_string(),
        chunk_bytes,
        message_count: messages.len(),
        request_hash_hashout: request_hash,
        request_commitment,
    };
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = out_path {
        std::fs::write(path, json)?;
    } else {
        println!("{json}");
    }
    Ok(())
}
