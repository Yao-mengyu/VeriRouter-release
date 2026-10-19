use anyhow::Result;
use veriroute_rewrite::proof::{run_input, verify_rewrite_report, RunOptions};
use veriroute_rewrite::witness::InputParams;

fn value<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter()
        .position(|arg| arg == key)
        .and_then(|index| args.get(index + 1))
        .map(String::as_str)
}

fn number(args: &[String], key: &str, default: usize) -> Result<usize> {
    match value(args, key) {
        Some(value) => Ok(value.parse()?),
        None if args.iter().any(|arg| arg == key) => anyhow::bail!("missing value for {key}"),
        None => Ok(default),
    }
}

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    let flag = |name: &str| args.iter().any(|arg| arg == name);
    if let Some(path) = value(&args, "--verify-report") {
        let policy = value(&args, "--verify-policy")
            .ok_or_else(|| anyhow::anyhow!("--verify-report requires --verify-policy"))?;
        let report = verify_rewrite_report(
            path,
            value(&args, "--proof"),
            policy,
            value(&args, "--verifier-data"),
        )?;
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    let input = value(&args, "--input");
    let defaults = InputParams {
        batch: number(&args, "--batch", if input.is_some() { 16 } else { 2 })?,
        chunk_bytes: number(&args, "--chunk-bytes", 8)?,
        max_chunks: number(&args, "--max-chunks", 0)?,
        max_orig_chunks: number(&args, "--max-orig-chunks", 0)?,
        max_msg_chunks: number(&args, "--max-msg-chunks", 0)?,
        max_frag_chunks: number(&args, "--max-frag-chunks", 0)?,
        h_msg: number(&args, "--h-msg", if input.is_some() { 8 } else { 1 })?,
        h_frag: number(&args, "--h-frag", if input.is_some() { 8 } else { 1 })?,
        max_frags: number(&args, "--max-frags", 1)?,
        max_gap_msgs: number(&args, "--max-gap-msgs", 1)?,
    };
    let path = value(&args, "--report-json");
    let report = run_input(
        input,
        defaults,
        RunOptions {
            public_orig: flag("--public-orig"),
            print_public_orig: flag("--print-public-orig"),
            strict_commit: flag("--strict-commit"),
            do_agg: !flag("--no-agg"),
            fan_in: number(&args, "--fan-in", 2)?,
            verify_leaves: flag("--verify-leaves"),
            dedup_merkle: flag("--dedup-merkle"),
            zero_knowledge: !flag("--no-zk"),
            save_proof: !flag("--no-save-proof"),
            print_level_sizes: flag("--print-level-sizes"),
        },
        path,
    )?;
    if let Some(path) = path {
        let path = std::path::Path::new(path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
    }
    Ok(())
}
