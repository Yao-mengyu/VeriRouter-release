use anyhow::Result;
use clap::Parser;
use std::fs;
use std::path::PathBuf;

use veriroute_route::proof::RouteVerificationPolicy;
use veriroute_route::proof::{
    run_trace, verify_final_proof_artifact, verify_final_proof_artifact_with_policy, RunOptions,
};

#[derive(Parser, Debug)]
#[command(version, about = "VeriRouteRoute recursive proof runner")]
struct Args {
    #[arg(long)]
    trace: Option<PathBuf>,
    #[arg(long)]
    verifier_data: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    modules_only: bool,
    #[arg(long, default_value_t = false)]
    verify_subproofs: bool,
    #[arg(long, default_value_t = false)]
    parallel_subproofs: bool,
    #[arg(long, default_value_t = false)]
    zk: bool,
    #[arg(long)]
    report: Option<PathBuf>,
    #[arg(long)]
    verify_report: Option<PathBuf>,
    #[arg(long)]
    proof: Option<PathBuf>,
    #[arg(long)]
    verify_policy: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    no_save_proof: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(verify_report) = args.verify_report.as_deref() {
        if let Some(vk_path) = args.verifier_data.as_deref() {
            let policy_path = args
                .verify_policy
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("public verification requires --verify-policy"))?;
            let policy: RouteVerificationPolicy = serde_json::from_slice(&fs::read(policy_path)?)?;
            let report = veriroute_route::proof::verify_public_route_artifact(
                verify_report,
                args.proof.as_deref(),
                vk_path,
                &policy,
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            return Ok(());
        }
        let trace = args
            .trace
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("diagnostic reconstruction requires --trace"))?;
        let report = if let Some(policy_path) = args.verify_policy.as_deref() {
            let policy: RouteVerificationPolicy = serde_json::from_slice(&fs::read(policy_path)?)?;
            verify_final_proof_artifact_with_policy(
                trace,
                verify_report,
                args.proof.as_deref(),
                &policy,
            )?
        } else {
            eprintln!(
                "warning: unpinned route verification is for development only; pass --verify-policy for a trust-boundary verification"
            );
            verify_final_proof_artifact(trace, verify_report, args.proof.as_deref())?
        };
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    let report = run_trace(
        args.trace
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("proving requires --trace"))?,
        args.zk,
        RunOptions {
            modules_only: args.modules_only,
            verify_subproofs: args.verify_subproofs,
            parallel_subproofs: args.parallel_subproofs,
            save_final_proof: !args.no_save_proof,
        },
        args.report.as_deref(),
    )?;
    if args.report.is_none() {
        println!("{}", serde_json::to_string_pretty(&report)?);
    }
    Ok(())
}
