#!/usr/bin/env python3
"""Single command-line entry point for VeriRouter artifacts."""
import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run VeriRouter demos, route experiments, and certificate verification."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("demo", "Run a component or multi-turn demo"),
        ("experiment", "Prepare route inputs and run proof experiments"),
        ("chain", "Build certificates and verify full or pinned chains"),
        ("signature", "Issue and verify serving certificates"),
    ]:
        commands.add_parser(name, help=help_text, add_help=False)
    args, rest = parser.parse_known_args(argv)
    if args.command == "demo":
        from scripts.demo import main as run
    elif args.command == "experiment":
        sys.path.insert(
            0, str(Path(__file__).resolve().parent / "control-plane/zkroute/python")
        )
        from experiment import main as run
    elif args.command == "chain":
        from scripts.chain import main as run
    else:
        sys.path.insert(
            0, str(Path(__file__).resolve().parent / "data-plane/signature/python")
        )
        from execution_certificate import main as run
    run(rest)


if __name__ == "__main__":
    main()
