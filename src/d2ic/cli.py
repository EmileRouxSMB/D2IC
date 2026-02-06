from __future__ import annotations

import argparse
from pathlib import Path

from .pipelines import run_sequence_from_config


def build_parser() -> argparse.ArgumentParser:
    """Build the d2ic CLI parser."""
    parser = argparse.ArgumentParser(prog="d2ic", description="D2IC command-line interface.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_seq = subparsers.add_parser(
        "run-sequence", help="Run the sequential DIC pipeline from a TOML config."
    )
    run_seq.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a TOML config file (e.g., examples/configs/sequential_dic.toml).",
    )
    run_seq.set_defaults(_handler=_handle_run_sequence)

    return parser


def _handle_run_sequence(args: argparse.Namespace) -> None:
    run_sequence_from_config(args.config)


def main() -> None:
    """Entry point for the d2ic CLI."""
    parser = build_parser()
    args = parser.parse_args()
    handler = getattr(args, "_handler", None)
    if handler is None:
        parser.error("No command selected.")
    handler(args)


if __name__ == "__main__":
    main()
