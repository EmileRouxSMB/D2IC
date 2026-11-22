"""Thin wrapper for running the sequential DIC pipeline from a TOML config."""

from __future__ import annotations

from pathlib import Path

from d2ic.pipelines import run_sequence_from_config


def main() -> None:
    cfg_path = Path(__file__).resolve().parent / "configs" / "sequential_dic.toml"
    run_sequence_from_config(cfg_path)


if __name__ == "__main__":
    main()
