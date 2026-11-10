"""Command-line dispatch for the world-state experiment. Argument parsing only."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import analysis, audit, data, runner
from .contracts import (
    CACHE_DIR,
    CONDITION_CHOICES,
    DEFAULT_ANSWER_TOKENS,
    DEFAULT_API_KEY_ENV,
    DEFAULT_AUDIT_N,
    DEFAULT_CONCURRENCY,
    DEFAULT_DATASET,
    DEFAULT_ENDPOINT,
    DEFAULT_EXTRACT_TOKENS,
    DEFAULT_MAX_SIDE,
    DEFAULT_SERVED_MODEL,
    DEFAULT_SPLIT,
    DEFAULT_TIMEOUT_SECONDS,
    GPUS,
    PROFILES,
)


def main(argv=None):
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(
        description="World-state graph experiment. Commands: prepare, run, freeze, analyze, audit {make, import, outcomes}.",
        formatter_class=formatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Sample profiles and write the prepare artifact", formatter_class=formatter)
    prepare.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="MMSI-Bench parquet")
    prepare.add_argument("--split", type=Path, default=DEFAULT_SPLIT, help="explore/confirm split")
    prepare.add_argument("--out", type=Path, required=True)

    run = sub.add_parser("run", help="Run raw and evidence conditions", formatter_class=formatter)
    run.add_argument("--prepare-dir", type=Path, required=True)
    run.add_argument("--profile", choices=PROFILES, required=True)
    run.add_argument(
        "--conditions",
        choices=CONDITION_CHOICES,
        required=True,
        metavar="{raw|flat,graph|raw,flat,graph}",
    )
    run.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="OpenAI-compatible base URL")
    run.add_argument("--served-model", default=DEFAULT_SERVED_MODEL, help="name reported by /v1/models")
    run.add_argument("--server-manifest", type=Path, required=True)
    run.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV, help="environment variable holding the server token")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--cache", type=Path, default=CACHE_DIR, help="shared terminal-result cache")
    run.add_argument("--extract-tokens", type=int, default=DEFAULT_EXTRACT_TOKENS, help="extraction max tokens")
    run.add_argument("--answer-tokens", type=int, default=DEFAULT_ANSWER_TOKENS, help="final-answer max tokens")
    run.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE, help="maximum image side in pixels")
    run.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="request timeout in seconds")
    run.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="in-flight request limit")
    run.add_argument("--freeze", type=Path, help="protocol freeze artifact")
    run.add_argument(
        "--resume",
        action="store_true",
        help="continue a saved run; a new server session is allowed when the model and inference settings match",
    )
    run.add_argument("--mock", action="store_true", help="use the mock backend and mark outputs synthetic")

    freeze = sub.add_parser("freeze", help="Freeze the protocol and choose the active profile", formatter_class=formatter)
    freeze.add_argument("--dev-run", type=Path, required=True)
    freeze.add_argument("--timing-run", type=Path, required=True)
    freeze.add_argument("--synthetic-run", type=Path, required=True)
    freeze.add_argument("--concurrency-run", type=Path)
    freeze.add_argument("--allocation-su", type=float, required=True)
    freeze.add_argument("--gpu", choices=GPUS, required=True)
    freeze.add_argument("--walltime-hours", type=float, required=True)
    freeze.add_argument("--deadline", required=True, help="YYYY-MM-DD")
    freeze.add_argument("--out", type=Path, required=True)

    analyze = sub.add_parser("analyze", help="Score a run and write the analysis tables", formatter_class=formatter)
    analyze.add_argument("--run", type=Path, required=True)
    analyze.add_argument("--raw-run", type=Path)
    analyze.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="MMSI-Bench parquet")
    analyze.add_argument("--out", type=Path, required=True)

    audit_parser = sub.add_parser("audit", help="State audit: make, import, outcomes", formatter_class=formatter)
    audit_sub = audit_parser.add_subparsers(dest="audit_command", required=True)

    make = audit_sub.add_parser("make", help="Draw a blinded audit packet", formatter_class=formatter)
    make.add_argument("--run", type=Path, required=True)
    make.add_argument("--n", type=int, default=DEFAULT_AUDIT_N, help="audit sample size")
    make.add_argument("--out", type=Path, required=True)

    import_labels = audit_sub.add_parser("import", help="Import human state labels", formatter_class=formatter)
    import_labels.add_argument("--packet", type=Path, required=True)
    import_labels.add_argument("--labels", type=Path, required=True)

    outcomes = audit_sub.add_parser("outcomes", help="List corrected and introduced items", formatter_class=formatter)
    outcomes.add_argument("--run", type=Path, required=True)
    outcomes.add_argument("--analysis", type=Path, required=True)
    outcomes.add_argument("--out", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        data.cli_prepare(args)
    elif args.command == "run":
        runner.cli_run(args)
    elif args.command == "freeze":
        runner.cli_freeze(args)
    elif args.command == "analyze":
        analysis.cli_analyze(args)
    elif args.command == "audit" and args.audit_command == "make":
        audit.cli_audit_make(args)
    elif args.command == "audit" and args.audit_command == "import":
        audit.cli_audit_import(args)
    elif args.command == "audit" and args.audit_command == "outcomes":
        audit.cli_audit_outcomes(args)
    else:
        parser.error(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    main()
