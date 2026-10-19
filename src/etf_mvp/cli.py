from __future__ import annotations

import argparse
import logging

from etf_mvp.config import AppConfig, load_config
from etf_mvp.data.kinds import DataKind
from etf_mvp.data.registry import available_providers
from etf_mvp.pipeline import fetch_data, run_pipeline, train_models

KIND_CHOICES = [kind.value for kind in DataKind]


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--provider",
        choices=available_providers(DataKind.MARKET),
        help="Market data provider (shortcut for --source market=<provider>)",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="KIND=PROVIDER",
        help="Override a kind's provider, e.g. --source macro=fred (repeatable)",
    )


def _apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    overrides: dict[str, str] = {}
    if getattr(args, "provider", None):
        overrides[DataKind.MARKET.value] = args.provider
    for item in getattr(args, "source", []):
        kind, _, provider = item.partition("=")
        if kind not in KIND_CHOICES or not provider:
            raise SystemExit(f"--source expects KIND=PROVIDER with KIND in {KIND_CHOICES}")
        if provider not in available_providers(DataKind(kind)):
            raise SystemExit(
                f"Unknown {kind} provider '{provider}': {available_providers(DataKind(kind))}"
            )
        overrides[kind] = provider
    return config.with_providers(overrides)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="etf-mvp",
        description="US ETF prediction MVP. Research output only; no order execution.",
    )
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Fetch all enabled kinds, train, predict")
    _add_source_arguments(run_parser)

    fetch_parser = subparsers.add_parser("fetch", help="Fetch and persist data")
    _add_source_arguments(fetch_parser)
    fetch_parser.add_argument(
        "--kind",
        action="append",
        choices=KIND_CHOICES,
        help="Only fetch these kinds (repeatable; default: every enabled kind)",
    )
    subparsers.add_parser("train", help="Train from persisted data")
    subparsers.add_parser("sources", help="List data kinds and their configured providers")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    config = _apply_overrides(load_config(args.config), args)

    if args.command == "sources":
        for kind in DataKind:
            source = config.source(kind)
            state = "on " if source.enabled else "off"
            print(
                f"{state} {kind.value:<13} {kind.label:<6} provider={source.provider:<16} "
                f"available={','.join(available_providers(kind))}"
            )
        return
    if args.command == "fetch":
        kinds = [DataKind(kind) for kind in args.kind] if args.kind else None
        bundle = fetch_data(config, kinds)
        for kind, frame in bundle.frames.items():
            print(f"{kind.value:<13} {len(frame):>9,} rows  ({config.source(kind).provider})")
        return
    predictions = train_models(config) if args.command == "train" else run_pipeline(config)
    print("Research predictions (not investment advice):")
    print(predictions.to_string(index=False))
    print(f"Artifacts: {config.output.artifacts_dir}")


if __name__ == "__main__":
    main()
