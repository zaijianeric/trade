from __future__ import annotations

import argparse
import logging
from pathlib import Path

from etf_mvp.config import AppConfig, load_config
from etf_mvp.data.kinds import DataKind
from etf_mvp.data.registry import available_providers
from etf_mvp.interpretation import load_local_env
from etf_mvp.intraday import (
    IntradayPaths,
    fetch_intraday_symbols,
    run_intraday_pipeline,
    train_intraday_symbols,
)
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
    for command, help_text in (
        ("intraday-fetch", "Fetch multi-year 5-minute QQQ and SOXX bars"),
        ("intraday-run", "Fetch and train separate 5-minute QQQ and SOXX models"),
    ):
        intraday_parser = subparsers.add_parser(command, help=help_text)
        intraday_parser.add_argument("--start", default="2018-01-01")
        intraday_parser.add_argument("--end")
        intraday_parser.add_argument("--feed", choices=("sip", "iex"), default="iex")
    subparsers.add_parser(
        "intraday-train", help="Train separate QQQ and SOXX models from stored 5-minute bars"
    )
    return parser


def main() -> None:
    load_local_env()
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    config = _apply_overrides(load_config(args.config), args)

    if args.command.startswith("intraday-"):
        project_root = Path(config.data.prices_path).parent.parent.parent
        paths = IntradayPaths(
            data_dir=project_root / "data" / "intraday",
            artifacts_dir=config.output.artifacts_dir / "intraday",
        )
        try:
            if args.command == "intraday-fetch":
                frames = fetch_intraday_symbols(
                    paths,
                    start=args.start,
                    end=args.end,
                    feed=args.feed,
                )
                for symbol, frame in frames.items():
                    print(f"{symbol:<4} {len(frame):>9,} five-minute bars")
                return
            if args.command == "intraday-train":
                results = train_intraday_symbols(paths, config.model.random_seed)
            else:
                results = run_intraday_pipeline(
                    paths,
                    start=args.start,
                    end=args.end,
                    feed=args.feed,
                    random_seed=config.model.random_seed,
                )
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            raise SystemExit(str(error)) from None
        for result in results:
            print(
                f"{result.symbol}: {result.model_name}, "
                f"forecast={result.forecast_date.date()}, "
                f"direction_accuracy={result.metrics['direction_accuracy']:.1%}"
            )
        print(f"Artifacts: {paths.artifacts_dir}")
        return

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
