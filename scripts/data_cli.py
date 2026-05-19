from __future__ import annotations

import argparse

from scripts.common import deep_update, load_config, print_config_summary


DEFAULT_CONFIG = {
    "branch": "local",
    "batch_size": 8,
    "num_workers": 0,
    "n_batches": 20,
    "audit": True,
    "skip_regions": [],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and audit project dataloaders.")
    parser.add_argument("--config", type=str, default="configs/data/local_data.yaml")
    parser.add_argument("--branch", choices=["local", "global", "both"], default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--n-batches", type=int, default=None)
    parser.add_argument("--no-audit", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = deep_update(DEFAULT_CONFIG, load_config(args.config))

    if args.branch is not None:
        config["branch"] = args.branch
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.n_batches is not None:
        config["n_batches"] = args.n_batches
    if args.no_audit:
        config["audit"] = False

    if args.print_config:
        print_config_summary(config)

    if config["audit"]:
        from data.analyze_loaders import analyze_global, analyze_local

        if config["branch"] in {"local", "both"}:
            analyze_local(
                batch_size=int(config["batch_size"]),
                num_workers=int(config["num_workers"]),
                n_batches=int(config["n_batches"]),
                skip_regions=config.get("skip_regions"),
            )

        if config["branch"] in {"global", "both"}:
            analyze_global(
                batch_size=min(int(config["batch_size"]), 4),
                num_workers=int(config["num_workers"]),
                n_batches=int(config["n_batches"]),
                skip_regions=config.get("skip_regions"),
            )
    else:
        from data.create_data import build_global_dataloaders, build_local_dataloaders

        if config["branch"] in {"local", "both"}:
            build_local_dataloaders(
                batch_size=int(config["batch_size"]),
                num_workers=int(config["num_workers"]),
                pin_memory=False,
                skip_regions=config.get("skip_regions"),
            )

        if config["branch"] in {"global", "both"}:
            build_global_dataloaders(
                batch_size=min(int(config["batch_size"]), 4),
                num_workers=int(config["num_workers"]),
                pin_memory=False,
                skip_regions=config.get("skip_regions"),
            )


if __name__ == "__main__":
    main()
