"""Point d'entrée en ligne de commande : `chatbot-eval --config config.json`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import load_config
from .runner import run_evaluation, summarize


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Évalue un chatbot à partir d'un golden dataset.")
    parser.add_argument("-c", "--config", required=True, type=Path, help="Fichier de configuration JSON")
    parser.add_argument("-n", "--limit", type=int, default=None, help="Ne traiter que les N premiers cas")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Surcharge du chemin du rapport Excel")
    parser.add_argument("--concurrency", type=int, default=None, help="Surcharge du nb d'appels parallèles au chatbot")
    parser.add_argument("-v", "--verbose", action="store_true", help="Logs détaillés")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    try:
        config = load_config(args.config)
    except Exception as exc:
        logging.error("Configuration invalide : %s", exc)
        return 2
    if args.concurrency:
        config.chatbot.concurrency = args.concurrency

    results, path = run_evaluation(
        config, limit=args.limit, output_path=args.output, config_dir=args.config.resolve().parent
    )
    summary = summarize(results, config)
    print("\n=== Synthèse ===")
    print(summary.to_string(index=False) if not summary.empty else "(aucune métrique)")
    errors = int(results["error"].notna().sum())
    print(f"\n{len(results)} cas, {errors} appel(s) en erreur. Rapport : {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
