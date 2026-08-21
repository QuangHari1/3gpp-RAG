"""Command-line entry point for one paper-compatible RAG question."""

from __future__ import annotations

import argparse
import json

from .service import PaperRagService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the one-round Telco-oRAG pipeline.")
    parser.add_argument("question", nargs="+", help="Telecom question to retrieve against Release 18.")
    parser.add_argument("--no-answer", action="store_true", help="Stop after retrieval; do not call the answer model.")
    parser.add_argument("--json", action="store_true", help="Emit the complete trace as JSON.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = PaperRagService().run(" ".join(args.question), include_answer=not args.no_answer)
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if result.answer:
        print(result.answer)
    print("\nRouter selected:", ", ".join(result.router_selected_series))
    if result.empty_selected_series:
        print("Empty selected:", ", ".join(result.empty_selected_series))
    print("Searched:", ", ".join(result.searched_series))
    for hit in result.retrievals:
        print(f"\n[{hit.series}] score={hit.score:.4f} {hit.metadata.get('document_name', '')}")
        print(hit.text[:500])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
