"""Run PersonaMem experiments using mem0 utilities."""

from __future__ import annotations

import argparse
import os

from src.memzero.personamemadd import PersonaMemConfig, PersonaMemUploader
from src.memzero.personasearch import PersonaMemSearch, run_personamem_search
from src.utils import METHODS

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_QUESTIONS_PATH = os.path.join(_SCRIPT_DIR, "dataset", "PersonaMem", "questions_32k.csv")
_DEFAULT_CONTEXTS_PATH = os.path.join(_SCRIPT_DIR, "dataset", "PersonaMem", "shared_contexts_32k.jsonl")
_DEFAULT_OUTPUT_FOLDER = os.path.join(_SCRIPT_DIR, "results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PersonaMem experiments using mem0")
    parser.add_argument(
        "--method",
        choices=METHODS,
        default="add",
        help="Pipeline to execute (mem0 add/search).",
    )
    parser.add_argument(
        "--questions_file",
        type=str,
        default=_DEFAULT_QUESTIONS_PATH,
        help="Path to the PersonaMem questions CSV file.",
    )
    parser.add_argument(
        "--contexts_file",
        type=str,
        default=_DEFAULT_CONTEXTS_PATH,
        help="Path to the PersonaMem shared contexts JSON/JSONL file.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of chat messages to send to mem0 in a single request.",
    )
    parser.add_argument(
        "--include_system",
        action="store_true",
        help="Include system messages when uploading contexts.",
    )
    parser.add_argument(
        "--enable_graph",
        action="store_true",
        help="Enable graph mode when storing memories.",
    )
    parser.add_argument(
        "--max_contexts",
        type=int,
        default=None,
        help="Limit the number of shared contexts to upload (useful for smoke tests).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Parse the dataset without making mem0 API calls.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=_DEFAULT_OUTPUT_FOLDER,
        help="Directory where search results should be written.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=30,
        help="Number of top memories to retrieve for search.",
    )
    parser.add_argument(
        "--filter_memories",
        action="store_true",
        help="Apply memory filtering during search.",
    )
    parser.add_argument(
        "--search_is_graph",
        action="store_true",
        help="Use graph-based search when running the search method.",
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="Limit the number of PersonaMem questions processed during search.",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default=None,
        help="Optional explicit path for search results JSON output.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PersonaMemConfig:
    return PersonaMemConfig(
        questions_path=args.questions_file,
        contexts_path=args.contexts_file,
        batch_size=args.batch_size,
        include_system=args.include_system,
        enable_graph=args.enable_graph,
        max_contexts=args.max_contexts,
        dry_run=args.dry_run,
        method=args.method,
        output_folder=args.output_folder,
        top_k=args.top_k,
        filter_memories=args.filter_memories,
        search_is_graph=args.search_is_graph,
        max_questions=args.max_questions,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)

    if args.method == "search":
        if args.results_path:
            run_personamem_search(config, output_path=args.results_path)
        else:
            searcher = PersonaMemSearch(config)
            searcher.run()
    else:
        uploader = PersonaMemUploader(config)
        uploader.run()


if __name__ == "__main__":
    main()

