"""PersonaMem search helpers for mem0 experiments."""

from __future__ import annotations

import csv
import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

from mem0 import MemoryClient

from .personamemadd import PersonaMemConfig


class PersonaMemSearch:
    """Execute PersonaMem search runs against a mem0 project."""

    def __init__(self, config: PersonaMemConfig, output_path: Optional[str] = None) -> None:
        load_dotenv()

        self.config = config
        self.mem0_client = MemoryClient(
            api_key=os.getenv("MEM0_API_KEY"),
            org_id=os.getenv("MEM0_ORGANIZATION_ID"),
            project_id=os.getenv("MEM0_PERSONAMEM_PROJECT_ID", os.getenv("MEM0_PROJECT_ID")),
        )

        output_dir = config.output_folder or "results"
        os.makedirs(output_dir, exist_ok=True)
        if output_path is None:
            output_path = os.path.join(
                output_dir,
                f"personamem_mem0_results_top_{config.top_k}_filter_{config.filter_memories}_graph_{config.search_is_graph}.json",
            )
        self.output_path = output_path
        self.results: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Process all PersonaMem questions and persist mem0 search results."""

        questions = self._load_questions()
        for idx, row in enumerate(tqdm(questions, desc="Running PersonaMem search")):
            if self.config.max_questions is not None and idx >= self.config.max_questions:
                break

            persona_id = (row.get("persona_id") or "").strip()
            if not persona_id:
                continue

            question_text = self._extract_question(row)
            if not question_text:
                continue

            memories, graph_memories, latency_s, error = self._search_memories(persona_id, question_text)

            record = {
                "question_id": row.get("question_id"),
                "shared_context_id": row.get("shared_context_id"),
                "question": question_text,
                "correct_answer": row.get("correct_answer") or row.get("answer"),
                "topic": row.get("topic"),
                "question_type": row.get("question_type"),
                "search_latency_seconds": latency_s,
                "memories": memories,
                "graph_memories": graph_memories,
            }
            if error is not None:
                record["error"] = error

            self.results[persona_id].append(record)

            # Persist progress periodically to avoid data loss on long runs.
            if (idx + 1) % 50 == 0:
                self._flush_results()

        self._flush_results()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_questions(self) -> List[Dict[str, str]]:
        with open(self.config.questions_path, "r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None:
                raise ValueError("PersonaMem questions CSV is missing headers.")
            return [row for row in reader]

    def _extract_question(self, row: Dict[str, str]) -> str:
        for key in ("user_question_or_message", "question", "prompt"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _search_memories(self, persona_id: str, query: str) -> Tuple[List[Dict[str, object]], Optional[List[Dict[str, object]]], float, Optional[str]]:
        start_time = time.time()
        try:
            if self.config.search_is_graph:
                raw = self.mem0_client.search(
                    query,
                    user_id=persona_id,
                    top_k=self.config.top_k,
                    filter_memories=self.config.filter_memories,
                    enable_graph=True,
                    output_format="v1.1",
                )
                memories = [
                    {
                        "memory": item.get("memory"),
                        "score": item.get("score"),
                        "metadata": item.get("metadata"),
                    }
                    for item in raw.get("results", [])
                ]
                graph_memories = [
                    {
                        "source": relation.get("source"),
                        "relationship": relation.get("relationship"),
                        "target": relation.get("target"),
                    }
                    for relation in raw.get("relations", [])
                ]
            else:
                raw = self.mem0_client.search(
                    query,
                    user_id=persona_id,
                    top_k=self.config.top_k,
                    filter_memories=self.config.filter_memories,
                )
                memories = [
                    {
                        "memory": item.get("memory"),
                        "score": item.get("score"),
                        "metadata": item.get("metadata"),
                    }
                    for item in raw
                ]
                graph_memories = None
            latency_s = time.time() - start_time
            return memories, graph_memories, latency_s, None
        except Exception as exc:  # pragma: no cover - defensive logging path
            latency_s = time.time() - start_time
            return [], None, latency_s, str(exc)

    def _flush_results(self) -> None:
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4, ensure_ascii=False)


def run_personamem_search(config: PersonaMemConfig, output_path: Optional[str] = None) -> None:
    """Convenience wrapper mirroring the Locomo experiment API."""

    searcher = PersonaMemSearch(config, output_path=output_path)
    searcher.run()


__all__ = ["PersonaMemSearch", "run_personamem_search"]

