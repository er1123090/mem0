"""Upload the PersonaMem benchmark conversations to mem0.

This script mirrors the behaviour of the existing Locomo uploader. It reads the
PersonaMem question mapping (CSV) together with the shared contexts (JSONL),
normalises every conversation into the format expected by mem0, and finally
stores the conversations for each persona.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from dotenv import load_dotenv
from tqdm import tqdm

from mem0 import MemoryClient

from src.memzero.add import custom_instructions
from src.memzero.search import MemorySearch

# Fields that commonly contain the conversation payload in PersonaMem contexts.
_TURN_KEYS = (
    "messages",
    "context",
    "turns",
    "conversation",
    "conversations",
)

# Possible keys that provide the role/speaker of a turn.
_ROLE_KEYS = ("role", "speaker", "author", "name")

# Possible keys that provide the textual content of a turn.
_CONTENT_KEYS = ("content", "text", "message", "utterance", "response", "value", "body")


def _stringify(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_stringify(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        for key in _CONTENT_KEYS:
            if key in value:
                nested = _stringify(value[key])
                if nested:
                    return nested
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value).strip()


@dataclass
class PersonaMemConfig:
    """Runtime configuration for the PersonaMem uploader."""

    questions_path: str
    contexts_path: str
    method: str = "add"
    batch_size: int = 4
    include_system: bool = False
    enable_graph: bool = False
    filter_memories: bool = False
    top_k: int = 30
    output_folder: str = "results"
    max_contexts: Optional[int] = None
    dry_run: bool = False


class PersonaMemUploader:
    """Convert PersonaMem data and upload it to mem0."""

    def __init__(self, config: PersonaMemConfig) -> None:
        load_dotenv()

        self.config = config
        self._client: Optional[MemoryClient] = None

        if not config.dry_run:
            self._client = MemoryClient(
                api_key=os.getenv("MEM0_API_KEY"),
                org_id=os.getenv("MEM0_ORGANIZATION_ID"),
                project_id=os.getenv("MEM0_PROJECT_ID"),
            )
            self._client.project.update(custom_instructions=custom_instructions)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Upload all conversations referenced in the PersonaMem questions file."""

        context_to_persona = self._load_questions()
        contexts = self._load_contexts(context_to_persona.keys())

        missing_contexts: List[str] = []
        processed_personas: set[str] = set()

        iterator = list(context_to_persona.items())
        if self.config.max_contexts is not None:
            iterator = iterator[: self.config.max_contexts]

        for shared_context_id, persona_id in tqdm(iterator, desc="Uploading PersonaMem contexts"):
            context_record = contexts.get(shared_context_id)
            if context_record is None:
                missing_contexts.append(shared_context_id)
                continue

            if persona_id not in processed_personas:
                processed_personas.add(persona_id)
                self._reset_persona(persona_id)

            messages = self._normalise_context(context_record)
            if not messages:
                continue

            metadata = {"shared_context_id": shared_context_id, "source": "PersonaMem"}
            self._add_messages(persona_id, messages, metadata)

        if missing_contexts:
            print(
                "Warning: %d shared contexts were referenced in the questions file but missing from the JSONL." % len(missing_contexts)
            )

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------
    def _load_questions(self) -> Dict[str, str]:
        """Create a mapping from shared_context_id to persona_id."""

        context_to_persona: Dict[str, str] = {}

        with open(self.config.questions_path, "r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            required_columns = {"shared_context_id", "persona_id"}
            if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
                missing = required_columns - set(reader.fieldnames or [])
                raise ValueError(
                    "The questions file must include the columns 'shared_context_id' and 'persona_id'. Missing columns: %s."
                    % ", ".join(sorted(missing))
                )

            for row in reader:
                context_id = (row.get("shared_context_id") or "").strip()
                persona_id = (row.get("persona_id") or "").strip()
                if not context_id or not persona_id:
                    continue
                context_to_persona.setdefault(context_id, persona_id)

        if not context_to_persona:
            raise ValueError("No shared context identifiers were found in the questions CSV.")

        return context_to_persona

    def _load_contexts(self, required_context_ids: Iterable[str]) -> Dict[str, Dict[str, object]]:
        """Load the shared contexts JSONL file into a dictionary keyed by context id."""

        contexts: Dict[str, Dict[str, object]] = {}
        required = set(required_context_ids)

        with open(self.config.contexts_path, "r", encoding="utf-8") as jsonl_file:
            for line in jsonl_file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                record = json.loads(line)
                context_id = self._extract_context_id(record)
                if not context_id:
                    continue

                if context_id not in contexts:
                    contexts[context_id] = record

                if required and required.issubset(contexts.keys()):
                    break

        missing = required - contexts.keys()
        if missing:
            print(
                "Warning: %d contexts referenced in the questions file were not found in the JSONL file." % len(missing)
            )

        return contexts

    @staticmethod
    def _extract_context_id(record: Dict[str, object]) -> Optional[str]:
        for key in ("shared_context_id", "context_id", "id"):
            value = record.get(key)
            if isinstance(value, (str, int, float)):
                text = str(value).strip()
                if text:
                    return text
        return None

    # ------------------------------------------------------------------
    # Conversation normalisation helpers
    # ------------------------------------------------------------------
    def _normalise_context(self, context_record: Dict[str, object]) -> List[Dict[str, str]]:
        raw_turns: List[object] = []

        for key in _TURN_KEYS:
            value = context_record.get(key)
            if isinstance(value, list):
                raw_turns.extend(value)

        if not raw_turns:
            # Some records wrap the turns inside nested objects (e.g. {"context": {"messages": [...]}}).
            for key in _TURN_KEYS:
                value = context_record.get(key)
                if isinstance(value, dict):
                    for nested_key in _TURN_KEYS:
                        nested_value = value.get(nested_key)
                        if isinstance(nested_value, list):
                            raw_turns.extend(nested_value)

        messages: List[Dict[str, str]] = []
        for turn in raw_turns:
            messages.extend(self._expand_turn(turn))

        if not self.config.include_system:
            messages = [msg for msg in messages if msg["role"] != "system"]

        return messages

    def _expand_turn(self, turn: object) -> List[Dict[str, str]]:
        if isinstance(turn, dict):
            nested_messages: List[Dict[str, str]] = []
            for key in _TURN_KEYS:
                nested = turn.get(key)
                if isinstance(nested, list):
                    for item in nested:
                        nested_messages.extend(self._expand_turn(item))

            role = self._resolve_role(turn)
            content = self._resolve_content(turn)
            direct_messages: List[Dict[str, str]] = []
            if content:
                direct_messages.append({"role": role, "content": content})

            combined: List[Dict[str, str]] = []
            combined.extend(direct_messages)
            combined.extend(nested_messages)
            return combined

        if isinstance(turn, list):
            # Some datasets encode alternating speaker/content pairs in lists. We join them into a sentence.
            parts = [_stringify(item) for item in turn]
            content = " ".join(part for part in parts if part)
            if not content:
                return []
            return [{"role": "user", "content": content}]

        text = _stringify(turn)
        if not text:
            return []
        return [{"role": "user", "content": text}]

    def _resolve_role(self, turn: Dict[str, object]) -> str:
        for key in _ROLE_KEYS:
            value = turn.get(key)
            if isinstance(value, str) and value:
                role_lower = value.lower()
                if "system" in role_lower:
                    return "system"
                if any(alias in role_lower for alias in ("assistant", "bot", "model", "ai")):
                    return "assistant"
                return "user"
        return "user"

    def _resolve_content(self, turn: Dict[str, object]) -> str:
        for key in _CONTENT_KEYS:
            value = turn.get(key)
            content = _stringify(value)
            if content:
                return content
        return ""

    # ------------------------------------------------------------------
    # mem0 interaction helpers
    # ------------------------------------------------------------------
    def _reset_persona(self, persona_id: str) -> None:
        if self.config.dry_run or self._client is None:
            return
        self._client.delete_all(user_id=persona_id)

    def _add_messages(self, persona_id: str, messages: Sequence[Dict[str, str]], metadata: Dict[str, object]) -> None:
        if self.config.dry_run or self._client is None:
            return

        for start in range(0, len(messages), self.config.batch_size):
            batch = list(messages[start : start + self.config.batch_size])
            if not batch:
                continue
            self._client.add(
                batch,
                user_id=persona_id,
                metadata=metadata,
                version="v2",
                enable_graph=self.config.enable_graph,
            )


class PersonaMemSearchEvaluator:
    """Run mem0 search over PersonaMem questions mimicking the Locomo workflow."""

    def __init__(self, config: PersonaMemConfig) -> None:
        load_dotenv()

        self.config = config
        self._output_path = self._build_output_path()
        self._results: Dict[int, List[Dict[str, object]]] = defaultdict(list)

        if config.dry_run:
            self._searcher: Optional[MemorySearch] = None
        else:
            directory = os.path.dirname(self._output_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            self._searcher = MemorySearch(
                output_path=self._output_path,
                top_k=config.top_k,
                filter_memories=config.filter_memories,
                is_graph=config.enable_graph,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Evaluate PersonaMem questions using mem0 search."""

        persona_questions = self._group_questions_by_persona()

        iterator = list(persona_questions.items())
        if self.config.max_contexts is not None:
            iterator = iterator[: self.config.max_contexts]

        for idx, (persona_id, questions) in enumerate(tqdm(iterator, desc="Searching PersonaMem questions")):
            for question in questions:
                result = self._process_question(idx, persona_id, question)
                if result is None:
                    continue
                self._results[idx].append(result)
                self._persist_results()

    # ------------------------------------------------------------------
    # Question loading helpers
    # ------------------------------------------------------------------
    def _group_questions_by_persona(self) -> OrderedDict[str, List[Dict[str, str]]]:
        grouped: OrderedDict[str, List[Dict[str, str]]] = OrderedDict()

        with open(self.config.questions_path, "r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None:
                raise ValueError("The questions CSV must include headers.")

            for row in reader:
                persona_id = (row.get("persona_id") or "").strip()
                if not persona_id:
                    continue
                grouped.setdefault(persona_id, []).append(row)

        if not grouped:
            raise ValueError("No PersonaMem questions were found in the CSV file.")

        return grouped

    # ------------------------------------------------------------------
    # Search helpers
    # ------------------------------------------------------------------
    def _process_question(
        self,
        _conversation_index: int,
        persona_id: str,
        question_row: Dict[str, str],
    ) -> Optional[Dict[str, object]]:
        question = (question_row.get("user_question_or_message") or "").strip()
        answer = (question_row.get("correct_answer") or "").strip()
        category = (question_row.get("question_type") or "").strip()
        evidence = self._parse_evidence(question_row.get("all_options"))
        if not question:
            return None

        if self.config.dry_run or self._searcher is None:
            return {
                "question": question,
                "answer": answer,
                "category": category,
                "evidence": evidence,
                "response": None,
                "adversarial_answer": (question_row.get("adversarial_answer") or "").strip(),
                "speaker_1_memories": [],
                "speaker_2_memories": [],
                "num_speaker_1_memories": 0,
                "num_speaker_2_memories": 0,
                "speaker_1_memory_time": 0.0,
                "speaker_2_memory_time": 0.0,
                "speaker_1_graph_memories": None,
                "speaker_2_graph_memories": None,
                "response_time": 0.0,
            }

        speaker_1_user_id = persona_id
        speaker_2_user_id = f"{persona_id}_assistant"

        (
            response,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            response_time,
        ) = self._searcher.answer_question(
            speaker_1_user_id,
            speaker_2_user_id,
            question,
            answer,
            category,
        )

        return {
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": evidence,
            "response": response,
            "adversarial_answer": (question_row.get("adversarial_answer") or "").strip(),
            "speaker_1_memories": speaker_1_memories,
            "speaker_2_memories": speaker_2_memories,
            "num_speaker_1_memories": len(speaker_1_memories),
            "num_speaker_2_memories": len(speaker_2_memories),
            "speaker_1_memory_time": speaker_1_memory_time,
            "speaker_2_memory_time": speaker_2_memory_time,
            "speaker_1_graph_memories": speaker_1_graph_memories,
            "speaker_2_graph_memories": speaker_2_graph_memories,
            "response_time": response_time,
        }

    def _parse_evidence(self, raw_value: Optional[str]) -> List[str]:
        if not raw_value:
            return []

        raw_value = raw_value.strip()
        if not raw_value:
            return []

        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            separators = ["||", "|", ";", "\n"]
            for sep in separators:
                if sep in raw_value:
                    return [part.strip() for part in raw_value.split(sep) if part.strip()]
            return [raw_value]
        else:
            if isinstance(parsed, list):
                return [_stringify(item) for item in parsed if _stringify(item)]
            return [_stringify(parsed)]

    def _persist_results(self) -> None:
        if self.config.dry_run or self._searcher is None:
            return

        serialisable = {str(idx): results for idx, results in self._results.items()}
        with open(self._output_path, "w", encoding="utf-8") as handle:
            json.dump(serialisable, handle, indent=4)

    def _build_output_path(self) -> str:
        filename = "personamem_mem0_results_top_{top}_filter_{flt}_graph_{graph}.json".format(
            top=self.config.top_k,
            flt=int(self.config.filter_memories),
            graph=int(self.config.enable_graph),
        )
        return os.path.join(self.config.output_folder, filename)


def parse_args() -> PersonaMemConfig:
    parser = argparse.ArgumentParser(description="Run PersonaMem workflows with mem0")
    parser.add_argument(
        "--questions-file",
        type=str,
        default="dataset/PersonaMem/questions_32k.csv",
        help="Path to the PersonaMem questions CSV file.",
    )
    parser.add_argument(
        "--contexts-file",
        type=str,
        default="dataset/PersonaMem/shared_contexts_32k.jsonl",
        help="Path to the PersonaMem shared contexts JSONL file.",
    )
    parser.add_argument(
        "--method",
        choices=["add", "search"],
        default="add",
        help="Choose between uploading contexts (add) or running search evaluation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of chat messages to send to mem0 in a single request.",
    )
    parser.add_argument(
        "--include-system",
        action="store_true",
        help="Include system messages when uploading contexts.",
    )
    parser.add_argument(
        "--enable-graph",
        action="store_true",
        help="Enable graph mode when storing memories.",
    )
    parser.add_argument(
        "--filter-memories",
        action="store_true",
        help="Apply the mem0 filters when retrieving memories in search mode.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Number of top memories to retrieve when running search.",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="results",
        help="Destination directory for search results.",
    )
    parser.add_argument(
        "--max-contexts",
        type=int,
        default=None,
        help="Limit the number of shared contexts to upload (useful for smoke tests).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse the dataset without making mem0 API calls.",
    )

    args = parser.parse_args()

    return PersonaMemConfig(
        questions_path=args.questions_file,
        contexts_path=args.contexts_file,
        method=args.method,
        batch_size=args.batch_size,
        include_system=args.include_system,
        enable_graph=args.enable_graph,
        filter_memories=args.filter_memories,
        top_k=args.top_k,
        output_folder=args.output_folder,
        max_contexts=args.max_contexts,
        dry_run=args.dry_run,
    )


def main() -> None:
    config = parse_args()
    if config.method == "search":
        evaluator = PersonaMemSearchEvaluator(config)
        evaluator.run()
    else:
        uploader = PersonaMemUploader(config)
        uploader.run()


if __name__ == "__main__":
    main()
