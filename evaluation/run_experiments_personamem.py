"""Upload the PersonaMem benchmark conversations to mem0 or run search evaluations.

This script mirrors the behaviour of the existing Locomo uploader. It reads the
PersonaMem question mapping (CSV) together with the shared contexts (JSONL),
normalises every conversation into the format expected by mem0, and finally
stores the conversations for each persona. When executed with
``--method search`` it mirrors the search workflow provided by
``run_experiments`` for convenience.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

from dotenv import load_dotenv
from tqdm import tqdm

from mem0 import MemoryClient

from src.memzero.add import custom_instructions
from src.memzero.search import MemorySearch
from src.utils import METHODS

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_QUESTIONS_PATH = os.path.join(_SCRIPT_DIR, "dataset", "PersonaMem", "questions_32k.csv")
_DEFAULT_CONTEXTS_PATH = os.path.join(_SCRIPT_DIR, "dataset", "PersonaMem", "shared_contexts_32k.jsonl")
_DEFAULT_SEARCH_DATA_FILE = os.path.join(_SCRIPT_DIR, "dataset", "Locomo", "locomo10.json")
_DEFAULT_OUTPUT_FOLDER = os.path.join(_SCRIPT_DIR, "results")

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


@dataclass
class PersonaMemConfig:
    """Runtime configuration for the PersonaMem uploader."""

    questions_path: str
    contexts_path: str
    batch_size: int = 4
    include_system: bool = False
    enable_graph: bool = False
    max_contexts: Optional[int] = None
    dry_run: bool = False
    method: str = "add"
    output_folder: str = _DEFAULT_OUTPUT_FOLDER
    search_data_file: str = _DEFAULT_SEARCH_DATA_FILE
    top_k: int = 30
    filter_memories: bool = False
    search_is_graph: bool = False


class PersonaMemUploader:
    """Convert PersonaMem data and upload it to mem0."""

    def __init__(self, config: PersonaMemConfig) -> None:
        load_dotenv()

        self.config = config
        self._client: Optional[MemoryClient] = None
        self._global_speaker_registry: Dict[str, str] = {}

        if not config.dry_run:
            self._client = MemoryClient(
                api_key=os.getenv("MEM0_API_KEY"),
                org_id=os.getenv("MEM0_ORGANIZATION_ID"),
                project_id=os.getenv("MEM0_PERSONAMEM_PROJECT_ID"),
            )
            self._client.update_project(custom_instructions=custom_instructions)

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
            question = context_record.get("question")
            if isinstance(question, str) and question.strip():
                metadata["question"] = question.strip()
            self._add_messages(persona_id, messages, metadata)

        if missing_contexts:
            print(
                "Warning: %d shared contexts were referenced in the questions file but missing from the JSONL." % len(missing_contexts)
            )
            print("Missing context IDs: %s" % ", ".join(missing_contexts))

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

        for record in self._iterate_context_records(self.config.contexts_path):
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
                "***Warning: %d contexts referenced in the questions file were not found in the JSONL file." % len(missing)
            )

        print(f"Loaded {len(contexts)} shared contexts from {self.config.contexts_path}")

        return contexts

    def _iterate_context_records(self, path: str) -> Iterator[Dict[str, object]]:
        """Yield context records from JSON, JSONL, or lightly structured files."""

        buffer = ""
        lines: List[str] = []
        any_record = False
        with open(path, "r", encoding="utf-8") as json_file:
            for raw_line in json_file:
                lines.append(raw_line)
                stripped = raw_line.strip()
                if not stripped or stripped.startswith("#"):
                    continue

                buffer += raw_line

                try:
                    parsed = json.loads(buffer)
                except JSONDecodeError:
                    # The current buffer does not yet represent a full JSON object.
                    continue

                any_record = True
                yield from self._normalise_record_container(parsed)
                buffer = ""

        if buffer.strip():
            try:
                parsed = json.loads(buffer)
            except JSONDecodeError as exc:
                if any_record:
                    raise ValueError(f"Unable to parse contexts file '{path}'.") from exc
                yield from self._parse_plaintext_context_content("".join(lines))
                return
            yield from self._normalise_record_container(parsed)
            return

        if not any_record:
            yield from self._parse_plaintext_context_content("".join(lines))

    @staticmethod
    def _normalise_record_container(parsed: object) -> Iterator[Dict[str, object]]:
        if isinstance(parsed, dict):
            for key in ("data", "contexts", "records", "items"):
                value = parsed.get(key)
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            yield item
                    return
            yield parsed
            return

        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    yield item

    def _parse_plaintext_context_content(self, content: str) -> Iterator[Dict[str, object]]:
        """Fallback parser for simple key-value context files."""

        records: List[Dict[str, object]] = []
        current_record: Dict[str, object] = {}
        current_context: List[Dict[str, object]] = []
        current_entry: Optional[Dict[str, object]] = None
        in_context = False

        def flush_entry() -> None:
            nonlocal current_entry
            if current_entry:
                current_context.append(current_entry)
                current_entry = None

        def flush_record() -> None:
            nonlocal current_record, current_context, in_context
            flush_entry()
            if current_context:
                current_record["context"] = list(current_context)
            if current_record:
                records.append(current_record)
            current_record = {}
            current_context = []
            in_context = False

        for raw_line in content.splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            if stripped.startswith("---"):
                flush_record()
                continue

            if in_context and stripped.startswith("-"):
                flush_entry()
                current_entry = {}
                stripped = stripped[1:].strip()
                if ":" in stripped:
                    key, value = stripped.split(":", 1)
                    current_entry[key.strip()] = value.strip()
                continue

            if ":" in stripped:
                key, value = stripped.split(":", 1)
                key = key.strip()
                value = value.strip()

                if in_context:
                    if current_entry is None:
                        current_entry = {}
                    current_entry[key] = value
                else:
                    if key.lower() == "context":
                        in_context = True
                        if value:
                            current_record["context"] = value
                        continue
                    current_record[key] = value
                continue

            if in_context:
                if current_entry is None:
                    current_entry = {"content": stripped}
                else:
                    existing = current_entry.get("content")
                    if isinstance(existing, str) and existing:
                        current_entry["content"] = f"{existing} {stripped}"
                    else:
                        current_entry["content"] = stripped
            else:
                current_record.setdefault("context", [])
                context_value = current_record["context"]
                if isinstance(context_value, list):
                    context_value.append({"content": stripped})

        flush_record()

        for record in records:
            yield record

    @staticmethod
    def _extract_context_id(record: Dict[str, object]) -> Optional[str]:
        for key in ("shared_context_id", "context_id", "id"):
            value = record.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    # ------------------------------------------------------------------
    # Conversation normalisation helpers
    # ------------------------------------------------------------------
    def _normalise_context(self, context_record: Dict[str, object]) -> List[Dict[str, object]]:
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

        speaker_registry: Dict[str, str] = {}
        messages: List[Dict[str, object]] = []
        for turn in raw_turns:
            messages.extend(self._expand_turn(turn, speaker_registry))

        if not self.config.include_system:
            messages = [msg for msg in messages if msg["role"] != "system"]

        return messages

    def _expand_turn(
        self,
        turn: object,
        speaker_registry: Dict[str, str],
        parent_speaker: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        if isinstance(turn, dict):
            speaker_name = self._resolve_speaker_name(turn) or parent_speaker
            nested_messages: List[Dict[str, object]] = []
            for key in _TURN_KEYS:
                nested = turn.get(key)
                if isinstance(nested, list):
                    for item in nested:
                        nested_messages.extend(self._expand_turn(item, speaker_registry, speaker_name))

            role = self._resolve_role(turn)
            content = self._resolve_content(turn)
            direct_messages: List[Dict[str, object]] = []
            if content:
                message = {"role": role, "content": content}
                self._apply_speaker_metadata(message, speaker_name, speaker_registry)
                direct_messages.append(message)

            combined: List[Dict[str, object]] = []
            combined.extend(direct_messages)
            combined.extend(nested_messages)
            return combined

        if isinstance(turn, list):
            # Some datasets encode alternating speaker/content pairs in lists. We join them into a sentence.
            parts = [self._stringify(item) for item in turn]
            content = " ".join(part for part in parts if part)
            if not content:
                return []
            message = {"role": "user", "content": content}
            self._apply_speaker_metadata(message, parent_speaker, speaker_registry)
            return [message]

        text = self._stringify(turn)
        if not text:
            return []
        message = {"role": "user", "content": text}
        self._apply_speaker_metadata(message, parent_speaker, speaker_registry)
        return [message]

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

    def _resolve_speaker_name(self, turn: Dict[str, object]) -> Optional[str]:
        for key in ("name", "speaker", "author", "role"):
            value = turn.get(key)
            if isinstance(value, str):
                cleaned = value.strip()
                if cleaned and not self._looks_like_role(cleaned):
                    return cleaned
        return None

    def _apply_speaker_metadata(
        self, message: Dict[str, object], speaker_name: Optional[str], speaker_registry: Dict[str, str]
    ) -> None:
        if not speaker_name:
            return

        speaker_id = self._assign_speaker_id(speaker_name, speaker_registry)
        if not speaker_id:
            return

        message["speaker_id"] = speaker_id
        metadata = message.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        metadata.setdefault("speaker_name", speaker_name.strip())
        message["metadata"] = metadata

    def _assign_speaker_id(self, speaker_name: str, speaker_registry: Dict[str, str]) -> Optional[str]:
        key = self._normalise_speaker_key(speaker_name)
        if not key:
            return None

        if key not in speaker_registry:
            speaker_registry[key] = self._generate_speaker_id(speaker_name, speaker_registry.values())
        return speaker_registry[key]

    @staticmethod
    def _normalise_speaker_key(speaker_name: str) -> str:
        cleaned = re.sub(r"\s+", " ", speaker_name.casefold()).strip()
        return cleaned

    @staticmethod
    def _generate_speaker_id(speaker_name: str, existing_ids: Iterable[str]) -> str:
        slug = re.sub(r"[^0-9a-zA-Z]+", "_", speaker_name.strip().lower()).strip("_")
        if not slug:
            slug = "speaker"
        if slug[0].isdigit():
            slug = f"speaker_{slug}"

        candidate = slug
        counter = 2
        existing = set(existing_ids)
        while candidate in existing:
            candidate = f"{slug}_{counter}"
            counter += 1
        return candidate

    @staticmethod
    def _looks_like_role(value: str) -> bool:
        lowered = value.casefold()
        return any(alias in lowered for alias in ("system", "assistant", "bot", "model", "ai", "user"))

    def _resolve_content(self, turn: Dict[str, object]) -> str:
        for key in _CONTENT_KEYS:
            value = turn.get(key)
            content = self._stringify(value)
            if content:
                return content
        return ""

    def _stringify(self, value: object) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts = [self._stringify(item) for item in value]
            return "\n".join(part for part in parts if part)
        if isinstance(value, dict):
            for key in _CONTENT_KEYS:
                if key in value:
                    nested = self._stringify(value[key])
                    if nested:
                        return nested
            return json.dumps(value, ensure_ascii=False)
        if value is None:
            return ""
        return str(value).strip()

    # ------------------------------------------------------------------
    # mem0 interaction helpers
    # ------------------------------------------------------------------
    def _reset_persona(self, persona_id: str) -> None:
        if self.config.dry_run or self._client is None:
            return
        self._client.delete_all(user_id=persona_id)

    def _add_messages(self, persona_id: str, messages: Sequence[Dict[str, object]], metadata: Dict[str, object]) -> None:
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


def run_personamem_search(config: PersonaMemConfig) -> None:
    """Execute the Mem0 search pipeline mirroring ``run_experiments``."""

    if not config.search_data_file:
        raise ValueError("A dataset path must be provided when method is 'search'.")

    os.makedirs(config.output_folder, exist_ok=True)

    output_file_path = os.path.join(
        config.output_folder,
        f"mem0_results_top_{config.top_k}_filter_{config.filter_memories}_graph_{config.search_is_graph}.json",
    )

    memory_searcher = MemorySearch(
        output_file_path,
        config.top_k,
        config.filter_memories,
        config.search_is_graph,
    )
    memory_searcher.process_data_file(config.search_data_file)


def parse_args() -> PersonaMemConfig:
    parser = argparse.ArgumentParser(description="Upload PersonaMem conversations to mem0")
    parser.add_argument("--method", choices=METHODS, default="add", help="Method to execute.")
    parser.add_argument(
        "--questions-file",
        type=str,
        default=_DEFAULT_QUESTIONS_PATH,
        help="Path to the PersonaMem questions CSV file.",
    )
    parser.add_argument(
        "--contexts-file",
        type=str,
        default=_DEFAULT_CONTEXTS_PATH,
        help="Path to the PersonaMem shared contexts JSONL file.",
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
        default=False,
        help="Enable graph mode when storing memories.",
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
    parser.add_argument(
        "--output-folder",
        type=str,
        default=_DEFAULT_OUTPUT_FOLDER,
        help="Output directory for search results when using the 'search' method.",
    )
    parser.add_argument(
        "--search-data-file",
        type=str,
        default=_DEFAULT_SEARCH_DATA_FILE,
        help="Dataset JSON file to use for search evaluation.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=30,
        help="Number of top memories to retrieve during search.",
    )
    parser.add_argument(
        "--filter_memories",
        action="store_true",
        dest="filter_memories",
        default=False,
        help="Whether to filter memories during search.",
    )
    parser.add_argument(
        "--is_graph",
        action="store_true",
        dest="search_is_graph",
        default=False,
        help="Use graph-based search when running the 'search' method.",
    )

    args = parser.parse_args()

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
        search_data_file=args.search_data_file,
        top_k=args.top_k,
        filter_memories=args.filter_memories,
        search_is_graph=args.search_is_graph,
    )


def main() -> None:
    config = parse_args()
    if config.method == "search":
        run_personamem_search(config)
    else:
        uploader = PersonaMemUploader(config)
        uploader.run()


if __name__ == "__main__":
    main()
