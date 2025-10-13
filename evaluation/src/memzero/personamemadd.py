"""Utilities for uploading PersonaMem conversations to mem0.

This module mirrors the behaviour of the ``MemoryADD`` implementation that
targets the Locomo benchmark but adapts the logic for the PersonaMem dataset.
The implementation is largely based on the previous
``run_experiments_personamem`` script which provided a rich uploader for the
benchmark.  Moving the code into a dedicated module makes it possible to reuse
the functionality from different entry points (e.g., the experiment runners)
and keeps parity with the Locomo layout where ``add``/``search`` live inside
``src/memzero``.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from dotenv import load_dotenv
from tqdm import tqdm

from mem0 import MemoryClient

from .add import custom_instructions

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
    """Runtime configuration for PersonaMem experiments."""

    questions_path: str
    contexts_path: Optional[str] = None
    batch_size: int = 4
    include_system: bool = False
    enable_graph: bool = False
    max_contexts: Optional[int] = None
    dry_run: bool = False
    method: str = "add"
    output_folder: str = "results"
    top_k: int = 30
    filter_memories: bool = False
    search_is_graph: bool = False
    max_questions: Optional[int] = None


class PersonaMemUploader:
    """Convert PersonaMem data and upload it to mem0."""

    def __init__(self, config: PersonaMemConfig) -> None:
        load_dotenv()

        if config.contexts_path is None:
            raise ValueError("'contexts_path' must be provided for the PersonaMem uploader.")

        self.config = config
        self._client: Optional[MemoryClient] = None

        if not config.dry_run:
            self._client = MemoryClient(
                api_key=os.getenv("MEM0_API_KEY"),
                org_id=os.getenv("MEM0_ORGANIZATION_ID"),
                project_id=os.getenv("MEM0_PERSONAMEM_PROJECT_ID", os.getenv("MEM0_PROJECT_ID")),
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

            metadata = {
                "shared_context_id": shared_context_id,
                "source": "PersonaMem",
            }
            self._add_messages(persona_id, messages, metadata)

        if missing_contexts:
            print(
                "Warning: %d shared contexts were referenced in the questions file but missing from the JSONL."
                % len(missing_contexts)
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
        """Load the shared contexts JSON/JSONL file into a dictionary keyed by context id."""

        contexts: Dict[str, Dict[str, object]] = {}
        required = set(required_context_ids)

        def _fallback_extract(record: object) -> tuple[Optional[str], Optional[object]]:
            """Return (context_id, payload) for common variants, else (None, None)."""

            if not isinstance(record, dict):
                return None, None

            # Primary: explicit id fields
            for k in ("shared_context_id", "context_id", "id"):
                v = record.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip(), record

            # Single-key wrapper where the key itself is the id
            if len(record) == 1:
                k = next(iter(record))
                if isinstance(k, str) and k.strip():
                    return k.strip(), record[k]

            return None, None

        with open(self.config.contexts_path, "r", encoding="utf-8") as jsonl_file:
            for raw_line in jsonl_file:
                # Handle BOM + whitespace + comments
                line = raw_line.lstrip("\ufeff").strip()
                if not line or line.startswith("#"):
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    print("Warning: skipping malformed JSONL line.")
                    continue

                # Try existing extractor first
                context_id = None
                payload: Optional[object] = None
                try:
                    context_id = self._extract_context_id(record)  # type: ignore[arg-type]
                    if context_id:
                        payload = record
                except Exception:
                    # Do nothing; we'll fall back
                    pass

                if not context_id:
                    context_id, payload = _fallback_extract(record)

                if not context_id:
                    continue

                # Ensure payload is dict-like; if it's a list of turns, wrap it so downstream expects dict
                if context_id not in contexts:
                    if isinstance(payload, dict):
                        contexts[context_id] = payload
                    else:
                        # payload could be a list of messages/turns
                        contexts[context_id] = {"messages": payload}

                # Early out when we've loaded all required
                if required and required.issubset(contexts.keys()):
                    break

        missing = required - contexts.keys()
        if missing:
            print(
                "Warning: %d contexts referenced in the questions file were not found in the JSON/JSONL file."
                % len(missing)
            )

        return contexts

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
            parts = [self._stringify(item) for item in turn]
            content = " ".join(part for part in parts if part)
            if not content:
                return []
            return [{"role": "user", "content": content}]

        text = self._stringify(turn)
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
                metadata={**metadata, "batch_index": start // self.config.batch_size},
                version="v2",
                enable_graph=self.config.enable_graph,
            )


__all__ = ["PersonaMemConfig", "PersonaMemUploader"]

