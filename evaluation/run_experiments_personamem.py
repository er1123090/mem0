"""Utilities for uploading the PersonaMem benchmark to mem0.

This script mirrors the behaviour of the Locomo experiment runner by
loading the PersonaMem benchmark files (questions and shared contexts) and
adding the corresponding user memories to a mem0 project.
"""
import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from dotenv import load_dotenv
from tqdm import tqdm

from mem0 import MemoryClient

from src.memzero.add import custom_instructions

# Keys that may contain nested conversation turns inside the PersonaMem JSONL file.
NESTED_TURN_KEYS = (
    "messages",
    "turns",
    "conversation",
    "conversations",
    "dialogue",
    "dialogues",
    "history",
    "utterances",
    "responses",
    "chat",
    "chats",
    "contents",
    "chunks",
    "steps",
    "entries",
)


@dataclass
class PersonaMemConfig:
    """Configuration for uploading PersonaMem data to mem0."""

    questions_path: str
    contexts_path: str
    batch_size: int = 4
    include_system: bool = False
    enable_graph: bool = False
    max_contexts: Optional[int] = None
    dry_run: bool = False


class PersonaMemUploader:
    """Uploads PersonaMem user conversations to mem0."""

    def __init__(self, config: PersonaMemConfig) -> None:
        load_dotenv()

        self.config = config
        self.mem0_client: Optional[MemoryClient] = None

        if not config.dry_run:
            self.mem0_client = MemoryClient(
                api_key=os.getenv("MEM0_API_KEY"),
                org_id=os.getenv("MEM0_ORGANIZATION_ID"),
                project_id=os.getenv("MEM0_PROJECT_ID"),
            )
            self.mem0_client.update_project(custom_instructions=custom_instructions)

    def run(self) -> None:
        """Run the upload process for the configured dataset."""

        context_to_persona = self._load_questions()
        context_store = self._load_contexts(context_to_persona.keys())

        processed_personas: set[str] = set()
        missing_contexts: List[str] = []

        iterator = list(context_to_persona.items())
        if self.config.max_contexts is not None:
            iterator = iterator[: self.config.max_contexts]

        for shared_context_id, persona_id in tqdm(iterator, desc="Uploading PersonaMem contexts"):
            context_data = context_store.get(shared_context_id)
            if context_data is None:
                missing_contexts.append(shared_context_id)
                continue

            if persona_id not in processed_personas:
                processed_personas.add(persona_id)
                if not self.config.dry_run and self.mem0_client is not None:
                    self.mem0_client.delete_all(user_id=persona_id)

            messages = self._context_to_messages(context_data)
            if not messages:
                continue

            metadata = {"shared_context_id": shared_context_id, "source": "PersonaMem"}
            self._add_messages(persona_id, messages, metadata)

        if missing_contexts:
            print(
                f"Warning: {len(missing_contexts)} shared contexts were referenced in the questions file "
                "but missing from the contexts JSONL."
            )

    def _load_questions(self) -> Dict[str, str]:
        """Load the PersonaMem questions file and return a context-to-persona map."""

        context_to_persona: Dict[str, str] = {}

        with open(self.config.questions_path, "r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            if "shared_context_id" not in reader.fieldnames or "persona_id" not in reader.fieldnames:
                missing = {"shared_context_id", "persona_id"} - set(reader.fieldnames or [])
                raise ValueError(
                    "The questions file must include the columns 'shared_context_id' and 'persona_id'. "
                    f"Missing columns: {', '.join(sorted(missing))}."
                )

            for row in reader:
                context_id = row.get("shared_context_id")
                persona_id = row.get("persona_id")

                if not context_id or not persona_id:
                    continue

                context_to_persona.setdefault(context_id, persona_id)

        if not context_to_persona:
            raise ValueError(
                "No mappings between shared_context_id and persona_id were found in the questions file."
            )

        return context_to_persona

    def _load_contexts(self, required_context_ids: Iterable[str]) -> Dict[str, Any]:
        """Load the PersonaMem shared contexts JSONL file."""

        context_store: Dict[str, Any] = {}
        required = set(required_context_ids)

        with open(self.config.contexts_path, "r", encoding="utf-8") as jsonl_file:
            for line in jsonl_file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                record = json.loads(line)
                context_id = self._extract_context_id(record)
                if context_id:
                    if context_id not in context_store:
                        context_store[context_id] = record
                    if required and required.issubset(context_store.keys()):
                        # Early exit once all required contexts are loaded.
                        break

        missing = required - context_store.keys()
        if missing:
            print(
                f"Warning: {len(missing)} contexts referenced in the questions file were not found in the JSONL file."
            )

        return context_store

    def _extract_context_id(self, record: Dict[str, Any]) -> Optional[str]:
        """Attempt to find a shared context identifier for a JSONL record."""

        for key in ("shared_context_id", "context_id", "id", "shared_context_key"):
            value = record.get(key)
            if isinstance(value, str) and value:
                return value
        return None

    def _context_to_messages(self, context_record: Dict[str, Any]) -> List[Dict[str, str]]:
        """Convert a PersonaMem context record into a list of chat messages."""

        raw_turns: List[Any] = []

        for key in NESTED_TURN_KEYS:
            if key in context_record and isinstance(context_record[key], list):
                raw_turns.extend(context_record[key])

        if not raw_turns:
            if "sessions" in context_record and isinstance(context_record["sessions"], dict):
                for session in context_record["sessions"].values():
                    raw_turns.extend(self._ensure_list(session))
            else:
                raw_turns = self._ensure_list(context_record)

        messages: List[Dict[str, str]] = []
        for turn in raw_turns:
            messages.extend(self._normalise_turn(turn))

        cleaned_messages = [msg for msg in messages if msg.get("content")]

        if not self.config.include_system:
            cleaned_messages = [msg for msg in cleaned_messages if msg.get("role") != "system"]

        return cleaned_messages

    def _ensure_list(self, value: Any) -> List[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            turns: List[Any] = []
            for key, nested in value.items():
                if key in NESTED_TURN_KEYS and isinstance(nested, list):
                    turns.extend(nested)
            if turns:
                return turns
        return [value]

    def _normalise_turn(self, turn: Any) -> List[Dict[str, str]]:
        if isinstance(turn, list):
            messages: List[Dict[str, str]] = []
            for item in turn:
                messages.extend(self._normalise_turn(item))
            return messages

        if isinstance(turn, dict):
            messages: List[Dict[str, str]] = []

            for key in NESTED_TURN_KEYS:
                if key in turn and isinstance(turn[key], list):
                    for nested in turn[key]:
                        messages.extend(self._normalise_turn(nested))

            role = self._normalise_role(turn.get("role") or turn.get("speaker") or turn.get("author") or turn.get("name"))
            content = self._stringify_content(
                turn.get("content")
                or turn.get("text")
                or turn.get("message")
                or turn.get("utterance")
                or turn.get("response")
                or turn.get("value")
            )

            if content:
                messages.append({"role": role, "content": content})

            return messages

        text = str(turn).strip()
        if not text:
            return []
        return [{"role": "user", "content": text}]

    def _stringify_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [self._stringify_content(item) for item in content]
            return "\n".join(part for part in parts if part)
        if isinstance(content, dict):
            for key in ("text", "content", "message", "value"):
                if key in content:
                    return self._stringify_content(content[key])
            return json.dumps(content, ensure_ascii=False)
        if content is None:
            return ""
        return str(content).strip()

    def _normalise_role(self, role: Optional[str]) -> str:
        if not role:
            return "user"
        role_lower = role.lower()
        if any(tag in role_lower for tag in ("assistant", "bot", "model", "ai")):
            return "assistant"
        if "system" in role_lower:
            return "system"
        return "user"

    def _add_messages(self, persona_id: str, messages: Sequence[Dict[str, str]], metadata: Dict[str, Any]) -> None:
        if self.config.dry_run or self.mem0_client is None:
            return

        for index in range(0, len(messages), self.config.batch_size):
            batch = list(messages[index : index + self.config.batch_size])
            if not batch:
                continue
            self.mem0_client.add(
                batch,
                user_id=persona_id,
                metadata=metadata,
                version="v2",
                enable_graph=self.config.enable_graph,
            )


def parse_args() -> PersonaMemConfig:
    parser = argparse.ArgumentParser(description="Upload PersonaMem conversations to mem0")
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
        "--max-contexts",
        type=int,
        default=None,
        help="Limit the number of shared contexts to upload (for testing).",
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
        batch_size=args.batch_size,
        include_system=args.include_system,
        enable_graph=args.enable_graph,
        max_contexts=args.max_contexts,
        dry_run=args.dry_run,
    )


def main() -> None:
    config = parse_args()
    uploader = PersonaMemUploader(config)
    uploader.run()


if __name__ == "__main__":
    main()
