from __future__ import annotations

import copy
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import tiktoken
except ImportError:  # pragma: no cover - exercised in dependency-light environments.
    tiktoken = None

from src.utils import get_dialogue_utterance


TokenCounter = Callable[[str, str], int]


class OversizedDialogueTurnError(ValueError):
    """Raised when one complete dialogue turn cannot fit in a source chunk."""

    def __init__(self, dia_id: str, estimated_tokens: int, limit: int):
        self.dia_id = dia_id
        self.estimated_tokens = estimated_tokens
        self.limit = limit
        super().__init__(
            f"dialogue turn {dia_id or '<missing>'} requires "
            f"{estimated_tokens} tokens, exceeding chunk limit {limit}"
        )


@dataclass(frozen=True)
class AblationChunk:
    chunk_id: str
    conversation: Dict[str, Any]
    dia_ids: List[str]
    start_order: int
    end_order: int
    estimated_tokens: int


def estimate_tokens(text: str, model_name: str) -> int:
    """Estimate tokens with the configured model encoding and a stable fallback."""

    if tiktoken is None:
        return max(1, math.ceil(len(str(text or "").encode("utf-8")) / 4))
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except (KeyError, ValueError):
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(str(text or "")))


def _session_keys(conversation: Dict[str, Any]) -> List[str]:
    keys = [
        str(key)
        for key, value in conversation.items()
        if re.fullmatch(r"session_\d+", str(key)) and isinstance(value, list)
    ]
    return sorted(keys, key=lambda key: int(key.split("_")[1]))


def _base_metadata(conversation: Dict[str, Any]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for key, value in conversation.items():
        key_text = str(key)
        if re.fullmatch(r"session_\d+", key_text):
            continue
        if re.fullmatch(r"session_\d+_(?:date_time|time)", key_text):
            continue
        metadata[key] = copy.deepcopy(value)
    return metadata


def _build_chunk_conversation(
    original: Dict[str, Any],
    turns: List[Tuple[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    conversation = _base_metadata(original)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    ordered_sessions: List[str] = []

    for session_key, turn in turns:
        if session_key not in grouped:
            grouped[session_key] = []
            ordered_sessions.append(session_key)
        grouped[session_key].append(copy.deepcopy(turn))

    for session_key in ordered_sessions:
        for suffix in ("date_time", "time"):
            metadata_key = f"{session_key}_{suffix}"
            if metadata_key in original:
                conversation[metadata_key] = copy.deepcopy(original[metadata_key])
        conversation[session_key] = grouped[session_key]

    return conversation


def chunk_conversation(
    conversation: Dict[str, Any],
    max_source_tokens: int,
    model_name: str,
    token_counter: Optional[TokenCounter] = None,
) -> List[AblationChunk]:
    """Split trace conversation into ordered, non-overlapping, complete-turn chunks."""

    if not isinstance(conversation, dict):
        return []
    if int(max_source_tokens) <= 0:
        raise ValueError("max_source_tokens must be positive")

    counter = token_counter or estimate_tokens
    chunks: List[AblationChunk] = []
    current_turns: List[Tuple[str, Dict[str, Any]]] = []
    current_ids: List[str] = []
    current_tokens = 0
    current_start = 0
    global_order = 0

    def flush() -> None:
        nonlocal current_turns, current_ids, current_tokens, current_start
        if not current_turns:
            return
        chunk_number = len(chunks) + 1
        chunks.append(
            AblationChunk(
                chunk_id=f"C{chunk_number:04d}",
                conversation=_build_chunk_conversation(conversation, current_turns),
                dia_ids=list(current_ids),
                start_order=current_start,
                end_order=current_start + len(current_turns) - 1,
                estimated_tokens=current_tokens,
            )
        )
        current_turns = []
        current_ids = []
        current_tokens = 0

    for session_key in _session_keys(conversation):
        for turn in conversation.get(session_key, []):
            if not isinstance(turn, dict):
                continue
            dia_id = str(turn.get("dia_id", "")).strip()
            utterance = str(get_dialogue_utterance(turn) or "").strip()
            if not dia_id or not utterance:
                continue

            turn_tokens = max(1, int(counter(utterance, model_name)))
            if turn_tokens > max_source_tokens:
                raise OversizedDialogueTurnError(
                    dia_id=dia_id,
                    estimated_tokens=turn_tokens,
                    limit=max_source_tokens,
                )

            if current_turns and current_tokens + turn_tokens > max_source_tokens:
                flush()

            if not current_turns:
                current_start = global_order
            current_turns.append((session_key, turn))
            current_ids.append(dia_id)
            current_tokens += turn_tokens
            global_order += 1

    flush()
    return chunks


def bisect_ablation_chunk(
    chunk: AblationChunk,
    model_name: str,
    token_counter: Optional[TokenCounter] = None,
) -> List[AblationChunk]:
    """Split one chunk near the middle by complete-turn count."""

    turns: List[Tuple[str, Dict[str, Any]]] = []
    for session_key in _session_keys(chunk.conversation):
        for turn in chunk.conversation.get(session_key, []):
            if isinstance(turn, dict):
                turns.append((session_key, turn))
    if len(turns) < 2:
        return []

    counter = token_counter or estimate_tokens
    midpoint = len(turns) // 2
    child_turn_groups = [turns[:midpoint], turns[midpoint:]]
    children: List[AblationChunk] = []
    order_offset = 0
    for child_index, child_turns in enumerate(
        child_turn_groups,
        start=1,
    ):
        dia_ids = [
            str(turn.get("dia_id", "")).strip()
            for _, turn in child_turns
        ]
        estimated_tokens = sum(
            max(
                1,
                int(
                    counter(
                        str(get_dialogue_utterance(turn) or ""),
                        model_name,
                    )
                ),
            )
            for _, turn in child_turns
        )
        children.append(
            AblationChunk(
                chunk_id=f"{chunk.chunk_id}.S{child_index:02d}",
                conversation=_build_chunk_conversation(
                    chunk.conversation,
                    child_turns,
                ),
                dia_ids=dia_ids,
                start_order=chunk.start_order + order_offset,
                end_order=(
                    chunk.start_order
                    + order_offset
                    + len(child_turns)
                    - 1
                ),
                estimated_tokens=estimated_tokens,
            )
        )
        order_offset += len(child_turns)
    return children


def _normalized_terms(text: str) -> Counter[str]:
    return Counter(re.findall(r"[a-z0-9\u4e00-\u9fff]+", str(text or "").casefold()))


def _chunk_search_text(chunk: AblationChunk) -> str:
    values: List[str] = []
    for session_key in _session_keys(chunk.conversation):
        for turn in chunk.conversation.get(session_key, []):
            if not isinstance(turn, dict):
                continue
            values.append(str(get_dialogue_utterance(turn) or ""))
    return "\n".join(values)


def rank_chunks(
    chunks: List[AblationChunk],
    query: str,
    limit: int,
) -> List[AblationChunk]:
    """Rank chunks by deterministic lexical overlap and preserve order on ties."""

    if limit <= 0:
        return []
    query_terms = _normalized_terms(query)
    scored = []
    for chunk in chunks:
        chunk_terms = _normalized_terms(_chunk_search_text(chunk))
        score = sum(
            min(query_count, chunk_terms.get(term, 0))
            for term, query_count in query_terms.items()
        )
        scored.append((score, chunk.start_order, chunk))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored[:limit]]
