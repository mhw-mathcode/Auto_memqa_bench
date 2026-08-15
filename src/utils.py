import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union


def ensure_parent_dir(path: Union[str, Path]) -> None:
    """Create the parent directory for a file path when it has one."""
    parent = Path(path).parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)


def load_json_file(path: Union[str, Path]) -> Any:
    """Load a JSON file with UTF-8 encoding."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_file(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Write a JSON file with UTF-8 encoding and stable non-ASCII output."""
    ensure_parent_dir(path)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=indent)


def count_qa_items(data: Any) -> int:
    """Count QA items in a normalized or raw dataset payload."""
    return sum(len(record.get("qa", [])) for record in normalize_dataset_records(data))


def normalize_dialogue_text(text: Any) -> str:
    """Normalize dialogue text for strict evidence matching with stable whitespace."""
    normalized = str(text or "")
    normalized = (
        normalized.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2026", "...")
    )
    return re.sub(r"\s+", " ", normalized).strip()


def get_dialogue_utterance(chat: Dict[str, Any]) -> str:
    """Return the canonical text field for a dialogue turn."""
    if not isinstance(chat, dict):
        return ""
    for key in ("text", "utterance", "content"):
        value = chat.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def build_speaker_map(conversation: Dict[str, Any]) -> Dict[str, str]:
    """Build a best-effort speaker alias map from a conversation object."""
    speaker_map: Dict[str, str] = {}
    if not isinstance(conversation, dict):
        return speaker_map

    raw_speakers = conversation.get("speakers")
    if isinstance(raw_speakers, list):
        for idx, speaker_name in enumerate(raw_speakers, start=1):
            if isinstance(speaker_name, str) and speaker_name.strip():
                speaker_map[f"speaker_{idx}"] = speaker_name.strip()

    for key, value in conversation.items():
        if re.fullmatch(r"speaker_\d+", str(key)) and isinstance(value, str) and value.strip():
            speaker_map[str(key)] = value.strip()

    for alias in ("speaker_a", "speaker_b"):
        alias_value = conversation.get(alias)
        if isinstance(alias_value, str) and alias_value.strip():
            speaker_map[alias] = alias_value.strip()

    return speaker_map


def resolve_speaker_name(raw_speaker: Any, speaker_map: Dict[str, str]) -> Optional[str]:
    """Resolve speaker aliases like speaker_1 to display names when possible."""
    if raw_speaker in (None, ""):
        return None
    if isinstance(raw_speaker, str):
        return speaker_map.get(raw_speaker, raw_speaker)
    if isinstance(raw_speaker, int):
        return speaker_map.get(f"speaker_{raw_speaker}", str(raw_speaker))
    return str(raw_speaker)


def build_dialogue_index(conversation: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Index dialogue turns by dia_id for evidence validation."""
    index: Dict[str, Dict[str, Any]] = {}
    if not isinstance(conversation, dict):
        return index

    speaker_map = build_speaker_map(conversation)
    session_keys = [
        str(key)
        for key in conversation.keys()
        if re.fullmatch(r"session_\d+", str(key))
    ]
    session_keys.sort(key=lambda key: int(re.match(r"session_(\d+)", key).group(1)))

    for session_key in session_keys:
        chats = conversation.get(session_key, [])
        if not isinstance(chats, list):
            continue
        for turn_index, chat in enumerate(chats):
            if not isinstance(chat, dict):
                continue
            dia_id = str(chat.get("dia_id") or "").strip()
            if not dia_id:
                continue
            text = get_dialogue_utterance(chat)
            index[dia_id.casefold()] = {
                "dia_id": dia_id,
                "speaker": resolve_speaker_name(chat.get("speaker"), speaker_map),
                "utterance": text,
                "session": session_key,
                "turn_index": turn_index,
            }

    return index


def _expand_dia_id_range(dia_id: str) -> List[str]:
    """Expand simple dia_id ranges such as D1:43-44 into [D1:43, D1:44]."""
    value = str(dia_id or "").strip()
    match = re.fullmatch(r"([A-Za-z]+\d+:)(\d+)-(\d+)", value)
    if not match:
        return []
    prefix, start_text, end_text = match.groups()
    start = int(start_text)
    end = int(end_text)
    if end < start or end - start > 20:
        return []
    return [f"{prefix}{idx}" for idx in range(start, end + 1)]


def _has_shared_source_excerpt(evidence_text: str, source_text: str, min_chars: int = 12) -> bool:
    """Check whether evidence and source share a non-trivial exact character span."""
    evidence_compact = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", normalize_dialogue_text(evidence_text).casefold())
    source_compact = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", normalize_dialogue_text(source_text).casefold())
    if not evidence_compact or not source_compact:
        return False
    if evidence_compact in source_compact or source_compact in evidence_compact:
        return True

    # Avoid importing difflib at module load for callers that do not validate ranges.
    import difflib

    match = difflib.SequenceMatcher(None, evidence_compact, source_compact).find_longest_match(
        0,
        len(evidence_compact),
        0,
        len(source_compact),
    )
    return match.size >= min_chars


def _classify_source_evidence_match(evidence_text: str, source_text: str) -> str:
    """
    Classify how an evidence utterance matches its source turn.

    Valid evidence can be either the complete source utterance or an exact,
    contiguous source excerpt. Ordered ellipsis evidence is also accepted when
    every non-trivial segment appears in source order. Paraphrases or invented
    text are rejected.
    """
    evidence_norm = normalize_dialogue_text(evidence_text)
    source_norm = normalize_dialogue_text(source_text)
    evidence_match_text = evidence_norm.casefold()
    source_match_text = source_norm.casefold()
    if not evidence_norm:
        return "empty"
    if evidence_match_text == source_match_text:
        return "exact_turn"
    if evidence_match_text in source_match_text:
        return "contiguous_excerpt"

    if "..." not in evidence_norm and "…" not in evidence_norm:
        return "mismatch"

    parts = [
        part.strip()
        for part in re.split(r"(?:\.\.\.|…)+", evidence_norm)
        if part.strip()
    ]
    if not parts:
        return "mismatch"

    cursor = 0
    matched_parts = 0
    for part in parts:
        if len(re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", part.casefold())) < 8:
            continue
        found_at = source_match_text.find(part.casefold(), cursor)
        if found_at < 0:
            return "mismatch"
        cursor = found_at + len(part)
        matched_parts += 1
    return "ordered_ellipsis" if matched_parts else "mismatch"


def _normalize_speaker_label(value: Any) -> str:
    """Normalize speaker labels for comparing embedded role names."""
    text = normalize_dialogue_text(value).strip()
    text = text.strip("\"'`[]()（）【】")
    text = re.sub(r"[:：]+$", "", text).strip()
    return text.casefold()


def _is_technical_speaker(value: Any) -> bool:
    """Return whether a speaker is a transport role rather than an in-story role."""
    return _normalize_speaker_label(value) in {"assistant", "user", "system", "tool"}


def _line_speaker_label(line: str) -> str:
    """Extract a short dialogue label from a source-text line when one is present."""
    clean = str(line or "").strip()
    if not clean:
        return ""

    colon_match = re.match(r"^([^:：]{1,40})[:：]\s*", clean)
    if colon_match:
        return _normalize_speaker_label(colon_match.group(1))

    compact = re.sub(r"\s+", "", clean)
    if len(compact) > 40:
        return ""
    if re.search(r"[。？！!?，,；;\"“”‘’]", clean):
        return ""
    return _normalize_speaker_label(clean)


def _source_turn_embeds_speaker(
    source_text: str,
    evidence_text: str,
    evidence_speaker: Any,
) -> bool:
    """
    Check whether a technical chat turn embeds a character-labeled dialogue span.

    Some datasets store long role-play turns as speaker=assistant/user while the
    text itself contains lines such as `Mint\n"..."`. In that case the model may
    correctly cite the in-story speaker. Accept it only when the cited utterance
    is located inside that labeled span.
    """
    speaker_label = _normalize_speaker_label(evidence_speaker)
    evidence_norm = normalize_dialogue_text(evidence_text).casefold()
    if not speaker_label or not evidence_norm:
        return False

    raw_lines = str(source_text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [line.strip() for line in raw_lines]
    label_positions = [
        (idx, label)
        for idx, line in enumerate(lines)
        for label in [_line_speaker_label(line)]
        if label
    ]

    for position_idx, (line_idx, label) in enumerate(label_positions):
        if label != speaker_label:
            continue

        current_line = lines[line_idx]
        colon_split = re.split(r"[:：]\s*", current_line, maxsplit=1)
        same_line_text = colon_split[1] if len(colon_split) == 2 else ""
        next_label_idx = (
            label_positions[position_idx + 1][0]
            if position_idx + 1 < len(label_positions)
            else len(lines)
        )
        segment_text = "\n".join([same_line_text] + lines[line_idx + 1:next_label_idx])
        if evidence_norm in normalize_dialogue_text(segment_text).casefold():
            return True

    if speaker_label in {"narrator", "旁白", "叙述者"}:
        for position_idx, (line_idx, _) in enumerate(label_positions):
            next_label_idx = (
                label_positions[position_idx + 1][0]
                if position_idx + 1 < len(label_positions)
                else len(lines)
            )
            labelled_segment = "\n".join(lines[line_idx + 1:next_label_idx])
            if evidence_norm in normalize_dialogue_text(labelled_segment).casefold():
                return False
        return evidence_norm in normalize_dialogue_text(source_text).casefold()

    return False


def _find_unique_dialogue_turn_by_utterance(
    dialogue_index: Dict[str, Dict[str, Any]],
    utterance: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Repair a missing dia_id only when an evidence excerpt has one clear source turn."""
    matches: List[Tuple[Dict[str, Any], str]] = []
    for source_turn in dialogue_index.values():
        match_type = _classify_source_evidence_match(
            utterance,
            str(source_turn.get("utterance") or ""),
        )
        if match_type in {"exact_turn", "contiguous_excerpt", "ordered_ellipsis"}:
            matches.append((source_turn, match_type))

    if len(matches) != 1:
        return None, ""
    return matches[0]


def normalize_evidence_dialogues(value: Any) -> List[Dict[str, Any]]:
    """Normalize evidence_dialogues to a list of dicts."""
    if value in (None, ""):
        return []
    if isinstance(value, dict):
        return [deepcopy(value)]
    if isinstance(value, list):
        return [deepcopy(item) for item in value if isinstance(item, dict)]
    if isinstance(value, str) and value.strip():
        return [{"id": "E1", "speaker": None, "utterance": value.strip(), "dia_id": "N/A"}]
    return []


def align_evidence_dialogues(
    evidence_dialogues: Any,
    conversation: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Keep only evidence items whose dia_id exists and whose utterance is an exact
    contiguous excerpt of the same original dialogue turn after stable whitespace
    normalization. This allows annotators to cite a necessary subspan while still
    preventing invented/paraphrased evidence.
    """
    raw_evidence = normalize_evidence_dialogues(evidence_dialogues)
    dialogue_index = build_dialogue_index(conversation)
    aligned: List[Dict[str, Any]] = []
    invalid: List[Dict[str, Any]] = []
    match_type_counts: Dict[str, int] = {}

    for idx, evidence in enumerate(raw_evidence, start=1):
        evidence_id = str(evidence.get("id") or f"E{idx}").strip() or f"E{idx}"
        dia_id = str(evidence.get("dia_id") or "").strip()
        utterance = str(evidence.get("utterance") or "").strip()
        source_turn: Optional[Dict[str, Any]] = None
        source_repair = None
        match_type = ""

        if not dia_id or dia_id.upper() == "N/A":
            source_turn, match_type = _find_unique_dialogue_turn_by_utterance(
                dialogue_index,
                utterance,
            )
            if source_turn:
                original_dia_id = dia_id or "missing"
                dia_id = str(source_turn.get("dia_id") or "").strip()
                source_repair = {
                    "type": "unique_utterance_match",
                    "original_dia_id": original_dia_id,
                    "repaired_dia_id": dia_id,
                }
            else:
                invalid.append(
                    {
                        "id": evidence_id,
                        "dia_id": dia_id or "missing",
                        "reason": "missing_dia_id",
                    }
                )
                continue

        if source_turn is None:
            source_turn = dialogue_index.get(dia_id.casefold())
        if not source_turn:
            expanded_ids = _expand_dia_id_range(dia_id)
            expanded_turns = [
                dialogue_index.get(expanded_id.casefold())
                for expanded_id in expanded_ids
            ]
            if expanded_ids and all(expanded_turns):
                repairable_turns = [
                    turn for turn in expanded_turns
                    if isinstance(turn, dict)
                    and _has_shared_source_excerpt(utterance, str(turn.get("utterance") or ""))
                ]
                if repairable_turns:
                    for turn in repairable_turns:
                        aligned.append(
                            {
                                "id": f"E{len(aligned) + 1}",
                                "speaker": evidence.get("speaker") or turn.get("speaker"),
                                "utterance": str(turn.get("utterance") or ""),
                                "dia_id": turn.get("dia_id"),
                                "source_repair": {
                                    "type": "expanded_dia_id_range",
                                    "original_dia_id": dia_id,
                                },
                            }
                        )
                    continue

            invalid.append(
                {
                    "id": evidence_id,
                    "dia_id": dia_id,
                    "reason": "dia_id_not_found",
                    "utterance": utterance,
                }
            )
            continue

        source_utterance = str(source_turn.get("utterance") or "")
        source_speaker = source_turn.get("speaker")
        if not match_type:
            match_type = _classify_source_evidence_match(utterance, source_utterance)
        if match_type not in {"exact_turn", "contiguous_excerpt", "ordered_ellipsis"}:
            invalid.append(
                {
                    "id": evidence_id,
                    "dia_id": dia_id,
                    "reason": "utterance_mismatch",
                    "utterance": utterance,
                    "source_utterance": source_utterance,
                    "match_type": match_type,
                }
            )
            continue
        match_type_counts[match_type] = match_type_counts.get(match_type, 0) + 1

        evidence_speaker = evidence.get("speaker")
        character_speaker = evidence.get("character_speaker")
        speaker_normalization = None
        if evidence_speaker not in (None, ""):
            evidence_speaker_norm = normalize_dialogue_text(evidence_speaker)
            source_speaker_norm = normalize_dialogue_text(source_speaker)
            if evidence_speaker_norm != source_speaker_norm:
                can_treat_as_embedded_speaker = (
                    _is_technical_speaker(source_speaker)
                    and _source_turn_embeds_speaker(source_utterance, utterance, evidence_speaker)
                )
                if not can_treat_as_embedded_speaker:
                    invalid.append(
                        {
                            "id": evidence_id,
                            "dia_id": dia_id,
                            "reason": "speaker_mismatch",
                            "speaker": evidence_speaker,
                            "source_speaker": source_speaker,
                        }
                    )
                    continue
                character_speaker = character_speaker or evidence_speaker
                speaker_normalization = {
                    "type": "embedded_character_speaker",
                    "original_speaker": evidence_speaker,
                    "source_speaker": source_speaker,
                }

        aligned_item = {
            "id": f"E{len(aligned) + 1}",
            "speaker": source_speaker,
            "utterance": utterance,
            "dia_id": source_turn.get("dia_id"),
        }
        if character_speaker not in (None, ""):
            aligned_item["character_speaker"] = character_speaker
        if speaker_normalization:
            aligned_item["speaker_normalization"] = speaker_normalization
        if source_repair:
            aligned_item["source_repair"] = source_repair
        aligned.append(aligned_item)

    report = {
        "result": "pass" if not invalid and aligned else "fail",
        "total": len(raw_evidence),
        "valid": len(aligned),
        "invalid": len(invalid),
        "invalid_items": invalid[:20],
        "match_type_counts": match_type_counts,
    }
    if not raw_evidence:
        report["result"] = "fail"
        report["reason"] = "empty_evidence_dialogues"
    elif aligned and invalid:
        report["result"] = "partial"

    return aligned, report


def normalize_reasoning_steps(value: Any, evidence_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Normalize reasoning_steps to a list of structured dicts."""
    evidence_ids = evidence_ids or []
    if isinstance(value, list):
        normalized: List[Dict[str, Any]] = []
        for idx, item in enumerate(value, start=1):
            if isinstance(item, dict):
                step = deepcopy(item)
                step["step"] = int(step.get("step") or idx)
                step["inference"] = str(step.get("inference") or step.get("reasoning") or "").strip()
                based_on = step.get("based_on", evidence_ids)
                step["based_on"] = based_on if isinstance(based_on, list) else [based_on]
                normalized.append(step)
            elif item not in (None, ""):
                normalized.append(
                    {
                        "step": idx,
                        "inference": str(item).strip(),
                        "based_on": evidence_ids,
                    }
                )
        return normalized
    if isinstance(value, dict):
        return normalize_reasoning_steps([value], evidence_ids=evidence_ids)
    if isinstance(value, str) and value.strip():
        return [{"step": 1, "inference": value.strip(), "based_on": evidence_ids}]
    return []


def _coerce_numeric_key(value: Any) -> Any:
    """
    Helper to convert stringified numeric keys to integers so ordering is stable
    between list-based and dict-based QA payloads.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _is_dataset_record(value: Any) -> bool:
    """
    Detect a single conversation record instead of a mapping of records.
    """
    return isinstance(value, dict) and any(
        key in value for key in ("qa", "conversation", "speaker_a", "speaker_b")
    )


def _iter_dataset_records(data: Any) -> Iterator[Dict[str, Any]]:
    """
    Iterate over dataset records while supporting both collection-style payloads
    and files whose top-level value is a single conversation record.
    """
    if isinstance(data, dict):
        if _is_dataset_record(data):
            yield data
            return

        for key in sorted(data.keys(), key=_coerce_numeric_key):
            item = data[key]
            if isinstance(item, dict):
                yield item
        return

    for item in data or []:
        if isinstance(item, dict):
            yield item


def normalize_qa_section(qa_section: Any) -> List[Dict[str, Any]]:
    """
    Normalize the QA section of a dataset entry so downstream code can assume it is a list.
    Supports both the legacy list-of-dicts format and the new dict-of-dicts format.
    """
    if isinstance(qa_section, dict):
        ordered_keys = sorted(qa_section.keys(), key=_coerce_numeric_key)
        return [deepcopy(qa_section[key]) for key in ordered_keys]
    if isinstance(qa_section, list):
        return [deepcopy(item) for item in qa_section]
    return []


def normalize_dataset_records(data: Union[Sequence[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize an entire dataset so each entry contains a list-based QA section.
    """
    normalized_records: List[Dict[str, Any]] = []
    for item in _iter_dataset_records(data):
        item_copy = deepcopy(item)
        item_copy["qa"] = normalize_qa_section(item_copy.get("qa", []))
        normalized_records.append(item_copy)
    return normalized_records


def normalize_dataset_record(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a single dataset record while avoiding the cost of copying the
    entire collection into memory. Only the QA section is deep-copied so callers
    can safely mutate it.
    """
    if not isinstance(item, dict):
        return {}
    item_copy: Dict[str, Any] = dict(item)
    item_copy["qa"] = normalize_qa_section(item_copy.get("qa", []))
    return item_copy


def _stream_json_array_fast(path: Path, chunk_size: int) -> Iterator[Any]:
    """
    Efficiently stream JSON objects from an on-disk array without loading the
    entire payload into memory.
    """
    decoder = json.JSONDecoder()
    buffer = ""
    inside_array = False

    with path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            buffer += chunk

            while True:
                stripped = buffer.lstrip()
                if not stripped:
                    buffer = ""
                    break

                if not inside_array:
                    lead = stripped[0]
                    if lead in "\ufeff":
                        buffer = stripped[1:]
                        continue
                    if lead == "[":
                        inside_array = True
                        buffer = stripped[1:]
                        continue
                    raise ValueError("Top-level JSON value is not an array.")

                head = stripped[0]
                if head == "]":
                    return
                if head == ",":
                    buffer = stripped[1:]
                    continue

                try:
                    obj, offset = decoder.raw_decode(stripped)
                except json.JSONDecodeError:
                    # Need more data from disk.
                    buffer = stripped
                    break

                yield obj
                buffer = stripped[offset:]

        # Handle any trailing content after the final read.
        stripped = buffer.lstrip()
        if not stripped:
            return
        if not inside_array:
            raise ValueError("Top-level JSON value is not an array.")
        if stripped[0] == "]":
            return
        if stripped[0] == ",":
            stripped = stripped[1:].lstrip()
        if stripped and stripped[0] != "]":
            raise ValueError("Unexpected trailing content in JSON array.")


def stream_json_array(path: Union[str, Path], chunk_size: int = 65_536) -> Iterator[Any]:
    """
    Stream JSON records from a file whose top-level value is an array. Falls back
    to loading the full payload when streaming is not possible (e.g. dict input).
    """
    json_path = Path(path)
    try:
        yield from _stream_json_array_fast(json_path, chunk_size)
        return
    except ValueError:
        pass

    data = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if _is_dataset_record(data):
            yield data
        else:
            for key in sorted(data.keys(), key=_coerce_numeric_key):
                item = data[key]
                if isinstance(item, dict):
                    yield item
    elif isinstance(data, list):
        for item in data:
            yield item


def stream_normalized_dataset(path: Union[str, Path], chunk_size: int = 65_536) -> Iterator[Dict[str, Any]]:
    """
    Stream and normalize dataset records from disk. Each iteration yields a
    single normalized item so downstream callers can process records lazily.
    """
    for raw_record in stream_json_array(path, chunk_size=chunk_size):
        normalized = normalize_dataset_record(raw_record)
        if normalized:
            yield normalized


def compute_dataset_stats(path: Union[str, Path], chunk_size: int = 65_536) -> Dict[str, Any]:
    """
    Collect aggregate statistics for a dataset without retaining every record in
    memory. Returns counts for conversations, sessions, dialog turns, and QA
    items, as well as the QA counts per conversation for downstream bookkeeping.
    """
    totals = {
        "total_conversations": 0,
        "total_sessions": 0,
        "total_dialogues": 0,
        "total_questions": 0,
        "qa_per_conversation": [],
    }

    for record in stream_normalized_dataset(path, chunk_size=chunk_size):
        totals["total_conversations"] += 1

        conversation = record.get("conversation") or {}
        session_keys = [
            key for key in conversation.keys() if key.startswith("session_") and not key.endswith("_date_time")
        ]
        totals["total_sessions"] += len(session_keys)

        dialogue_count = 0
        for key in session_keys:
            chats = conversation.get(key, [])
            if isinstance(chats, list):
                dialogue_count += len(chats)
        totals["total_dialogues"] += dialogue_count

        qa_count = len(record.get("qa", []))
        totals["total_questions"] += qa_count
        totals["qa_per_conversation"].append(qa_count)

    return totals
