from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Callable, Optional

from src.benchmark_qa_schema import (
    PUBLIC_QA_FIELDS,
    normalize_public_qa,
    validate_public_qa,
)
from src.benchmark_question_rewrites import (
    find_construction_issues,
    make_rewrite_key,
    rewrite_publication_item,
)
from src.question_formatting import extract_core_question_text


@dataclass(frozen=True)
class PolicyAction:
    action: str
    reason: str = ""
    text: Optional[str] = None

    @classmethod
    def keep(cls) -> "PolicyAction":
        return cls("keep")

    @classmethod
    def delete(cls, reason: str) -> "PolicyAction":
        return cls("delete", reason=reason)

    @classmethod
    def replace_text(cls, text: str, reason: str) -> "PolicyAction":
        return cls("replace_text", reason=reason, text=text)


TurnPolicy = Callable[[int, dict], PolicyAction]


def _dialogue_number(turn: dict) -> int:
    dia_id = str(turn.get("dia_id") or "")
    match = re.fullmatch(r"D\d+:(\d+)", dia_id)
    if not match:
        raise ValueError(f"Invalid dia_id: {dia_id!r}")
    return int(match.group(1))


def _replace_literal(text: str, old: str) -> str:
    if old not in text:
        raise ValueError(f"Expected choice marker not found: {old!r}")
    return re.sub(r"\s+", " ", text.replace(old, " ")).strip()


def build_policy(title_key: str) -> TurnPolicy:
    def policy(session: int, turn: dict) -> PolicyAction:
        number = _dialogue_number(turn)
        text = str(turn.get("text") or "")

        if title_key == "arknights":
            if re.search(r"(?:^|\n)Option_\d+:", text):
                return PolicyAction.delete("interactive option turn")
            return PolicyAction.keep()

        if title_key == "highway-blossoms":
            if session == 1 and number >= 4054:
                return PolicyAction.delete("post-story blooper/easter-egg montage")
            return PolicyAction.keep()

        if title_key == "fata-morgana-requiem":
            if session == 5:
                return PolicyAction.delete("backstage bonus scenario")
            if session == 2 and number == 7706:
                return PolicyAction.delete("extreme repeated-scream noise")
            return PolicyAction.keep()

        if title_key == "heart-of-the-woods":
            if session != 1:
                return PolicyAction.keep()
            if 3943 <= number <= 4031:
                return PolicyAction.delete("duplicate romance branch")
            if number == 4458:
                marker = "*Madison, look! There it is!*"
                marker_index = text.find(marker)
                if marker_index < 0:
                    raise ValueError("Heart happy-ending marker missing from D1:4458")
                return PolicyAction.replace_text(
                    text[marker_index:].strip(),
                    "retain only four-survivor happy-ending transition",
                )
            if number >= 4496:
                return PolicyAction.delete("pre-story scene appended after ending")
            return PolicyAction.keep()

        if title_key == "nurse-love-addiction":
            if session >= 62:
                return PolicyAction.delete("non-canonical ending or heroine route")
            if session == 18 and number == 44:
                marker = "Accept her offer Don’t accept her offer"
                return (
                    PolicyAction.replace_text(
                        _replace_literal(text, marker), "remove visible umbrella choice labels"
                    )
                    if marker in text
                    else PolicyAction.keep()
                )
            if session == 18 and 49 <= number <= 60:
                return PolicyAction.delete("declined-umbrella alternative")
            if session == 28 and 90 <= number <= 108:
                return PolicyAction.delete("discarded ice-cream alternative")
            if session == 32 and number == 62:
                marker = "Look for it with her Don’t look for it"
                return (
                    PolicyAction.replace_text(
                        _replace_literal(text, marker), "remove visible search choice labels"
                    )
                    if marker in text
                    else PolicyAction.keep()
                )
            if session == 34 and number == 191:
                return PolicyAction.delete("four-way beach choice menu")
            if session == 34 and 277 <= number <= 637:
                return PolicyAction.delete("non-Nao beach alternative")
            if session == 38 and (
                92 <= number <= 95
                or 101 <= number <= 104
                or 119 <= number <= 145
                or 171 <= number <= 185
                or 194 <= number <= 223
            ):
                return PolicyAction.delete("non-Nao hospital alternative")
            if session == 59 and number == 403:
                marker = "Forgive her Don’t forgive her"
                return (
                    PolicyAction.replace_text(
                        _replace_literal(text, marker), "remove visible ending choice labels"
                    )
                    if marker in text
                    else PolicyAction.keep()
                )
            return PolicyAction.keep()

        return PolicyAction.keep()

    return policy


def _session_number(key: str) -> Optional[int]:
    match = re.fullmatch(r"session_(\d+)", str(key))
    return int(match.group(1)) if match else None


def normalize_text(value: object) -> str:
    text = str(value or "")
    text = (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2026", "...")
    )
    return re.sub(r"\s+", " ", text).strip()


def _conversation_index(conversation: dict) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for key, turns in conversation.items():
        session = _session_number(str(key))
        if session is None or not isinstance(turns, list):
            continue
        for turn in turns:
            dia_id = str(turn.get("dia_id") or "").strip()
            if dia_id:
                index[dia_id] = turn
    return index


def find_turn(conversation: dict, dia_id: str) -> Optional[dict]:
    return _conversation_index(conversation).get(str(dia_id or "").strip())


def _evidence_matches_turn(evidence: dict, turn: dict) -> bool:
    evidence_speaker = normalize_text(evidence.get("speaker")).casefold()
    source_speaker = normalize_text(turn.get("speaker")).casefold()
    if evidence_speaker and evidence_speaker != source_speaker:
        return False
    evidence_text = normalize_text(evidence.get("utterance"))
    source_text = normalize_text(turn.get("text"))
    return bool(source_text and (not evidence_text or evidence_text in source_text))


def _is_standalone_abstain(answer: object) -> bool:
    return re.sub(r"[\s()]", "", str(answer or "")).upper() == "F"


_DANGLING_REFERENCE_RE = re.compile(
    r"\b(?:this|the)\s+cited\s+(?:moment|dialogue|passage|exchange|scene)\b",
    re.IGNORECASE,
)

def sanitize_question_wording(question: object) -> str:
    """Retain source wording; naturalization is registry-only and ID-specific."""
    return str(question or "")


def sanitize_qa_item(item: dict) -> dict:
    sanitized = {
        key: deepcopy(item[key])
        for key in PUBLIC_QA_FIELDS
        if key in item
    }
    if "question" in sanitized:
        sanitized["question"] = sanitize_question_wording(sanitized["question"])
    return sanitized


def _migrate_canonical_qa_items(
    qa_items: list[dict],
    old_conversation: dict,
    new_conversation: dict,
    id_map: dict[str, str],
    title_key: str,
    contaminated_sessions: set[int],
) -> tuple[list[dict], dict]:
    del old_conversation  # The retained conversation is the only repair authority.
    new_index = _conversation_index(new_conversation)
    canonical: list[dict] = []
    audit = {"repaired": [], "removed": [], "migrated": [], "question_rewrite": []}

    for qa_index, raw_qa in enumerate(qa_items, start=1):
        qa = deepcopy(raw_qa)
        question = str(qa.get("question") or "")
        if _DANGLING_REFERENCE_RE.search(question):
            audit["removed"].append(
                {
                    "qa_index": qa_index,
                    "reason": "dangling_reference_question",
                }
            )
            continue

        raw_evidence = qa.get("evidence_dialogues") or []
        if not isinstance(raw_evidence, list):
            audit["removed"].append(
                {"qa_index": qa_index, "reason": "invalid_evidence_schema"}
            )
            continue

        if title_key == "arknights":
            source_sessions = set()
            for evidence in raw_evidence:
                match = re.fullmatch(r"D(\d+):\d+", str(evidence.get("dia_id") or ""))
                if match:
                    source_sessions.add(int(match.group(1)))
            if source_sessions & contaminated_sessions:
                audit["removed"].append(
                    {
                        "qa_index": qa_index,
                        "reason": "contaminated_option_session",
                    }
                )
                continue

        if not raw_evidence and not _is_standalone_abstain(qa.get("answer")):
            audit["removed"].append(
                {"qa_index": qa_index, "reason": "empty_evidence"}
            )
            continue

        migrated_evidence: list[dict] = []
        evidence_id_map: dict[str, str] = {}
        repairs: list[dict] = []
        failed_reason = ""

        for evidence_number, raw_item in enumerate(raw_evidence, start=1):
            if not isinstance(raw_item, dict):
                failed_reason = "invalid_evidence_schema"
                break
            evidence = deepcopy(raw_item)
            old_dia_id = str(evidence.get("dia_id") or "").strip()
            target_turn = None
            repaired = False
            new_dia_id = id_map.get(old_dia_id)
            if new_dia_id:
                candidate = new_index.get(new_dia_id)
                if candidate and _evidence_matches_turn(evidence, candidate):
                    target_turn = candidate

            if (
                target_turn is None
                and title_key == "heart-of-the-woods"
                and old_dia_id == "D1:4458"
                and re.search(
                    r"\breach(?:es)? out the open window to point at the ocean\b",
                    question,
                    re.IGNORECASE,
                )
                and new_dia_id in new_index
            ):
                target_turn = new_index[new_dia_id]
                evidence["utterance"] = target_turn.get("text", "")
                evidence["speaker"] = target_turn.get("speaker")
                repaired = True

            if target_turn is None:
                matches = [
                    turn
                    for turn in new_index.values()
                    if _evidence_matches_turn(evidence, turn)
                    and normalize_text(evidence.get("utterance"))
                ]
                if len(matches) != 1:
                    failed_reason = "evidence_not_uniquely_repairable"
                    break
                target_turn = matches[0]
                repaired = True

            old_evidence_id = str(evidence.get("id") or f"E{evidence_number}")
            new_evidence_id = f"E{len(migrated_evidence) + 1}"
            evidence_id_map[old_evidence_id] = new_evidence_id
            migrated_item = {
                "id": new_evidence_id,
                "dia_id": target_turn["dia_id"],
                "speaker": target_turn.get("speaker"),
                "utterance": evidence.get("utterance") or target_turn.get("text", ""),
            }
            for optional_key in ("option", "sequence_position", "character_speaker"):
                if optional_key in evidence:
                    migrated_item[optional_key] = deepcopy(evidence[optional_key])
            migrated_evidence.append(migrated_item)
            if repaired:
                repairs.append(
                    {"old_dia_id": old_dia_id, "new_dia_id": target_turn["dia_id"]}
                )

        if failed_reason:
            audit["removed"].append(
                {"qa_index": qa_index, "reason": failed_reason}
            )
            continue

        reasoning_steps = qa.get("reasoning_steps") or []
        migrated_steps = []
        reasoning_invalid = False
        for raw_step in reasoning_steps if isinstance(reasoning_steps, list) else []:
            if not isinstance(raw_step, dict):
                reasoning_invalid = True
                break
            step = deepcopy(raw_step)
            based_on = step.get("based_on") or []
            if not isinstance(based_on, list):
                based_on = [based_on]
            mapped_based_on = []
            for evidence_id in based_on:
                mapped_id = evidence_id_map.get(str(evidence_id))
                if not mapped_id:
                    reasoning_invalid = True
                    break
                if mapped_id not in mapped_based_on:
                    mapped_based_on.append(mapped_id)
            if reasoning_invalid:
                break
            step["based_on"] = mapped_based_on
            migrated_steps.append(step)
        if reasoning_invalid:
            audit["removed"].append(
                {
                    "qa_index": qa_index,
                    "reason": "unresolved_reasoning_evidence",
                }
            )
            continue

        qa["evidence_dialogues"] = migrated_evidence
        qa["reasoning_steps"] = migrated_steps
        canonical.append(qa)
        if repairs:
            audit["repaired"].append(
                {"qa_index": qa_index, "evidence_repairs": repairs}
            )
        elif any(
            str(old.get("dia_id") or "") != str(new.get("dia_id") or "")
            for old, new in zip(raw_evidence, migrated_evidence)
        ):
            audit["migrated"].append(qa_index)

    return canonical, audit


def publish_qa_items(
    canonical_qa_items: list[dict], title_key: str, audit: dict
) -> tuple[list[dict], dict]:
    """Apply pre-publication rewrites, then assign gapless public QA IDs."""
    survivors: list[tuple[dict, str, str]] = []
    protected_fields = (
        "question_type",
        "answer",
        "evidence_dialogues",
        "reasoning_steps",
    )

    for canonical_index, canonical_qa in enumerate(canonical_qa_items, start=1):
        rewrite_key = make_rewrite_key(title_key, canonical_index)
        qa = deepcopy(canonical_qa)
        retained_fields = {
            field: deepcopy(canonical_qa.get(field)) for field in protected_fields
        }
        source_stem = extract_core_question_text(
            qa.get("question", ""), unknown_placeholder=""
        )
        rewritten_stem, rewritten_options, rewrite_action = rewrite_publication_item(
            rewrite_key, source_stem, qa.get("option") or []
        )
        if rewritten_stem is None:
            audit["removed"].append(
                {
                    "qa_index": canonical_index,
                    "rewrite_key": rewrite_key,
                    "reason": "unsafe_question_rewrite",
                }
            )
            continue

        qa["question"] = rewritten_stem
        qa["option"] = rewritten_options
        qa.update(retained_fields)
        try:
            provisional, _normalization_audit = normalize_public_qa(
                qa, title_key, canonical_index
            )
        except ValueError as error:
            audit["removed"].append(
                {
                    "qa_index": canonical_index,
                    "rewrite_key": rewrite_key,
                    "reason": "invalid_public_schema",
                    "detail": str(error),
                }
            )
            continue
        validation_errors = validate_public_qa(provisional)
        if validation_errors:
            audit["removed"].append(
                {
                    "qa_index": canonical_index,
                    "rewrite_key": rewrite_key,
                    "reason": "invalid_public_schema",
                    "detail": "; ".join(validation_errors),
                }
            )
            continue
        survivors.append((qa, rewrite_key, rewrite_action))

    published: list[dict] = []
    for final_index, (qa, rewrite_key, rewrite_action) in enumerate(survivors, start=1):
        normalized, _normalization_audit = normalize_public_qa(qa, title_key, final_index)
        published.append(normalized)
        if rewrite_action == "rewrite":
            audit["question_rewrite"].append(
                {
                    "rewrite_key": rewrite_key,
                    "qa_id": normalized["qa_id"],
                    "action": "rewrite",
                }
            )
    return published, audit


def migrate_qa_items(
    qa_items: list[dict],
    old_conversation: dict,
    new_conversation: dict,
    id_map: dict[str, str],
    title_key: str,
    contaminated_sessions: set[int],
) -> tuple[list[dict], dict]:
    """Produce public QA from canonical QA in its surviving source order."""
    canonical, audit = _migrate_canonical_qa_items(
        qa_items,
        old_conversation,
        new_conversation,
        id_map,
        title_key,
        contaminated_sessions,
    )
    return publish_qa_items(canonical, title_key, audit)


def _iter_turns(conversation: dict):
    for key, turns in conversation.items():
        session = _session_number(str(key))
        if session is None or not isinstance(turns, list):
            continue
        for turn in turns:
            yield session, turn


def _is_extreme_noise(text: object) -> bool:
    raw = str(text or "")
    compact = re.sub(r"\s+", "", raw).casefold()
    if len(compact) < 1500:
        return False
    if re.search(r"(.)\1{499,}", compact):
        return True
    alphanumeric = re.sub(r"[^a-z0-9]", "", compact)
    if alphanumeric:
        dominant = max(alphanumeric.count(char) for char in set(alphanumeric))
        if dominant / len(alphanumeric) >= 0.45:
            return True
    words = re.findall(r"[a-z]+", raw.casefold())
    return len(words) >= 150 and len(set(words)) / len(words) < 0.03


def validate_curated_record(record: dict, title_key: str) -> list[str]:
    errors: list[str] = []
    conversation = record.get("conversation")
    qa_items = record.get("qa")
    if not isinstance(conversation, dict):
        return ["conversation is not an object"]
    if not isinstance(qa_items, list):
        return ["qa is not an array"]

    dialogue_index: dict[str, dict] = {}
    for key, turns in conversation.items():
        session = _session_number(str(key))
        if session is None:
            continue
        if not isinstance(turns, list):
            errors.append(f"{key} is not an array")
            continue
        for position, turn in enumerate(turns, start=1):
            expected_id = f"D{session}:{position}"
            actual_id = str(turn.get("dia_id") or "")
            if actual_id != expected_id:
                errors.append(
                    f"{key} has non-contiguous dia_id {actual_id} at position {position}"
                )
            if actual_id in dialogue_index:
                errors.append(f"duplicate dia_id {actual_id}")
            dialogue_index[actual_id] = turn
            text = str(turn.get("text") or "")
            if _is_extreme_noise(text):
                errors.append(f"extreme noise at {actual_id}")
            if title_key == "arknights" and re.search(r"(?:^|\n)Option_\d+:", text):
                errors.append(f"forbidden interactive option marker at {actual_id}")
            if title_key == "highway-blossoms" and "define: dis =" in text:
                errors.append(f"forbidden blooper marker at {actual_id}")
            if title_key == "nurse-love-addiction" and any(
                marker in text
                for marker in (
                    "Accept her offer Don’t accept her offer",
                    "Look for it with her Don’t look for it",
                    "Nao Sakuya Itsuki Me",
                    "Forgive her Don’t forgive her",
                    "Resist Don’t resist",
                    "I’ll go every day Maybe I’ll go like every three days",
                )
            ):
                errors.append(f"forbidden visible choice marker at {actual_id}")

    if title_key == "fata-morgana-requiem" and "session_5" in conversation:
        errors.append("forbidden backstage session_5 remains")
    if title_key == "nurse-love-addiction":
        for key in conversation:
            session = _session_number(str(key))
            if session is not None and session >= 62:
                errors.append(f"forbidden non-Nao route {key} remains")

    qa_ids: set[str] = set()
    for qa_index, qa in enumerate(qa_items, start=1):
        public_errors = validate_public_qa(qa)
        errors.extend(
            f"qa {qa_index} public schema: {error}" for error in public_errors
        )
        qa_id = str(qa.get("qa_id") or "")
        if qa_id in qa_ids:
            errors.append(f"duplicate qa_id {qa_id}")
        qa_ids.add(qa_id)
        question = str(qa.get("question") or "")
        if _DANGLING_REFERENCE_RE.search(question):
            errors.append(f"qa {qa_index} contains a dangling cited-moment reference")
        core_stem = extract_core_question_text(question, unknown_placeholder="")
        construction_issues = find_construction_issues(core_stem)
        if construction_issues:
            errors.append(
                f"qa {qa_index} retains construction wording: {', '.join(construction_issues)}"
            )
        evidence_items = qa.get("evidence_dialogues") or []
        if not isinstance(evidence_items, list):
            errors.append(f"qa {qa_index} evidence_dialogues is not an array")
            continue
        if not evidence_items and not _is_standalone_abstain(qa.get("answer")):
            errors.append(f"qa {qa_index} has no evidence")
        evidence_ids: set[str] = set()
        for evidence_position, evidence in enumerate(evidence_items, start=1):
            evidence_id = str(evidence.get("id") or "")
            expected_evidence_id = f"E{evidence_position}"
            if evidence_id != expected_evidence_id:
                errors.append(
                    f"qa {qa_index} evidence id {evidence_id} should be {expected_evidence_id}"
                )
            evidence_ids.add(evidence_id)
            dia_id = str(evidence.get("dia_id") or "")
            turn = dialogue_index.get(dia_id)
            if turn is None:
                errors.append(
                    f"qa {qa_index} evidence {evidence_id} references missing {dia_id}"
                )
                continue
            if normalize_text(evidence.get("speaker")).casefold() != normalize_text(
                turn.get("speaker")
            ).casefold():
                errors.append(
                    f"qa {qa_index} evidence {evidence_id} speaker mismatch at {dia_id}"
                )
            if not _evidence_matches_turn(evidence, turn):
                errors.append(
                    f"qa {qa_index} evidence {evidence_id} utterance mismatch at {dia_id}"
                )
        reasoning_steps = qa.get("reasoning_steps") or []
        if isinstance(reasoning_steps, list):
            for step_position, step in enumerate(reasoning_steps, start=1):
                if not isinstance(step, dict):
                    continue
                based_on = step.get("based_on") or []
                if not isinstance(based_on, list):
                    based_on = [based_on]
                for evidence_id in based_on:
                    if str(evidence_id) not in evidence_ids:
                        errors.append(
                            f"qa {qa_index} reasoning step {step_position} references missing {evidence_id}"
                        )
    return errors


INPUTS = {
    "9-nine-episode-1": (
        "runs/9-nine-_Episode_1_20260822_154315/result/9-nine-_Episode_1_final.json",
        "9-nine-_Episode_1_final.json",
    ),
    "a-kiss-for-the-petals": (
        "runs/a-kiss-for-the-petals-remembering-how-we-met_revised_20260824_104328/result/a-kiss-for-the-petals-remembering-how-we-met_revised_final.json",
        "a-kiss-for-the-petals-remembering-how-we-met_revised_final.json",
    ),
    "arknights": (
        "runs/arknights_revised_20260831_223935/result/arknights_revised_final.json",
        "arknights_revised_final.json",
    ),
    "fault-milestone-two": (
        "runs/fault-milestone-two-sidea-bove_revised_20260824_115522/result/fault-milestone-two-sidea-bove_revised_final.json",
        "fault-milestone-two-sidea-bove_revised_final.json",
    ),
    "heart-of-the-woods": (
        "runs/heart-of-the-woods_revised_20260824_122111/result/heart-of-the-woods_revised_final.json",
        "heart-of-the-woods_revised_final.json",
    ),
    "highway-blossoms": (
        "runs/highway-blossoms_revised_20260824_131041/result/highway-blossoms_revised_final.json",
        "highway-blossoms_revised_final.json",
    ),
    "nurse-love-addiction": (
        "runs/nurse-love-addiction_revised_20260828_132217/result/nurse-love-addiction_revised_final.json",
        "nurse-love-addiction_revised_final.json",
    ),
    "fata-morgana-requiem": (
        "runs/the-house-in-fata-morgana-a-requiem-for-innocence_revised_20260828_132537/result/the-house-in-fata-morgana-a-requiem-for-innocence_revised_final.json",
        "the-house-in-fata-morgana-a-requiem-for-innocence_revised_final.json",
    ),
}

AFFECTED_TITLES = {
    "arknights",
    "heart-of-the-woods",
    "highway-blossoms",
    "nurse-love-addiction",
    "fata-morgana-requiem",
}


def _dialogue_count(conversation: dict) -> int:
    return sum(1 for _session, _turn in _iter_turns(conversation))


def _json_bytes(data: object) -> bytes:
    return (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _remove_empty_deleted_sessions(conversation: dict, original: dict) -> dict:
    cleaned = deepcopy(conversation)
    removed_sessions = []
    for key in list(cleaned):
        if _session_number(str(key)) is not None and not cleaned[key] and original.get(key):
            removed_sessions.append(key)
            del cleaned[key]
    for session_key in removed_sessions:
        date_key = f"{session_key}_date_time"
        cleaned.pop(date_key, None)
    return cleaned


def _curate_record(record: dict, title_key: str) -> tuple[dict, dict]:
    original = deepcopy(record)
    original_conversation = original.get("conversation") or {}
    original_qa = original.get("qa") or []
    contaminated_sessions: set[int] = set()
    if title_key == "arknights":
        for session, turn in _iter_turns(original_conversation):
            if re.search(r"(?:^|\n)Option_\d+:", str(turn.get("text") or "")):
                contaminated_sessions.add(session)

    if title_key in AFFECTED_TITLES:
        curated_conversation, id_map, turn_audit = renumber_conversation(
            original_conversation, build_policy(title_key)
        )
        curated_conversation = _remove_empty_deleted_sessions(
            curated_conversation, original_conversation
        )
    else:
        curated_conversation = deepcopy(original_conversation)
        id_map = {
            str(turn.get("dia_id")): str(turn.get("dia_id"))
            for _session, turn in _iter_turns(curated_conversation)
            if str(turn.get("dia_id") or "")
        }
        turn_audit = []
    curated_qa, qa_audit = migrate_qa_items(
        original_qa,
        original_conversation,
        curated_conversation,
        id_map,
        title_key,
        contaminated_sessions,
    )
    curated_record = deepcopy(original)
    curated_record["conversation"] = curated_conversation
    curated_record["qa"] = curated_qa
    errors = validate_curated_record(curated_record, title_key)
    if errors:
        raise ValueError(f"{title_key} output validation failed: {errors[:20]}")

    return curated_record, {
        "original_dialogues": _dialogue_count(original_conversation),
        "final_dialogues": _dialogue_count(curated_conversation),
        "original_qa": len(original_qa),
        "final_qa": len(curated_qa),
        "deleted_turns": [
            item["dia_id"] for item in turn_audit if item["action"] == "delete"
        ],
        "sanitized_turns": [
            item["dia_id"] for item in turn_audit if item["action"] == "replace_text"
        ],
        "id_map_count": len(id_map),
        "contaminated_sessions": sorted(contaminated_sessions),
        "qa_audit": qa_audit,
        "validation": "pass",
    }


def curate_all(repo_root: Path, output_dir: Path, check: bool = False) -> dict:
    repo_root = Path(repo_root).resolve()
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    outputs: dict[str, bytes] = {}
    report = {"schema_version": 1, "titles": {}}
    emitted_qa_ids: set[str] = set()
    for title_key, (input_rel, output_name) in INPUTS.items():
        input_path = repo_root / input_rel
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
            raise ValueError(f"Unexpected one-record-array schema: {input_rel}")
        curated_record, title_report = _curate_record(payload[0], title_key)
        for qa in curated_record["qa"]:
            qa_id = qa["qa_id"]
            if qa_id in emitted_qa_ids:
                raise ValueError(f"duplicate qa_id across output titles: {qa_id}")
            emitted_qa_ids.add(qa_id)
        output_payload = [curated_record]
        outputs[output_name] = _json_bytes(output_payload)
        title_report["input"] = input_rel
        title_report["output"] = f"result_qa/{output_name}"
        report["titles"][title_key] = title_report

    outputs["curation_report.json"] = _json_bytes(report)

    if check:
        mismatches = []
        for name, expected in outputs.items():
            path = output_dir / name
            if not path.exists() or path.read_bytes() != expected:
                mismatches.append(name)
        if mismatches:
            raise ValueError(f"Output mismatch: {mismatches}")
        return report

    output_dir.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[Path, Path]] = []
    try:
        for name, content in outputs.items():
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=output_dir, prefix=f".{name}.", suffix=".tmp", delete=False
            ) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
                staged.append((Path(handle.name), output_dir / name))
        for temporary, destination in staged:
            os.replace(temporary, destination)
        staged.clear()
    finally:
        for temporary, _destination in staged:
            temporary.unlink(missing_ok=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate canonical novel benchmark results.")
    parser.add_argument("--output", default="result_qa")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    report = curate_all(repo_root, Path(args.output), check=args.check)
    for title, details in report["titles"].items():
        print(
            f"{title}: dialogues {details['original_dialogues']}->{details['final_dialogues']}, "
            f"qa {details['original_qa']}->{details['final_qa']}"
        )


def renumber_conversation(
    conversation: dict,
    policy: TurnPolicy,
) -> tuple[dict, dict[str, str], list[dict]]:
    curated = {}
    id_map: dict[str, str] = {}
    audit: list[dict] = []

    for key, value in conversation.items():
        session_number = _session_number(key)
        if session_number is None:
            curated[key] = deepcopy(value)
            continue

        retained: list[dict] = []
        for raw_turn in value if isinstance(value, list) else []:
            turn = deepcopy(raw_turn)
            old_id = str(turn.get("dia_id") or "").strip()
            action = policy(session_number, turn)
            if action.action == "delete":
                audit.append(
                    {"action": "delete", "dia_id": old_id, "reason": action.reason}
                )
                continue
            if action.action == "replace_text":
                turn["text"] = action.text or ""
                audit.append(
                    {
                        "action": "replace_text",
                        "dia_id": old_id,
                        "reason": action.reason,
                    }
                )
            elif action.action != "keep":
                raise ValueError(f"Unsupported policy action: {action.action}")

            new_id = f"D{session_number}:{len(retained) + 1}"
            turn["dia_id"] = new_id
            retained.append(turn)
            if old_id:
                id_map[old_id] = new_id
        curated[key] = retained

    return curated, id_map, audit


if __name__ == "__main__":
    main()
