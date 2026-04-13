from __future__ import annotations

import json
import re
from pathlib import Path


LINE_RE = re.compile(
    r"^\s*(?:(?P<lead>[^\s\[]+)\s+)?(?:\[(?P<bracket>[^\]]+)\]\s+)?(?P<rest>.+?)\s*$"
)

DIA_ID_RE = re.compile(r"^(?:[A-Za-z]?\d+(?::\d+)?|D\d+(?::\d+)?)$")


def _looks_like_dia_id(token: str) -> bool:
    return bool(DIA_ID_RE.match(token.strip()))


def parse_evidence_dialogue(text: str) -> dict[str, str]:
    if not isinstance(text, str):
        return {
            "speaker": "NARRATION",
            "utterance": str(text),
            "dia_id": "",
        }

    match = LINE_RE.match(text)
    if not match:
        return {
            "speaker": "NARRATION",
            "utterance": text.strip(),
            "dia_id": "",
        }

    lead = (match.group("lead") or "").strip()
    bracket = (match.group("bracket") or "").strip()
    rest = (match.group("rest") or "").strip()

    # Prefer bracket token as dia_id when available, because many rows use
    # labels like F1/F2 before [dia_id].
    if bracket:
        dia_id = bracket
    elif lead and _looks_like_dia_id(lead):
        dia_id = lead
    else:
        dia_id = ""

    if "：" in rest:
        speaker, utterance = rest.split("：", 1)
    elif ":" in rest:
        speaker, utterance = rest.split(":", 1)
    else:
        speaker, utterance = "NARRATION", rest

    speaker = speaker.strip() or "NARRATION"
    utterance = utterance.strip()

    return {
        "speaker": speaker,
        "utterance": utterance,
        "dia_id": dia_id,
    }


def convert_evidence_dialogues(payload: object) -> int:
    changed = 0

    if isinstance(payload, dict):
        for evidence_key in ("evidence_dialogues", "evidences"):
            if isinstance(payload.get(evidence_key), list):
                converted_dialogues = []
                for dialogue in payload[evidence_key]:
                    if isinstance(dialogue, str):
                        converted_dialogues.append(parse_evidence_dialogue(dialogue))
                        changed += 1
                    else:
                        converted_dialogues.append(dialogue)
                payload[evidence_key] = converted_dialogues

        for value in payload.values():
            changed += convert_evidence_dialogues(value)

    elif isinstance(payload, list):
        for item in payload:
            changed += convert_evidence_dialogues(item)

    return changed


def main() -> None:
    base_dir = Path(__file__).resolve().parent / "new_qa"
    v0_files = sorted(base_dir.glob("*_v0.json"))
    final_files = sorted(base_dir.glob("*_final/all_questions_new.json"))
    json_files = v0_files + final_files

    if not json_files:
        raise SystemExit(f"No target files found under {base_dir}")

    conversation_by_stem: dict[str, object] = {}
    for file_path in v0_files:
        stem = file_path.stem.removesuffix("_v0")
        with file_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list) and data and isinstance(data[0], dict):
            conversation_obj = data[0].get("conversation")
            if conversation_obj is not None:
                conversation_by_stem[stem] = conversation_obj

    for file_path in json_files:
        with file_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        if (
            isinstance(data, list)
            and data
            and isinstance(data[0], dict)
            and data[0].get("conversation") is None
        ):
            stem = file_path.parent.name.removesuffix("_final") if file_path.parent.name.endswith("_final") else file_path.stem.removesuffix("_v0")
            conversation_obj = conversation_by_stem.get(stem)
            if conversation_obj is not None:
                data[0]["conversation"] = conversation_obj

        changed_count = convert_evidence_dialogues(data)

        with file_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=4)
            handle.write("\n")

        print(f"{file_path.name}: updated {changed_count} evidence_dialogues entries")


if __name__ == "__main__":
    main()