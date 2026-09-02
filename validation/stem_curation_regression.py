"""Standalone stem-only regression checks for benchmark naturalization."""

from __future__ import annotations

import json
import re
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import src.curate_novel_benchmark as curation
from src.benchmark_qa_schema import extract_core_question_text
from src.benchmark_question_rewrites import (
    QUESTION_REWRITES,
    find_construction_issues,
    rewrite_publication_item,
)


INVENTED_TITLE_KEYS = {
    "fault-milestone-two": {
        5, 13, 16, 18, 21, 22, 30, 35, 39, 40, 44, 45, 46, 47, 53, 55,
        58, 61, 62, 65, 67, 74, 75, 76, 80, 88, 90, 91, 98, 102, 106,
        107, 108, 111, 112,
    },
    "highway-blossoms": {
        10, 14, 19, 21, 22, 23, 27, 30, 31, 36, 38, 39, 47, 50, 51, 57,
        61, 62, 67, 69, 71, 73, 75, 77, 81, 83, 88, 91, 96, 97, 99, 101,
        104, 106,
    },
}

FATA_ORDERING_IDS = {
    3, 6, 12, 13, 15, 19, 22, 26, 30, 33, 34, 37, 38, 39, 41, 43,
    45, 48, 51, 55, 60, 64, 68, 70, 73, 75, 76, 80, 84, 88, 90, 94,
    96, 97, 98, 101, 104, 108, 111, 113, 119, 121, 124, 127, 131, 137,
    140, 143, 151, 155, 158, 162, 164, 169, 173, 176, 180,
}

# Each stem below was reviewed against its own options and evidence.  These exact
# anchors deliberately name one scene, relationship, decision, or story phase;
# none narrates two or more option events in the gold-answer order.
APPROVED_FATA_STEMS = {
    3: "How do the selected moments unfold during Imeon's conversation with Michel about survival?",
    6: "How does the conversation about Danish seafaring unfold?",
    12: "What is the chronology of the selected moments from Imeon's first extended conversation with Michel?",
    13: "During Imeon's first visit to the mansion, what is the order of these moments?",
    15: "How does Imeon's conversation with Michel about adventure unfold?",
    19: "What is the story order of these moments in Imeon's early arc?",
    22: "How do the selected moments in Imeon's early mansion storyline unfold?",
    26: "How does the disturbance surrounding an unexpected mansion visitor unfold?",
    30: "What is the order of these moments in Imeon's conversation with Michel?",
    33: "How does Michel's first encounter with the unexpected visitor unfold?",
    34: "What is the chronology of the selected moments from Imeon's arrival at the mansion?",
    37: "How does the mansion commotion surrounding the visitor unfold?",
    38: "During the unexpected visitor's arrival, what is the order of these moments?",
    39: "How does the early disturbance inside the mansion unfold?",
    41: "What is the chronology of these moments during the mansion's early commotion?",
    43: "How do the selected moments from the studio disturbance unfold?",
    45: "What is the story order of these moments in Imeon's early mansion arc?",
    48: "How does Mell's conversation with Morgana about friendship unfold?",
    51: "During Morgana's first night at the estate, what is the order of these moments?",
    55: "How does the disturbance at Morgana's doorway unfold?",
    60: "What is the chronology of these selected moments from life at the estate?",
    64: "How do the selected moments from Morgana's unsettled first night unfold?",
    68: "How does Morgana's move away from the great hall unfold?",
    70: "How does Jacopo's care for Morgana unfold?",
    73: "What is the chronology of these moments in Morgana's early relationships at the estate?",
    75: "During the estate's unsettled night, what is the order of these moments?",
    76: "How do the selected moments from Morgana's early days at the estate unfold?",
    80: "What is the chronology of these moments during the estate's period of upheaval?",
    84: "How do the selected moments from the estate's upheaval unfold?",
    88: "How does Michel's movie date with Giselle get underway?",
    90: "How does the couple's post-film conversation unfold?",
    94: "During Michel and Giselle's movie date, what is the order of these moments?",
    96: "How does Giselle's reaction to the film develop during the date?",
    97: "How does Michel's reunion conversation with Giselle unfold?",
    98: "What is the chronology of these moments near the end of the date?",
    101: "How does Michel's decision about a future with Giselle unfold?",
    104: "During the post-film discussion, what is the order of these moments?",
    108: "How does Michel's invitation to Giselle unfold?",
    111: "How does Michel's conversation about building a life with Giselle unfold?",
    113: "What is the chronology of these moments in the couple's post-film conversation?",
    119: "Across the movie outing, what is the order of these selected moments?",
    121: "How does Michel's response to the film develop over the date?",
    124: "How does the couple's reflection on living again unfold?",
    127: "How does the date's reflective conversation unfold?",
    131: "How does the couple's conversation about a shared future unfold?",
    137: "How does Morgana's conversation about Midsummer unfold?",
    140: "How does Morgana's final conversation about illusion unfold?",
    143: "How does Morgana's effort to recover missed experiences unfold?",
    151: "What is the chronology of these moments from Morgana's time in the idealized realm?",
    155: "How does Morgana's reflection on her lost childhood unfold?",
    158: "How does Morgana's effort to experience an ordinary life unfold?",
    162: "What is the chronology of these moments as Morgana adjusts to the peaceful realm?",
    164: "During Morgana's time in the peaceful realm, what is the order of these moments?",
    169: "How does Morgana's view of the idealized realm develop?",
    173: "How does Morgana's outlook during the peaceful interlude develop?",
    176: "How does Morgana's reassessment of her companion unfold?",
    180: "How does Morgana's relationship with her companion evolve in the peaceful realm?",
}

APPROVED_ROUND2_STEMS = {
    "fault-milestone-two-R0046": (
        "Which statements accurately summarize Volthal and Flora's report about the missing pair?"
    ),
    "highway-blossoms-R0030": (
        "What is the chronology of the trip- and future-related moments involving "
        "Marina, the canyon guide, Linda, Jane, and Lacey?"
    ),
    "highway-blossoms-R0067": (
        "Which statements accurately describe Amber and Marina's decisions as they prepare to leave Arches?"
    ),
}

PREVIOUS_TASK3_KEYS = {
    "fault-milestone-two": {6, 8, 29, 37, 70, 81, 96, 104},
    "highway-blossoms": {11, 29, 41, 43, 45, 55, 72, 84, 85, 94},
    "fata-morgana-requiem": FATA_ORDERING_IDS,
    "nurse-love-addiction": set(range(1, 18)),
}

ORDERING_LEAKAGE_KEYS = {
    *(f"fata-morgana-requiem-R{item_id:04d}" for item_id in FATA_ORDERING_IDS),
    "highway-blossoms-R0011",
    "highway-blossoms-R0029",
    "highway-blossoms-R0055",
    "highway-blossoms-R0072",
    "highway-blossoms-R0094",
}

_DIRECTIONAL_LEAK_RE = re.compile(
    r"\bbefore\b|\bafter\b|\bpreced(?:e|es|ed|ing)\b|\blater\b|"
    r"\bearliest\b|\blatest\b|\bfrom\b.+\bto\b",
    re.I,
)
_QUOTED_FRAGMENT_RE = re.compile(r"[“\"]([^”\"]{12,})[”\"]")


def _canonical_items(repo_root: Path, title_key: str) -> list[dict]:
    input_rel, _output_name = curation.INPUTS[title_key]
    record = json.loads((repo_root / input_rel).read_text(encoding="utf-8"))[0]
    conversation, id_map, _audit = curation.renumber_conversation(
        record["conversation"], curation.build_policy(title_key)
    )
    conversation = curation._remove_empty_deleted_sessions(
        conversation, record["conversation"]
    )
    items, _audit = curation._migrate_canonical_qa_items(
        record["qa"],
        record["conversation"],
        conversation,
        id_map,
        title_key,
        set(),
    )
    return items


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def _shared_ngram(stem: str, option: str, size: int = 5) -> bool:
    stem_tokens = _tokens(stem)
    option_tokens = _tokens(re.sub(r"^[A-F][.:]\s*", "", option))
    stem_ngrams = {
        tuple(stem_tokens[index : index + size])
        for index in range(max(0, len(stem_tokens) - size + 1))
    }
    return any(
        tuple(option_tokens[index : index + size]) in stem_ngrams
        for index in range(max(0, len(option_tokens) - size + 1))
    )


def _answer_indices(answer: object) -> list[int]:
    return [ord(letter) - ord("A") for letter in re.findall(r"[A-F]", str(answer))]


def test_invented_titles_are_explicitly_rewritten(repo_root: Path) -> None:
    for title_key, item_ids in INVENTED_TITLE_KEYS.items():
        items = _canonical_items(repo_root, title_key)
        for item_id in sorted(item_ids):
            rewrite_key = f"{title_key}-R{item_id:04d}"
            source = extract_core_question_text(items[item_id - 1]["question"], "")
            quoted = _QUOTED_FRAGMENT_RE.findall(source)
            assert quoted, f"{rewrite_key} no longer contains the reviewed quoted title"
            assert rewrite_key in QUESTION_REWRITES, f"missing rewrite for {rewrite_key}"
            rewritten = QUESTION_REWRITES[rewrite_key]
            assert rewritten is not None
            assert all(fragment.casefold() not in rewritten.casefold() for fragment in quoted)


def test_task3_stems_do_not_leak_gold_answers(repo_root: Path) -> None:
    assert QUESTION_REWRITES["highway-blossoms-R0045"] is None

    reviewed = {
        title_key: set(item_ids) | PREVIOUS_TASK3_KEYS.get(title_key, set())
        for title_key, item_ids in INVENTED_TITLE_KEYS.items()
    }
    for title_key, item_ids in PREVIOUS_TASK3_KEYS.items():
        reviewed.setdefault(title_key, set()).update(item_ids)

    for title_key, item_ids in reviewed.items():
        items = _canonical_items(repo_root, title_key)
        for item_id in sorted(item_ids):
            rewrite_key = f"{title_key}-R{item_id:04d}"
            rewritten = QUESTION_REWRITES.get(rewrite_key)
            if rewritten is None:
                continue
            qa = items[item_id - 1]
            for option_index in _answer_indices(qa.get("answer")):
                assert not _shared_ngram(rewritten, qa["option"][option_index]), (
                    f"{rewrite_key} repeats a five-word phrase from a gold option"
                )
            if rewrite_key in ORDERING_LEAKAGE_KEYS:
                assert not _DIRECTIONAL_LEAK_RE.search(rewritten), (
                    f"{rewrite_key} contains directional chronology leakage"
                )

    fault_r0006 = QUESTION_REWRITES["fault-milestone-two-R0006"]
    assert fault_r0006 is not None
    assert "helplessness" not in fault_r0006.casefold()
    assert "ultimatum" not in fault_r0006.casefold()


def test_round2_stems_match_manual_approval_list() -> None:
    assert set(APPROVED_FATA_STEMS) == FATA_ORDERING_IDS
    for item_id, approved_stem in APPROVED_FATA_STEMS.items():
        rewrite_key = f"fata-morgana-requiem-R{item_id:04d}"
        assert QUESTION_REWRITES[rewrite_key] == approved_stem, rewrite_key
    for rewrite_key, approved_stem in APPROVED_ROUND2_STEMS.items():
        assert QUESTION_REWRITES[rewrite_key] == approved_stem, rewrite_key

    # R0068 is the review's concrete regression case: one scene anchor replaces
    # the previous rescue -> visitor -> refuge narration.
    r0068 = QUESTION_REWRITES["fata-morgana-requiem-R0068"]
    assert r0068 == "How does Morgana's move away from the great hall unfold?"
    assert not re.search(r"rescue|visitor|refuge", r0068, re.I)


def test_rewrites_preserve_protected_fields(repo_root: Path) -> None:
    reviewed = {
        title_key: set(item_ids) | PREVIOUS_TASK3_KEYS.get(title_key, set())
        for title_key, item_ids in INVENTED_TITLE_KEYS.items()
    }
    for title_key, item_ids in PREVIOUS_TASK3_KEYS.items():
        reviewed.setdefault(title_key, set()).update(item_ids)

    for title_key, item_ids in reviewed.items():
        items = _canonical_items(repo_root, title_key)
        for item_id in sorted(item_ids):
            qa = items[item_id - 1]
            rewrite_key = f"{title_key}-R{item_id:04d}"
            before = deepcopy(qa)
            rewritten, options, _action = rewrite_publication_item(
                rewrite_key,
                extract_core_question_text(qa["question"], ""),
                qa["option"],
            )
            assert options == before["option"]
            assert qa == before
            if rewritten is not None:
                assert "Select all that apply" not in rewritten


def test_clean_rebuild_stems(repo_root: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="stem-curation-regression-") as temp_dir:
        report = curation.curate_all(repo_root, Path(temp_dir))
        published_count = 0
        highway_items: list[dict] = []
        for title_key, (_input_rel, output_name) in curation.INPUTS.items():
            record = json.loads(
                (Path(temp_dir) / output_name).read_text(encoding="utf-8")
            )[0]
            if title_key == "highway-blossoms":
                highway_items = record["qa"]
            for expected_index, qa in enumerate(record["qa"], start=1):
                published_count += 1
                assert qa["qa_id"] == f"{title_key}-Q{expected_index:04d}"
                stem = extract_core_question_text(qa["question"], "")
                assert find_construction_issues(stem) == []

        assert highway_items
        removed = report["titles"]["highway-blossoms"]["qa_audit"]["removed"]
        assert {
            "qa_index": 45,
            "rewrite_key": "highway-blossoms-R0045",
            "reason": "unsafe_question_rewrite",
        } in removed
        rewrites = report["titles"]["highway-blossoms"]["qa_audit"][
            "question_rewrite"
        ]
        assert {
            "rewrite_key": "highway-blossoms-R0085",
            "qa_id": "highway-blossoms-Q0084",
            "action": "rewrite",
        } in rewrites
        highway_q0084 = highway_items[83]
        assert highway_q0084["qa_id"] == "highway-blossoms-Q0084"
        assert "share of the treasure" in extract_core_question_text(
            highway_q0084["question"], ""
        )
        highway_q0085 = highway_items[84]
        assert highway_q0085["qa_id"] == "highway-blossoms-Q0085"
        assert find_construction_issues(
            extract_core_question_text(highway_q0085["question"], "")
        ) == []
        print(f"clean rebuild stem audit passed: {published_count} published questions")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    test_invented_titles_are_explicitly_rewritten(repo_root)
    print(
        "invented-title audit passed: "
        f"{sum(map(len, INVENTED_TITLE_KEYS.values()))}/69 reviewed stems"
    )
    test_task3_stems_do_not_leak_gold_answers(repo_root)
    reviewed_count = sum(
        len(set(item_ids) | PREVIOUS_TASK3_KEYS.get(title_key, set()))
        for title_key, item_ids in INVENTED_TITLE_KEYS.items()
    ) + sum(
        len(item_ids)
        for title_key, item_ids in PREVIOUS_TASK3_KEYS.items()
        if title_key not in INVENTED_TITLE_KEYS
    )
    deleted_count = sum(
        QUESTION_REWRITES.get(
            f"{title_key}-R{item_id:04d}"
        ) is None
        for title_key, item_ids in PREVIOUS_TASK3_KEYS.items()
        for item_id in item_ids
    )
    print(
        "gold-answer leakage audit passed: "
        f"{reviewed_count - deleted_count} live Task 3 stems"
    )
    test_round2_stems_match_manual_approval_list()
    print("manual Fata non-leak approval list passed: 57/57 stems")
    test_rewrites_preserve_protected_fields(repo_root)
    test_clean_rebuild_stems(repo_root)
    print("stem curation regression checks passed")


if __name__ == "__main__":
    main()
