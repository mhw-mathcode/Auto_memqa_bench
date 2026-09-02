"""Standalone stem-only regression checks for benchmark naturalization."""

from __future__ import annotations

import json
import re
import sys
import tempfile
from collections import Counter
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import src.curate_novel_benchmark as curation
from src.benchmark_qa_schema import extract_core_question_text
from src.benchmark_question_rewrites import (
    OPTION_REWRITES,
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
    3: "Which sequence best traces Michel and Imeon's debate over survival?",
    6: "Reconstruct the progression of Imeon's conversation about Danish seafaring.",
    12: "Place the listed turning points from Imeon's first extended conversation with Michel in story order.",
    13: "Which sequence best traces Imeon's first visit to Michel's mansion?",
    15: "From the options, identify the progression of Imeon's debate with Michel about adventure.",
    19: "Reconstruct the progression of Imeon's opening storyline at the mansion.",
    22: "Place the listed turning points from Imeon's growing entanglement at the mansion in story order.",
    26: "From the options, identify the progression of the commotion over an unexpected mansion visitor.",
    30: "Which sequence best traces Imeon's evolving outlook during his conversation with Michel?",
    33: "Reconstruct the progression of Michel's attempt to handle the unexpected visitor.",
    34: "Place the listed turning points from the visitor's exchange with Michel in story order.",
    37: "From the options, identify the progression of the visitor's disruption at the mansion.",
    38: "Arrange the key exchanges in Michel's response to the intrusion in story order.",
    39: "Track the changes in the mansion's atmosphere during the visitor's arrival.",
    41: "Which ordering best captures the household's response to the unexpected guest?",
    43: "Put the developments from Georges's studio commotion in story order.",
    45: "Arrange the stages of Imeon's early mansion arc in story order.",
    48: "Track the changes in Mell's bond with Morgana across their conversation.",
    51: "Which sequence best traces Morgana's first night at the estate?",
    55: "Reconstruct the progression of Morgana's doorway encounter.",
    60: "Place the listed turning points from a period of strain at the estate in story order.",
    64: "From the options, identify the progression of Morgana's search for security at the estate.",
    68: "Arrange the developments in Morgana's departure from the great hall in story order.",
    70: "Which sequence best traces Jacopo's protective role toward Morgana?",
    73: "Reconstruct the progression of Morgana's early bonds at the estate.",
    75: "Place the listed turning points from a tense night at the estate in story order.",
    76: "From the options, identify the progression of Morgana's vulnerability during her early stay.",
    80: "Arrange the developments in the estate's escalating unrest in story order.",
    84: "Track the changes in the estate's atmosphere of crisis across this passage.",
    88: "Track the changes in tone as Michel's movie date with Giselle begins.",
    90: "Which ordering best captures the couple's reaction to the horror film?",
    94: "Which ordering best captures the overall shape of Michel and Giselle's movie date?",
    96: "Put the developments concerning the couple's attempt to process the horror film in story order.",
    97: "Which sequence best traces the couple's commitment during their reunion conversation?",
    98: "Put the developments from the closing phase of the date in story order.",
    101: "Reconstruct the progression of Michel's choice about a future with Giselle.",
    104: "Place the listed turning points from the couple's discussion of identity in story order.",
    108: "From the options, identify the progression of Michel's invitation to Giselle.",
    111: "Arrange the decisions shaping Michel's readiness to build a life with Giselle in story order.",
    113: "Track the changes in Michel's thinking during the post-film conversation.",
    119: "Which ordering best captures the emotional arc of the movie outing?",
    121: "Put the developments concerning Michel's response to the film in story order.",
    124: "Arrange the exchanges in the couple's reflection on living again in story order.",
    127: "Track the changes in the date's reflective tone across the conversation.",
    131: "Follow the arc of the couple's shared-future conversation by choosing the correct sequence.",
    137: "Which ordering best captures Morgana's approach to Midsummer?",
    140: "Which ordering best captures Morgana's final confrontation over the illusion?",
    143: "Put the developments concerning Morgana's effort to recover missed experiences in story order.",
    151: "Put the developments concerning Morgana's adjustment to the idealized realm in story order.",
    155: "Identify the story order of the developments shaping Morgana's view of her lost childhood.",
    158: "Identify the story order of the developments in Morgana's pursuit of an ordinary life.",
    162: "Identify the story order of the developments shaping Morgana's engagement with the peaceful realm.",
    164: "Identify the story order of the developments in Morgana's outlook on Midsummer.",
    169: "Identify the story order of the developments shaping Morgana's view of the idealized realm.",
    173: "Follow the arc of Morgana's outlook during the peaceful interlude by choosing the correct sequence.",
    176: "Follow the arc of Morgana's reassessment of her companion by choosing the correct sequence.",
    180: "Follow the arc of Morgana's relationship in the peaceful realm by choosing the correct sequence.",
}

APPROVED_ROUND2_STEMS = {
    "fault-milestone-two-R0046": (
        "Which statements accurately summarize Volthal and Flora's report about the missing pair?"
    ),
    "highway-blossoms-R0030": (
        "Which sequence correctly orders the travelers' remarks during discussions "
        "of destinations and personal goals?"
    ),
    "highway-blossoms-R0067": (
        "Which statements accurately describe Amber and Marina's decisions as they prepare to leave Arches?"
    ),
}

FATA_STYLE_FAMILY_PREFIXES = {
    "which-sequence": "Which sequence best traces ",
    "reconstruct": "Reconstruct the progression of ",
    "place-turning-points": "Place the listed turning points ",
    "from-options": "From the options, identify the progression of ",
    "arrange": "Arrange the ",
    "track": "Track the changes in ",
    "which-ordering": "Which ordering best captures ",
    "put-in-order": "Put the developments ",
    "identify-order": "Identify the story order of ",
    "follow-arc": "Follow the arc of ",
}

# Literal gold-order checklist used only to detect proposition enumeration in a
# stem.  It is independent of the approved-stem snapshot above and is verified
# against the canonical answer field before checking option/stem overlap.
FATA_GOLD_SEQUENCE_CHECKLIST = {
    3: "CDBA", 6: "BCAD", 12: "DCBA", 13: "DBCA", 15: "BADC",
    19: "CBAD", 22: "BCDA", 26: "BDCA", 30: "BCAD", 33: "CADB",
    34: "DBCA", 37: "DCAB", 38: "ABDC", 39: "ADBC", 41: "BDAC",
    43: "CDBA", 45: "CABD", 48: "ACBD", 51: "CBAD", 55: "BCAD",
    60: "BDCA", 64: "ADBC", 68: "CBAD", 70: "DABC", 73: "ACDB",
    75: "ADBC", 76: "ADCB", 80: "BCAD", 84: "BDAC", 88: "BDCA",
    90: "BADC", 94: "BCDA", 96: "CBDA", 97: "CBAD", 98: "DCAB",
    101: "DBAC", 104: "ADCB", 108: "DCBA", 111: "BADC", 113: "ACDB",
    119: "DABC", 121: "BACD", 124: "ADBC", 127: "BACD", 131: "DABC",
    137: "DACB", 140: "CBDA", 143: "CDAB", 151: "BADC", 155: "BDAC",
    158: "ADBC", 162: "BCDA", 164: "DACB", 169: "ABDC", 173: "DACB",
    176: "ACBD", 180: "ABCD",
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
    r"\bearliest\b|\blatest\b|\bfrom\b(?!\s+the options\b).+\bto\b",
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
    assert r0068 == (
        "Arrange the developments in Morgana's departure from the great hall in story order."
    )
    assert not re.search(r"rescue|visitor|refuge", r0068, re.I)


def test_fata_style_distribution_and_gold_enumeration(repo_root: Path) -> None:
    assert set(FATA_GOLD_SEQUENCE_CHECKLIST) == FATA_ORDERING_IDS
    family_counts: Counter[str] = Counter()
    moment_reference_count = 0
    items = _canonical_items(repo_root, "fata-morgana-requiem")

    for item_id in sorted(FATA_ORDERING_IDS):
        rewrite_key = f"fata-morgana-requiem-R{item_id:04d}"
        stem = QUESTION_REWRITES[rewrite_key]
        assert stem is not None
        matching_families = [
            family
            for family, prefix in FATA_STYLE_FAMILY_PREFIXES.items()
            if stem.startswith(prefix)
        ]
        assert len(matching_families) == 1, (rewrite_key, matching_families)
        family_counts[matching_families[0]] += 1
        moment_reference_count += len(
            re.findall(r"\b(?:these|selected) moments\b", stem, re.I)
        )

        qa = items[item_id - 1]
        answer_letters = "".join(re.findall(r"[A-D]", str(qa["answer"])))
        assert answer_letters == FATA_GOLD_SEQUENCE_CHECKLIST[item_id]
        enumerated_options = [
            letter
            for letter in answer_letters
            if _shared_ngram(stem, qa["option"][ord(letter) - ord("A")], size=3)
        ]
        assert len(enumerated_options) <= 1, (
            f"{rewrite_key} enumerates phrases from multiple gold propositions: "
            f"{enumerated_options}"
        )

    assert len(family_counts) >= 8, family_counts
    assert max(family_counts.values()) <= 10, family_counts
    assert moment_reference_count <= 5, moment_reference_count
    print(f"Fata style families: {dict(sorted(family_counts.items()))}")
    print(
        "Fata these/selected-moments references: "
        f"{moment_reference_count}/57 (maximum 5)"
    )


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
            expected_options = list(before["option"])
            for letter, body in OPTION_REWRITES.get(rewrite_key, {}).items():
                option_index = ord(letter) - ord("A")
                assert expected_options[option_index].startswith(f"{letter}. ")
                expected_options[option_index] = f"{letter}. {body}"
            assert options == expected_options
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
    test_fata_style_distribution_and_gold_enumeration(repo_root)
    print("Fata gold-sequence enumeration checklist passed: 57/57 stems")
    test_rewrites_preserve_protected_fields(repo_root)
    test_clean_rebuild_stems(repo_root)
    print("stem curation regression checks passed")


if __name__ == "__main__":
    main()
