from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.curate_novel_benchmark import (
    PolicyAction,
    build_policy,
    migrate_qa_items,
    renumber_conversation,
    sanitize_qa_item,
    sanitize_question_wording,
    validate_curated_record,
)
from src.benchmark_qa_schema import (
    make_qa_id,
    normalize_public_qa,
    validate_public_qa,
)
from src.benchmark_question_rewrites import (
    QUESTION_REWRITES,
    find_construction_issues,
    rewrite_question_stem,
)
from src.mcq_scoring import ORDERING_INSTRUCTION
from src.question_formatting import extract_core_question_text


SOURCE_QA_FILES = {
    "9-nine-episode-1": "9-nine-_Episode_1_final.json",
    "a-kiss-for-the-petals": (
        "a-kiss-for-the-petals-remembering-how-we-met_revised_final.json"
    ),
    "arknights": "arknights_revised_final.json",
    "fault-milestone-two": "fault-milestone-two-sidea-bove_revised_final.json",
    "heart-of-the-woods": "heart-of-the-woods_revised_final.json",
    "highway-blossoms": "highway-blossoms_revised_final.json",
    "nurse-love-addiction": "nurse-love-addiction_revised_final.json",
    "fata-morgana-requiem": (
        "the-house-in-fata-morgana-a-requiem-for-innocence_revised_final.json"
    ),
}


def _conversation(*turns: dict) -> dict:
    return {
        "speaker_0": "Narrator",
        "speaker_1": "Alice",
        "session_1": list(turns),
    }


def _turn(dia_id: str, speaker: str, text: str) -> dict:
    return {"dia_id": dia_id, "speaker": speaker, "text": text}


def test_generic_filter_renumbers_and_preserves_metadata() -> None:
    original = _conversation(
        _turn("D1:1", "Alice", "keep one"),
        _turn("D1:2", "Narrator", "delete me"),
        _turn("D1:3", "Alice", "old wording"),
    )

    def policy(session: int, turn: dict) -> PolicyAction:
        if turn["dia_id"] == "D1:2":
            return PolicyAction.delete("fixture deletion")
        if turn["dia_id"] == "D1:3":
            return PolicyAction.replace_text("new wording", "fixture sanitation")
        return PolicyAction.keep()

    curated, id_map, audit = renumber_conversation(original, policy)
    assert curated["speaker_1"] == "Alice"
    assert [turn["dia_id"] for turn in curated["session_1"]] == ["D1:1", "D1:2"]
    assert [turn["text"] for turn in curated["session_1"]] == ["keep one", "new wording"]
    assert id_map == {"D1:1": "D1:1", "D1:3": "D1:2"}
    assert [entry["action"] for entry in audit] == ["delete", "replace_text"]


def _policy_result(title: str, dia_id: str, text: str = "text") -> PolicyAction:
    session = int(dia_id.split(":", 1)[0][1:])
    return build_policy(title)(session, _turn(dia_id, "Narrator", text))


def test_title_policies_keep_only_canonical_content() -> None:
    assert _policy_result("highway-blossoms", "D1:4053").action == "keep"
    assert _policy_result("highway-blossoms", "D1:4054").action == "delete"
    assert _policy_result("fata-morgana-requiem", "D2:7706").action == "delete"
    assert _policy_result("fata-morgana-requiem", "D2:7707").action == "keep"
    assert _policy_result("fata-morgana-requiem", "D5:1").action == "delete"
    assert _policy_result("arknights", "D5:1", "Story\nOption_1: Yes").action == "delete"
    assert _policy_result("arknights", "D5:2", "Story only").action == "keep"

    assert _policy_result("heart-of-the-woods", "D1:3942").action == "keep"
    assert _policy_result("heart-of-the-woods", "D1:3943").action == "delete"
    assert _policy_result("heart-of-the-woods", "D1:4031").action == "delete"
    assert _policy_result("heart-of-the-woods", "D1:4032").action == "keep"
    mixed_ending = "bad ending\n*Madison, look! There it is!*\nhappy beach transition"
    heart_action = _policy_result("heart-of-the-woods", "D1:4458", mixed_ending)
    assert heart_action.action == "replace_text"
    assert heart_action.text == "*Madison, look! There it is!*\nhappy beach transition"
    assert _policy_result("heart-of-the-woods", "D1:4495").action == "keep"
    assert _policy_result("heart-of-the-woods", "D1:4496").action == "delete"

    assert _policy_result("nurse-love-addiction", "D61:107").action == "keep"
    assert _policy_result("nurse-love-addiction", "D62:1").action == "delete"
    assert _policy_result("nurse-love-addiction", "D18:48").action == "keep"
    assert _policy_result("nurse-love-addiction", "D18:49").action == "delete"
    assert _policy_result("nurse-love-addiction", "D28:90").action == "delete"
    assert _policy_result("nurse-love-addiction", "D28:109").action == "keep"
    assert _policy_result("nurse-love-addiction", "D34:191", "Nao Sakuya Itsuki Me").action == "delete"
    assert _policy_result("nurse-love-addiction", "D34:276").action == "keep"
    assert _policy_result("nurse-love-addiction", "D34:277").action == "delete"
    assert _policy_result("nurse-love-addiction", "D34:637").action == "delete"
    assert _policy_result("nurse-love-addiction", "D34:638").action == "keep"
    assert _policy_result("nurse-love-addiction", "D38:91").action == "keep"
    assert _policy_result("nurse-love-addiction", "D38:92").action == "delete"
    assert _policy_result("nurse-love-addiction", "D38:224").action == "keep"

    choice_action = _policy_result(
        "nurse-love-addiction",
        "D18:44",
        "What an extraordinary turn of events. Accept her offer Don’t accept her offer",
    )
    assert choice_action.action == "replace_text"
    assert choice_action.text == "What an extraordinary turn of events."


def _qa(question: str, dia_id: str, utterance: str, speaker: str = "Alice") -> dict:
    return {
        "question": question,
        "option": ["A. yes", "B. no"],
        "answer": "(A)",
        "evidence_dialogues": [
            {"id": "E7", "dia_id": dia_id, "speaker": speaker, "utterance": utterance}
        ],
        "reasoning_steps": [
            {"step": 1, "inference": "supported", "based_on": ["E7"]}
        ],
    }


def test_qa_migration_repairs_only_unique_surviving_evidence() -> None:
    old = _conversation(
        _turn("D1:1", "Alice", "alpha fact"),
        _turn("D1:2", "Alice", "duplicate fact"),
        _turn("D1:3", "Alice", "duplicate fact"),
    )

    def delete_second(session: int, turn: dict) -> PolicyAction:
        if turn["dia_id"] == "D1:2":
            return PolicyAction.delete("alternate")
        return PolicyAction.keep()

    new, id_map, _ = renumber_conversation(old, delete_second)
    migrated, audit = migrate_qa_items(
        [_qa("What fact is stated?", "D1:2", "duplicate fact")],
        old,
        new,
        id_map,
        "heart-of-the-woods",
        set(),
    )
    assert len(migrated) == 1
    evidence = migrated[0]["evidence_dialogues"]
    assert evidence == [
        {"id": "E1", "dia_id": "D1:2", "speaker": "Alice", "utterance": "duplicate fact"}
    ]
    assert migrated[0]["reasoning_steps"][0]["based_on"] == ["E1"]
    assert audit["repaired"] == [
        {
            "qa_index": 1,
            "evidence_repairs": [{"old_dia_id": "D1:2", "new_dia_id": "D1:2"}],
        }
    ]

    ambiguous = copy.deepcopy(new)
    ambiguous["session_1"].append(_turn("D1:3", "Alice", "duplicate fact"))
    rejected, rejected_audit = migrate_qa_items(
        [_qa("What fact is stated?", "D1:2", "duplicate fact")],
        old,
        ambiguous,
        {"D1:1": "D1:1"},
        "heart-of-the-woods",
        set(),
    )
    assert rejected == []
    assert rejected_audit["removed"][0]["reason"] == "evidence_not_uniquely_repairable"


def test_heart_happy_ending_evidence_is_rewritten_to_sanitized_turn() -> None:
    old = _conversation(
        _turn(
            "D1:4458",
            "Narrator",
            "bad ending text\n*Madison, look! There it is!*\n*Abigail reaches out the open window to point at the ocean.*",
        )
    )
    new = _conversation(
        _turn(
            "D1:1",
            "Narrator",
            "*Madison, look! There it is!*\n*Abigail reaches out the open window to point at the ocean.*",
        )
    )
    qa = _qa(
        "When does Abigail reach out the open window to point at the ocean?",
        "D1:4458",
        old["session_1"][0]["text"],
        speaker="Narrator",
    )
    migrated, audit = migrate_qa_items(
        [qa], old, new, {"D1:4458": "D1:1"}, "heart-of-the-woods", set()
    )
    assert len(migrated) == 1
    assert migrated[0]["evidence_dialogues"][0]["dia_id"] == "D1:1"
    assert migrated[0]["evidence_dialogues"][0]["utterance"] == new["session_1"][0]["text"]
    assert audit["repaired"][0]["qa_index"] == 1


def test_qa_migration_assigns_stable_ids_and_audits_rewrites_after_evidence_repair() -> None:
    conversation = _conversation(_turn("D1:1", "Amber", "search contribution"))
    item = {
        "character": "Amber",
        "category": "test",
        "question_type": "multiple_choice",
        "question": "How do Amber and BandanaGuy respond?",
        "option": ["A. Amber argues", "B. He argues"],
        "answer": "(A,B)",
        "label": "Fact Extraction (Multiple Dialogues)",
        "evidence_dialogues": [
            {"id": "E9", "dia_id": "D1:1", "speaker": "Amber", "utterance": "search contribution"}
        ],
        "reasoning_steps": [{"step": 1, "based_on": ["E9"]}],
        "pipeline_trace": "discard me",
    }
    migrated, audit = migrate_qa_items(
        [item] * 18, conversation, conversation, {"D1:1": "D1:1"}, "highway-blossoms", set()
    )
    assert len(migrated) == 18
    rewritten = migrated[-1]
    assert rewritten["qa_id"] == "highway-blossoms-Q0018"
    assert tuple(rewritten) == (
        "qa_id", "character", "category", "question_type", "question", "option",
        "answer", "label", "evidence_dialogues", "reasoning_steps",
    )
    assert rewritten["question"].startswith(
        "How do Amber and the man wearing a bandana respond while arguing about who "
        "contributed to the search?"
    )
    assert rewritten["question"].endswith(
        "No points will be awarded for incomplete or incorrect selections."
    )
    assert audit["question_rewrite"][-1] == {
        "qa_id": "highway-blossoms-Q0018", "action": "rewrite"
    }
    assert validate_public_qa(rewritten) == []

    QUESTION_REWRITES["fixture-Q0002"] = None
    try:
        deleted, deleted_audit = migrate_qa_items(
            [item, item], conversation, conversation, {"D1:1": "D1:1"}, "fixture", set()
        )
    finally:
        del QUESTION_REWRITES["fixture-Q0002"]
    assert len(deleted) == 1
    assert deleted_audit["removed"][-1] == {
        "qa_index": 2,
        "qa_id": "fixture-Q0002",
        "reason": "unsafe_question_rewrite",
    }


def test_public_qa_schema_strips_pipeline_traces() -> None:
    item = _qa("What fact is stated?", "D1:1", "alpha fact")
    item["pollution_check"] = {"response": "leaked model response"}
    item["iterative_evidence_ablation"] = [{"response": "Option_1: leaked"}]
    sanitized = sanitize_qa_item(item)
    assert "pollution_check" not in sanitized
    assert "iterative_evidence_ablation" not in sanitized
    assert sanitized["question"] == item["question"]
    assert sanitized["evidence_dialogues"] == item["evidence_dialogues"]


def test_question_wording_is_self_contained() -> None:
    assert sanitize_question_wording(
        "Arrange the cited moments in the episode in chronological order."
    ) == "Arrange the cited moments in the episode in chronological order."
    assert sanitize_question_wording(
        "What does the combination of the cited evidence establish?"
    ) == "What does the combination of the cited evidence establish?"
    assert sanitize_question_wording(
        "Which claims accurately reflect the cited episode?"
    ) == "Which claims accurately reflect the cited episode?"


def test_qa_migration_rejects_contaminated_and_dangling_questions() -> None:
    conversation = _conversation(_turn("D1:1", "Alice", "alpha fact"))
    qa_items = [
        _qa("What fact is stated?", "D1:1", "alpha fact"),
        _qa("Which factual interpretation is supported by this cited moment?", "D1:1", "alpha fact"),
    ]
    migrated, audit = migrate_qa_items(
        qa_items,
        conversation,
        conversation,
        {"D1:1": "D1:1"},
        "arknights",
        {1},
    )
    assert migrated == []
    assert audit["removed"] == [
        {
            "qa_index": 1,
            "qa_id": "arknights-Q0001",
            "reason": "contaminated_option_session",
        },
        {
            "qa_index": 2,
            "qa_id": "arknights-Q0002",
            "reason": "dangling_reference_question",
        },
    ]


def test_validation_rejects_broken_references_markers_and_noise() -> None:
    good_record = {
        "conversation": _conversation(_turn("D1:1", "Alice", "alpha fact")),
        "qa": [_qa("What fact is stated?", "D1:1", "alpha fact")],
    }
    good_record["qa"][0]["evidence_dialogues"][0]["id"] = "E1"
    good_record["qa"][0]["reasoning_steps"][0]["based_on"] = ["E1"]
    good_record["qa"][0], _audit = normalize_public_qa(
        good_record["qa"][0], "arknights", 1
    )
    assert validate_curated_record(good_record, "arknights") == []

    non_contiguous = copy.deepcopy(good_record)
    non_contiguous["conversation"]["session_1"][0]["dia_id"] = "D1:2"
    assert "session_1 has non-contiguous dia_id D1:2 at position 1" in validate_curated_record(
        non_contiguous, "heart-of-the-woods"
    )

    missing_reference = copy.deepcopy(good_record)
    missing_reference["qa"][0]["evidence_dialogues"][0]["dia_id"] = "D1:9"
    assert "qa 1 evidence E1 references missing D1:9" in validate_curated_record(
        missing_reference, "heart-of-the-woods"
    )

    option_marker = copy.deepcopy(good_record)
    option_marker["conversation"]["session_1"][0]["text"] = "Option_1: choose"
    assert "forbidden interactive option marker at D1:1" in validate_curated_record(
        option_marker, "arknights"
    )

    noisy = copy.deepcopy(good_record)
    noisy["conversation"]["session_1"][0]["text"] = "A" * 2500
    assert "extreme noise at D1:1" in validate_curated_record(noisy, "heart-of-the-woods")

    dangling = copy.deepcopy(good_record)
    dangling["qa"][0]["question"] = "What is supported by this cited moment?"
    assert "qa 1 contains a dangling cited-moment reference" in validate_curated_record(
        dangling, "heart-of-the-woods"
    )


def test_public_schema_normalizes_ids_types_options_answers_and_rendering() -> None:
    item = {
        "character": "Alice",
        "category": 4,
        "question_type": "Ordering",
        "question": (
            "Please answer the question: Arrange the events in chronological order.\n"
            "a) First event\n"
            "b. Second event\n"
            "c. Third event\n"
            "d. Fourth event\n"
            "stale instruction"
        ),
        "option": ["a) First event", "b. Second event", "c. Third event", "d. Fourth event"],
        "answer": "(B, A, D, C)",
        "label": "Memory Update",
        "evidence_dialogues": [],
        "reasoning_steps": [],
        "internal_trace": "remove me",
    }
    normalized, audit = normalize_public_qa(item, "nurse-love-addiction", 7)
    assert make_qa_id("nurse-love-addiction", 7) == "nurse-love-addiction-Q0007"
    assert list(normalized) == [
        "qa_id",
        "character",
        "category",
        "question_type",
        "question",
        "option",
        "answer",
        "label",
        "evidence_dialogues",
        "reasoning_steps",
    ]
    assert normalized["qa_id"] == "nurse-love-addiction-Q0007"
    assert normalized["question_type"] == "ordering"
    assert normalized["option"] == [
        "A. First event",
        "B. Second event",
        "C. Third event",
        "D. Fourth event",
    ]
    assert normalized["answer"] == "(B,A,D,C)"
    assert normalized["question"].endswith(ORDERING_INSTRUCTION)
    assert normalized["question"].split("\n", 1)[0] == "Arrange the events in chronological order."
    assert "stale instruction" not in normalized["question"]
    assert audit["qa_id"] == "nurse-love-addiction-Q0007"
    assert validate_public_qa(normalized) == []


def test_public_schema_rejects_invalid_answer_and_option_shape() -> None:
    item = {
        "question": "Choose one.",
        "option": ["A. Alpha", "C. Charlie"],
        "answer": "(A,A)",
        "label": "Fact Extraction (Single Dialogue)",
    }
    try:
        normalize_public_qa(item, "fixture", 1)
    except ValueError as error:
        assert "consecutive" in str(error) or "repeated" in str(error)
    else:
        raise AssertionError("invalid option/answer shape must be rejected")


def test_public_schema_permutates_abstain_option_and_nested_references() -> None:
    item = {
        "character": "Alice",
        "category": 4,
        "question_type": "single_choice",
        "question": "Which conclusion follows?",
        "option": [
            "A. First story statement",
            "B. Second story statement",
            "C. Third story statement",
            "D. Cannot determine from the information provided.",
            "E. Fifth story statement",
            "F. Sixth story statement",
        ],
        "answer": "(D)",
        "label": "Abstain",
        "evidence_dialogues": [
            {
                "id": "E1",
                "option": {"primary": "D", "alternatives": [{"option": "F"}]},
                "nested": {"option": "F"},
            }
        ],
        "reasoning_steps": [
            {"based_on": ["E1"], "details": {"option": [{"selected": "D"}, {"option": "F"}]}},
        ],
    }
    normalized, audit = normalize_public_qa(item, "fixture", 9)
    assert normalized["option"][3].startswith("D. Sixth")
    assert normalized["option"][5].startswith("F. Cannot determine")
    assert normalized["answer"] == "(F)"
    assert normalized["evidence_dialogues"][0]["option"]["primary"] == "F"
    assert normalized["evidence_dialogues"][0]["option"]["alternatives"][0]["option"] == "D"
    assert normalized["evidence_dialogues"][0]["nested"]["option"] == "D"
    assert normalized["reasoning_steps"][0]["details"]["option"][0]["selected"] == "F"
    assert normalized["reasoning_steps"][0]["details"]["option"][1]["option"] == "D"
    assert audit["abstain_permuted"] is True
    assert validate_public_qa(normalized) == []


def test_public_schema_rejects_colon_option_labels() -> None:
    item = {
        "question": "Choose one.",
        "option": ["A: Alpha", "B：Beta"],
        "answer": "A",
    }
    try:
        normalize_public_qa(item, "fixture", 11)
    except ValueError as error:
        assert "label" in str(error)
    else:
        raise AssertionError("colon option labels must not enter the public schema")


def test_public_schema_canonicalizes_multiple_choice_answer_order() -> None:
    item = {
        "character": "Alice",
        "category": 4,
        "question_type": "multiple_choice",
        "question": "Which statements are true?",
        "option": ["A. Alpha", "B. Beta", "C. Charlie", "D. Delta"],
        "answer": "(D,A)",
        "label": "Fact Extraction (Multiple Dialogues)",
        "evidence_dialogues": [],
        "reasoning_steps": [],
    }
    normalized, _audit = normalize_public_qa(item, "fixture", 10)
    assert normalized["answer"] == "(A,D)"
    assert validate_public_qa(normalized) == []


def test_question_rewrites_are_story_specific() -> None:
    cases = {
        "nurse-love-addiction-Q0001": (
            "Which event involving Asuka belongs to the interval bounded by the two "
            "boundary events? Earlier boundary: The narration establishes that the "
            "teacher's tone is friendly, but the words resound within the classroom. "
            "The narration establishes that within her. Later boundary: Itsuki "
            "apologizes, and states she must leave. Itsuki has got to split for a while.",
            "After the teacher's words resound through the classroom but before Itsuki "
            "says she has to leave for a while, what does Asuka do?",
        ),
        "fault-milestone-two-Q0028": (
            "Which long-range conclusions about Sol's relationship with the group are "
            "supported? Select all that apply?",
            "How does Sol's relationship with Selphine's group change over the course "
            "of the story? Select all that apply?",
        ),
        "highway-blossoms-Q0013": (
            "Place these moments in the “the music-festival plan” thread in "
            "chronological order.",
            "In what order do Amber and Marina discuss their plans to attend the music "
            "festival?",
        ),
        "nurse-love-addiction-Q0108": (
            "Identify the Itsuki statements that co-occur in the episode anchored by "
            "Itsuki observes that tell her the truth and she will get really mad at her.",
            "Which of Itsuki's other statements occur in the scene where she tells "
            "Asuka, ‘Tell me the truth and I’ll get really mad at you’?",
        ),
        "fault-milestone-two-Q0011": (
            "Which developments involving It's all right accurately describe the "
            "characters's episode?",
            "What happens as Selphine returns to herself after the confrontation? "
            "Select all that apply?",
        ),
        "nurse-love-addiction-Q0066": (
            "What is Asuka’s nearby response to the scene anchored by Asuka observes "
            "that come on, Asuka. Asuka observes that stop talking gibberish and answer "
            "the question?",
            "What does Asuka say after Sakuya tells her to stop talking gibberish and "
            "answer the question?",
        ),
    }
    for qa_id, (source_stem, expected_stem) in cases.items():
        rewritten, action = rewrite_question_stem(qa_id, source_stem)
        assert rewritten == expected_stem
        assert action == "rewrite"
        assert find_construction_issues(rewritten) == []

    rewritten, action = rewrite_question_stem(
        "highway-blossoms-Q0018",
        "Which developments occur in the common-route relationship thread?",
    )
    assert rewritten == (
        "How do Amber and the man wearing a bandana respond while arguing about who "
        "contributed to the search?"
    )
    assert action == "rewrite"

    private_thought, action = rewrite_question_stem(
        "highway-blossoms-Q0005",
        "At this point in the story, does the dialogue establish that Mariah has "
        "been explicitly told Amber's private thought?",
    )
    assert private_thought == (
        "By this point in the story, has Amber explicitly told Mariah that she sees "
        "her as a reckless person she can vent to but not trust?"
    )
    assert action == "rewrite"
    assert "ever" not in private_thought.casefold()


def test_construction_issue_detection_distinguishes_framing_from_plot_evidence() -> None:
    assert find_construction_issues(
        "Which event occurs between the earlier boundary and later boundary?"
    ) == ["boundary framing"]
    assert find_construction_issues(
        "Which long-range conclusions about Sol are supported?"
    ) == ["long-range framing"]
    assert find_construction_issues(
        "Place these moments in the music-festival thread in chronological order."
    ) == ["annotation thread"]
    assert find_construction_issues(
        "What color is the literal thread on Marina's jacket?"
    ) == []
    assert find_construction_issues(
        "Which events occur in the episode involving fault, sorry, and allow?"
    ) == ["keyword episode framing"]
    assert find_construction_issues(
        "What does the provided dialogue establish?"
    ) == ["source-material framing"]
    assert find_construction_issues(
        "Which conclusion follows from the supplied evidence?"
    ) == ["source-material framing"]
    assert find_construction_issues(
        "What changes when the separated evidence is connected?"
    ) == ["source-material framing"]
    assert find_construction_issues(
        "Which statements describe the characters's episode?"
    ) == ["malformed possessive"]
    assert find_construction_issues(
        "What follows after Asuka observes that she feels ill?"
    ) == ["mechanical clause"]
    assert find_construction_issues(
        "What conclusive evidence does Sceatoire provide that Rune killed her?"
    ) == []
    assert find_construction_issues(
        "Where do Amber and Marina find evidence that the treasure is real?"
    ) == []
    assert find_construction_issues(
        "Arrange the milestones in Sakuya and Itsuki's common-route relationship."
    ) == ["route-label framing"]
    assert find_construction_issues(
        "Arrange common moments involving Asuka across sessions 13–30."
    ) == ["long-range framing"]
    assert find_construction_issues(
        "What does Tara do later in the same local sequence?"
    ) == ["local-anchor framing"]
    assert find_construction_issues(
        "Order the events in Tara's local progression."
    ) == ["local-anchor framing"]


def test_question_rewrite_registry_is_complete_and_safe() -> None:
    assert len(QUESTION_REWRITES) == len(set(QUESTION_REWRITES))
    for qa_id, rewritten in QUESTION_REWRITES.items():
        assert qa_id == make_qa_id(qa_id.rsplit("-Q", 1)[0], int(qa_id[-4:]))
        if rewritten is None:
            continue
        assert rewritten.strip() == rewritten
        assert rewritten
        assert rewritten.endswith("?")
        assert find_construction_issues(rewritten) == []


def main() -> None:
    test_generic_filter_renumbers_and_preserves_metadata()
    test_title_policies_keep_only_canonical_content()
    test_qa_migration_repairs_only_unique_surviving_evidence()
    test_heart_happy_ending_evidence_is_rewritten_to_sanitized_turn()
    test_qa_migration_assigns_stable_ids_and_audits_rewrites_after_evidence_repair()
    test_public_qa_schema_strips_pipeline_traces()
    test_question_wording_is_self_contained()
    test_qa_migration_rejects_contaminated_and_dangling_questions()
    test_validation_rejects_broken_references_markers_and_noise()
    test_public_schema_normalizes_ids_types_options_answers_and_rendering()
    test_public_schema_rejects_invalid_answer_and_option_shape()
    test_public_schema_permutates_abstain_option_and_nested_references()
    test_public_schema_canonicalizes_multiple_choice_answer_order()
    test_public_schema_rejects_colon_option_labels()
    test_question_rewrites_are_story_specific()
    test_construction_issue_detection_distinguishes_framing_from_plot_evidence()
    test_question_rewrite_registry_is_complete_and_safe()
    print("curation regression checks passed")


if __name__ == "__main__":
    main()
