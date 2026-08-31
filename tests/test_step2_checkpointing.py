import copy
import logging
import sys
import tempfile
import threading
import time
import types
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch


if "dotenv" not in sys.modules:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda: None
    sys.modules["dotenv"] = dotenv_stub

if "jinja2" not in sys.modules:
    jinja_stub = types.ModuleType("jinja2")

    class _Template:
        def __init__(self, text):
            self.text = text

        def render(self, values):
            rendered = self.text
            for key, value in values.items():
                rendered = rendered.replace("{{" + key + "}}", str(value))
            return rendered

    jinja_stub.Template = _Template
    sys.modules["jinja2"] = jinja_stub

if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")
    openai_stub.OpenAI = object
    sys.modules["openai"] = openai_stub

if "tqdm" not in sys.modules:
    tqdm_stub = types.ModuleType("tqdm")

    class _Progress:
        def __init__(self, *args, **kwargs):
            self.n = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def update(self, amount):
            self.n += amount

    tqdm_stub.tqdm = _Progress
    sys.modules["tqdm"] = tqdm_stub

from src.step2_evidence_check import (
    EvidenceProcessingError,
    FullContextManager,
    evidence_check_main,
)
from src.ablation_chunking import AblationChunk
from src.question_formatting import strip_answer_instruction_suffix
from src.utils import load_json_file, write_json_file


def make_record(*questions):
    return {
        "conversation": {},
        "qa": [
            {
                "id": question_id,
                "question_type": "single_choice",
                "question": f"Question {question_id}?\n(A) one\n(B) two",
                "option": ["A. one", "B. two"],
                "answer": "(A)",
            }
            for question_id in questions
        ],
    }


class ControlledManager(FullContextManager):
    def __init__(self, output_path, delays=None, failures=None):
        self.output_path = str(output_path)
        self.logger = logging.getLogger("step2-checkpoint-test")
        self.lock = threading.Lock()
        self.results = []
        self.original_data = []
        self.delays = delays or {}
        self.failures = set(failures or [])
        self.calls = []

    def _get_aligned_dialogue_evidence(self, question_item, conversation):
        return [], {"result": "pass"}

    def _process_single_question(
        self,
        conversation_item,
        question_item,
        record_idx,
        qa_idx,
        pbar,
        only_evidence,
        except_evidence,
    ):
        question_id = question_item["id"]
        self.calls.append(question_id)
        time.sleep(self.delays.get(question_id, 0))
        if question_id in self.failures:
            raise RuntimeError(f"failed {question_id}")
        result = copy.deepcopy(question_item)
        if only_evidence:
            result["only_evidence_check"] = {"result": "right"}
        else:
            result["iterative_evidence_ablation_summary"] = {
                "result": "passed",
                "passed": True,
            }
        return result


class Step2CheckpointTests(unittest.TestCase):
    def test_ordering_reducer_fills_sequence_positions_from_final_answer(self):
        manager = object.__new__(FullContextManager)
        manager.model_name = "test-model"
        manager.ablation_config = {
            "context_limit": 32768,
            "prompt_safety_tokens": 4096,
        }
        manager._call_llm = lambda prompt, max_retries=3: (
            '{"answer":"(B,A,D,C)","evidence_dialogues":['
            '{"option":"B","dia_id":"D1:1","speaker":"Narrator","utterance":"first"},'
            '{"option":"A","dia_id":"D1:2","speaker":"Miyako","utterance":"second"},'
            '{"option":"D","dia_id":"D1:3","speaker":"Miyako","utterance":"third"},'
            '{"option":"C","dia_id":"D1:4","speaker":"Narrator","utterance":"fourth"}'
            "]}",
            0.01,
            0,
        )
        conversation = {
            "session_1": [
                {"dia_id": "D1:1", "speaker": "Narrator", "utterance": "first"},
                {"dia_id": "D1:2", "speaker": "Miyako", "utterance": "second"},
                {"dia_id": "D1:3", "speaker": "Miyako", "utterance": "third"},
                {"dia_id": "D1:4", "speaker": "Narrator", "utterance": "fourth"},
            ]
        }
        candidates = [
            {
                "option": option,
                "support_type": "full",
                "evidence_dialogues": [
                    {
                        "option": option,
                        "dia_id": f"D1:{index}",
                        "speaker": speaker,
                        "utterance": utterance,
                    }
                ],
            }
            for index, (option, speaker, utterance) in enumerate(
                [
                    ("B", "Narrator", "first"),
                    ("A", "Miyako", "second"),
                    ("D", "Miyako", "third"),
                    ("C", "Narrator", "fourth"),
                ],
                start=1,
            )
        ]

        result, _, diagnostics = manager._reduce_ablation_candidates(
            "Put the events in order.",
            candidates,
            conversation,
            "ordering",
        )

        self.assertEqual("(B,A,D,C)", result["answer"])
        self.assertNotIn("error", diagnostics)
        self.assertEqual(
            {"B": 1, "A": 2, "D": 3, "C": 4},
            {
                evidence["option"]: evidence["sequence_position"]
                for evidence in result["evidence_dialogues"]
            },
        )

    def test_ordering_reducer_still_rejects_missing_option_evidence(self):
        manager = object.__new__(FullContextManager)
        manager.model_name = "test-model"
        manager.ablation_config = {
            "context_limit": 32768,
            "prompt_safety_tokens": 4096,
        }
        manager._call_llm = lambda prompt, max_retries=3: (
            '{"answer":"(B,A,D,C)","evidence_dialogues":['
            '{"option":"B","dia_id":"D1:1","speaker":"Narrator","utterance":"first"},'
            '{"option":"A","dia_id":"D1:2","speaker":"Miyako","utterance":"second"},'
            '{"option":"D","dia_id":"D1:3","speaker":"Miyako","utterance":"third"}'
            "]}",
            0.01,
            0,
        )
        conversation = {
            "session_1": [
                {"dia_id": f"D1:{index}", "speaker": speaker, "utterance": utterance}
                for index, (speaker, utterance) in enumerate(
                    [
                        ("Narrator", "first"),
                        ("Miyako", "second"),
                        ("Miyako", "third"),
                        ("Narrator", "fourth"),
                    ],
                    start=1,
                )
            ]
        }
        candidates = [
            {
                "option": option,
                "support_type": "full",
                "evidence_dialogues": [
                    {
                        "option": option,
                        "dia_id": f"D1:{index}",
                        "speaker": speaker,
                        "utterance": utterance,
                    }
                ],
            }
            for index, (option, speaker, utterance) in enumerate(
                [
                    ("B", "Narrator", "first"),
                    ("A", "Miyako", "second"),
                    ("D", "Miyako", "third"),
                    ("C", "Narrator", "fourth"),
                ],
                start=1,
            )
        ]

        result, _, diagnostics = manager._reduce_ablation_candidates(
            "Put the events in order.",
            candidates,
            conversation,
            "ordering",
        )

        self.assertEqual({}, result)
        self.assertEqual(
            "incomplete_answer_evidence_coverage",
            diagnostics["error"],
        )

    def test_out_of_order_workers_preserve_source_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.json"
            output = Path(temp_dir) / "output.json"
            write_json_file([make_record("slow", "fast")], source)
            manager = ControlledManager(output, delays={"slow": 0.05, "fast": 0.0})

            manager.process_data_file(
                str(source), only_evidence=1, except_evidence=0, max_workers=2,
                checkpoint_every_questions=1,
            )

            questions = load_json_file(output)[0]["qa"]
            self.assertEqual(["slow", "fast"], [question["id"] for question in questions])
            self.assertTrue(all("only_evidence_check" in question for question in questions))

    def test_resume_skips_completed_v2a_question(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.json"
            output = Path(temp_dir) / "output.json"
            records = [make_record("done", "pending")]
            snapshot = copy.deepcopy(records)
            snapshot[0]["qa"][0]["only_evidence_check"] = {"result": "maybe_wrong"}
            write_json_file(records, source)
            write_json_file(snapshot, output)
            manager = ControlledManager(output)

            manager.process_data_file(
                str(source), only_evidence=1, except_evidence=0, max_workers=1,
                checkpoint_every_questions=1,
            )

            self.assertEqual(["pending"], manager.calls)
            questions = load_json_file(output)[0]["qa"]
            self.assertEqual("maybe_wrong", questions[0]["only_evidence_check"]["result"])
            self.assertEqual("right", questions[1]["only_evidence_check"]["result"])

    def test_failed_worker_leaves_question_pending_in_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.json"
            output = Path(temp_dir) / "output.json"
            write_json_file([make_record("ok", "retry")], source)
            manager = ControlledManager(output, failures={"retry"})

            manager.process_data_file(
                str(source), only_evidence=1, except_evidence=0, max_workers=2,
                checkpoint_every_questions=1,
            )

            questions = load_json_file(output)[0]["qa"]
            self.assertIn("only_evidence_check", questions[0])
            self.assertNotIn("only_evidence_check", questions[1])

    def test_v2b_skips_non_right_v2a_and_resumes_missing_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "v2a.json"
            output = Path(temp_dir) / "v2b.json"
            records = [make_record("filtered", "ablate")]
            records[0]["qa"][0]["only_evidence_check"] = {"result": "maybe_wrong"}
            records[0]["qa"][1]["only_evidence_check"] = {"result": "right"}
            write_json_file(records, source)
            manager = ControlledManager(output)

            manager.process_data_file(
                str(source), only_evidence=0, except_evidence=1, max_workers=1,
                checkpoint_every_questions=1,
            )

            self.assertEqual(["ablate"], manager.calls)
            questions = load_json_file(output)[0]["qa"]
            self.assertNotIn("iterative_evidence_ablation_summary", questions[0])
            self.assertEqual("passed", questions[1]["iterative_evidence_ablation_summary"]["result"])

    def test_resume_rejects_reordered_questions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.json"
            output = Path(temp_dir) / "output.json"
            records = [make_record("first", "second")]
            snapshot = [make_record("second", "first")]
            write_json_file(records, source)
            write_json_file(snapshot, output)

            with self.assertRaisesRegex(ValueError, "qa=0"):
                ControlledManager(output).process_data_file(
                    str(source), only_evidence=1, except_evidence=0,
                    checkpoint_every_questions=1,
                )

    def test_failed_run_resumes_only_pending_question_and_matches_full_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.json"
            resumed_output = Path(temp_dir) / "resumed.json"
            uninterrupted_output = Path(temp_dir) / "uninterrupted.json"
            write_json_file([make_record("first", "second")], source)

            interrupted = ControlledManager(resumed_output, failures={"second"})
            interrupted.process_data_file(
                str(source), only_evidence=1, except_evidence=0, max_workers=2,
                checkpoint_every_questions=1,
            )
            resumed = ControlledManager(resumed_output)
            resumed.process_data_file(
                str(source), only_evidence=1, except_evidence=0, max_workers=2,
                checkpoint_every_questions=1,
            )
            uninterrupted = ControlledManager(uninterrupted_output)
            uninterrupted.process_data_file(
                str(source), only_evidence=1, except_evidence=0, max_workers=2,
                checkpoint_every_questions=1,
            )

            self.assertEqual(["second"], resumed.calls)
            self.assertEqual(load_json_file(uninterrupted_output), load_json_file(resumed_output))


class OperationalFailureTests(unittest.TestCase):
    def test_answer_instruction_stripping_handles_all_types_without_truncating_options(self):
        cases = [
            (
                "Please provide the option corresponding to the only correct answer, "
                "enclosed in parentheses, e.g., (X).",
                "A. An option says Please provide all correct options enclosed in parentheses.",
            ),
            (
                "Please provide all correct options enclosed in parentheses, separated "
                "by commas, e.g., (X, Y, Z). No points will be awarded for incomplete "
                "or incorrect selections.",
                "A. First option",
            ),
            (
                "Please provide the options in the correct order enclosed in parentheses, "
                "separated by commas, e.g., (Y, Z, X, W). No points will be awarded for "
                "an incorrect sequence.",
                "A. First event",
            ),
        ]

        for instruction, option in cases:
            formatted = f"Question?\n{option}\nB. Second option\n{instruction}"
            self.assertEqual(
                f"Question?\n{option}\nB. Second option",
                strip_answer_instruction_suffix(formatted),
            )

    def test_answer_instruction_stripping_ignores_non_suffix_marker_line(self):
        question = (
            "Question?\n"
            "A. First option\n"
            "Please provide all correct options enclosed in parentheses during training, "
            "but this sentence is part of the option.\n"
            "B. Second option"
        )

        self.assertEqual(question, strip_answer_instruction_suffix(question))

    def test_internal_ablation_prompts_replace_user_answer_suffix_with_json_requirement(self):
        manager = object.__new__(FullContextManager)
        manager._format_conversation = lambda conversation: "dialogue"
        manager.model_name = "test-model"
        manager.ablation_config = {
            "context_limit": 32768,
            "prompt_safety_tokens": 4096,
        }
        question = (
            "Which statement is supported?\n"
            "A. First option\n"
            "B. Second option\n"
            "F. Cannot infer the answer based on the given information.\n"
            "Please provide the option corresponding to the only correct answer, "
            "enclosed in parentheses, e.g., (X)."
        )
        chunk = AblationChunk(
            chunk_id="chunk-1",
            conversation={"session_1": []},
            dia_ids=[],
            start_order=0,
            end_order=0,
            estimated_tokens=1,
        )
        candidate = {
            "option": "A",
            "support_type": "full",
            "evidence_dialogues": [
                {
                    "option": "A",
                    "dia_id": "D1:1",
                    "speaker": "Narrator",
                    "utterance": "evidence",
                }
            ],
        }

        chunk_prompt = manager._build_ablation_chunk_scan_prompt(chunk, question)
        reducer_prompt = manager._build_ablation_reducer_prompt(
            question,
            [candidate],
            "single_choice",
        )
        captured = {}
        manager._call_llm = lambda prompt, max_retries=3: (
            captured.setdefault("selector_prompt", prompt)
            and '{"candidate_indexes":[0]}',
            0.01,
            0,
        )
        manager._select_relevant_ablation_candidates(question, [candidate])

        for prompt in (
            chunk_prompt,
            reducer_prompt,
            captured["selector_prompt"],
        ):
            self.assertIn("A. First option", prompt)
            self.assertNotIn(
                "Please provide the option corresponding to the only correct answer",
                prompt,
            )
            self.assertGreater(
                prompt.rfind("FINAL OUTPUT REQUIREMENT:"),
                prompt.rfind("QUESTION:"),
            )

    def test_reducer_json_schema_shows_the_answer_format_for_each_question_type(self):
        manager = object.__new__(FullContextManager)
        candidate = {
            "option": "A",
            "support_type": "full",
            "evidence_dialogues": [
                {
                    "option": "A",
                    "dia_id": "D1:1",
                    "speaker": "Narrator",
                    "utterance": "evidence",
                }
            ],
        }
        cases = {
            "single_choice": '"answer": "(A)"',
            "multiple_choice": '"answer": "(A,C)"',
            "ordering": '"answer": "(B,A,D,C)"',
        }

        for question_type, expected_schema in cases.items():
            prompt = manager._build_ablation_reducer_prompt(
                "Question?\nA. First option\nB. Second option",
                [candidate],
                question_type,
            )
            self.assertIn(expected_schema, prompt)

    def test_reducer_accepts_only_standalone_bare_f_as_safe_fallback(self):
        manager = object.__new__(FullContextManager)
        manager.model_name = "test-model"
        manager.ablation_config = {
            "context_limit": 32768,
            "prompt_safety_tokens": 4096,
        }
        manager._call_llm = lambda prompt, max_retries=3: ("(F)", 0.01, 0)
        conversation = {
            "session_1": [
                {
                    "dia_id": "D1:1",
                    "speaker": "Narrator",
                    "utterance": "candidate evidence",
                }
            ]
        }
        candidates = [
            {
                "option": "A",
                "support_type": "partial",
                "evidence_dialogues": [
                    {
                        "option": "A",
                        "dia_id": "D1:1",
                        "speaker": "Narrator",
                        "utterance": "candidate evidence",
                    }
                ],
            }
        ]

        result, raw_response, diagnostics = manager._reduce_ablation_candidates(
            "Question?",
            candidates,
            conversation,
            "single_choice",
        )

        self.assertEqual("(F)", raw_response)
        self.assertEqual(
            {"answer": "(F)", "evidence_dialogues": []},
            result,
        )
        self.assertEqual("standalone_f_fallback", diagnostics["parse_status"])
        self.assertNotIn("error", diagnostics)

    def test_reducer_rejects_bare_non_f_answer(self):
        manager = object.__new__(FullContextManager)
        manager.model_name = "test-model"
        manager.ablation_config = {
            "context_limit": 32768,
            "prompt_safety_tokens": 4096,
        }
        manager._call_llm = lambda prompt, max_retries=3: ("(A)", 0.01, 0)
        conversation = {
            "session_1": [
                {
                    "dia_id": "D1:1",
                    "speaker": "Narrator",
                    "utterance": "candidate evidence",
                }
            ]
        }
        candidates = [
            {
                "option": "A",
                "support_type": "full",
                "evidence_dialogues": [
                    {
                        "option": "A",
                        "dia_id": "D1:1",
                        "speaker": "Narrator",
                        "utterance": "candidate evidence",
                    }
                ],
            }
        ]

        result, _, diagnostics = manager._reduce_ablation_candidates(
            "Question?",
            candidates,
            conversation,
            "single_choice",
        )

        self.assertEqual({}, result)
        self.assertEqual("invalid_reducer_response", diagnostics["error"])

    def test_partial_retrieval_answer_is_not_accepted_without_full_coverage(self):
        manager = object.__new__(FullContextManager)
        manager.model_name = "test-model"
        manager.ablation_config = {
            "context_limit": 1,
            "prompt_safety_tokens": 0,
            "chunk_tokens": 100,
            "retrieval_chunks": 2,
            "chunk_max_workers": 1,
        }
        manager._build_prompt = lambda **kwargs: "force chunk mode"
        manager._format_conversation = lambda conversation: "remaining context"
        manager._conversation_without_evidence = lambda conversation, evidence: conversation
        chunks = [
            AblationChunk(
                chunk_id=f"chunk-{index}",
                conversation={
                    "session_1": [
                        {
                            "dia_id": f"D1:{index}",
                            "speaker": "Narrator",
                            "utterance": f"turn {index}",
                        }
                    ]
                },
                dia_ids=[f"D1:{index}"],
                start_order=index,
                end_order=index,
                estimated_tokens=1,
            )
            for index in (1, 2)
        ]
        scan_batches = []

        def scan_chunks(requested_chunks, question):
            scan_batches.append([chunk.chunk_id for chunk in requested_chunks])
            return [
                {
                    "status": "ok" if chunk.chunk_id == "chunk-1" else "failed",
                    "chunk_id": chunk.chunk_id,
                    "candidate_evidence": (
                        [{"option": "A", "evidence_dialogues": []}]
                        if chunk.chunk_id == "chunk-1"
                        else []
                    ),
                    "attempts": 3,
                    "response": "(A)\nignored trailing text",
                    "error": "" if chunk.chunk_id == "chunk-1" else "invalid_json",
                }
                for chunk in requested_chunks
            ]

        manager._scan_ablation_chunks = scan_chunks
        manager._reduce_ablation_candidates = (
            lambda question, candidates, conversation, question_type: (
                {
                    "answer": "(A)",
                    "evidence_dialogues": [
                        {
                            "option": "A",
                            "dia_id": "D1:1",
                            "speaker": "Narrator",
                            "utterance": "turn 1",
                        }
                    ],
                },
                "",
                {},
            )
        )

        with patch(
            "src.step2_evidence_check.chunk_conversation",
            return_value=chunks,
        ), patch(
            "src.step2_evidence_check.rank_chunks",
            return_value=chunks,
        ):
            data, _, audit = manager._answer_after_ablation(
                conversation_item={"session_1": []},
                question="Question?",
                answer_candidates=["A"],
                cumulative_removed_evidence=[],
                question_type="single_choice",
                chunk_result_cache={},
            )

        self.assertEqual({}, data)
        self.assertTrue(audit["exhaustive_scan_used"])
        self.assertFalse(audit["coverage_complete"])
        self.assertEqual("chunk-2", audit["failed_chunks"][0]["chunk_id"])
        self.assertEqual(
            "(A) ignored trailing text",
            audit["failed_chunks"][0].get("response_preview"),
        )
        self.assertEqual([["chunk-1", "chunk-2"], ["chunk-2"]], scan_batches)

    def test_serial_chunk_scan_converts_scanner_exception_to_failed_result(self):
        manager = object.__new__(FullContextManager)
        manager.ablation_config = {"chunk_max_workers": 1}
        manager._scan_ablation_chunk_with_split = (
            lambda chunk, question: (_ for _ in ()).throw(RuntimeError("scanner failed"))
        )
        chunk = AblationChunk(
            chunk_id="chunk-1",
            conversation={"session_1": []},
            dia_ids=["D1:1"],
            start_order=1,
            end_order=1,
            estimated_tokens=1,
        )

        try:
            results = manager._scan_ablation_chunks([chunk], "Question?")
        except RuntimeError as exc:
            self.fail(f"serial scanner exception escaped without diagnostics: {exc}")

        self.assertEqual("failed", results[0]["status"])
        self.assertEqual("chunk-1", results[0]["chunk_id"])
        self.assertIn("RuntimeError: scanner failed", results[0]["error"])

    def test_maximum_context_error_returns_flag_for_chunk_fallback(self):
        manager = object.__new__(FullContextManager)
        manager.model_name = "test-model"
        manager.logger = logging.getLogger("step2-context-fallback-test")

        class ContextExceededCompletions:
            def create(self, **kwargs):
                raise RuntimeError("maximum context length exceeded")

        manager.openai_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=ContextExceededCompletions()),
            responses=types.SimpleNamespace(),
        )

        try:
            response, _, context_flag = manager._call_llm("prompt", max_retries=1)
        except EvidenceProcessingError as exc:
            self.fail(f"context overflow bypassed chunk fallback: {exc}")

        self.assertEqual(1, context_flag)
        self.assertIn("maximum context length exceeded", response)

    def test_recovered_coverage_failure_is_not_reported_as_terminal_reason(self):
        manager = object.__new__(FullContextManager)
        manager._conversation_without_evidence = lambda conversation, evidence: conversation
        attempts = 0

        def answer_after_ablation(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return {}, "", {
                    "coverage_complete": False,
                    "failed_chunks": [
                        {"chunk_id": "chunk-2", "error": "invalid_json"}
                    ],
                }
            return {}, "", {"coverage_complete": True}

        manager._answer_after_ablation = answer_after_ablation

        records = manager._run_iterative_ablation(
            conversation_item={"session_1": []},
            answer="(A)",
            question="Question?",
            base_evidence_blocks=[],
            question_type="single_choice",
        )

        self.assertEqual(5, attempts)
        self.assertEqual(
            "evidence_extraction_failed_after_retries",
            records[-1]["stop_reason"],
        )
        self.assertEqual([], records[-1]["failed_chunks"])

    def test_ablation_retries_only_failed_chunks_within_the_round(self):
        manager = object.__new__(FullContextManager)
        manager.model_name = "test-model"
        manager.ablation_config = {
            "context_limit": 1,
            "prompt_safety_tokens": 0,
            "chunk_tokens": 100,
            "retrieval_chunks": 2,
            "chunk_max_workers": 1,
        }
        manager._build_prompt = lambda **kwargs: "force chunk mode"
        manager._format_conversation = lambda conversation: "remaining context"
        manager._conversation_without_evidence = lambda conversation, evidence: conversation
        chunks = [
            AblationChunk(
                chunk_id=f"chunk-{index}",
                conversation={
                    "session_1": [
                        {
                            "dia_id": f"D1:{index}",
                            "speaker": "Narrator",
                            "utterance": f"turn {index}",
                        }
                    ]
                },
                dia_ids=[f"D1:{index}"],
                start_order=index,
                end_order=index,
                estimated_tokens=1,
            )
            for index in (1, 2)
        ]
        scan_batches = []
        chunk_two_scans = 0

        def scan_chunks(requested_chunks, question):
            nonlocal chunk_two_scans
            ids = [chunk.chunk_id for chunk in requested_chunks]
            scan_batches.append(ids)
            results = []
            for chunk in requested_chunks:
                if chunk.chunk_id == "chunk-2":
                    chunk_two_scans += 1
                    status = "ok" if chunk_two_scans >= 3 else "failed"
                else:
                    status = "ok"
                results.append(
                    {
                        "status": status,
                        "chunk_id": chunk.chunk_id,
                        "candidate_evidence": (
                            [{"option": "A", "evidence_dialogues": []}]
                            if status == "ok"
                            else []
                        ),
                        "attempts": 1,
                        "response": "",
                        "error": "" if status == "ok" else "invalid_json",
                    }
                )
            return results

        manager._scan_ablation_chunks = scan_chunks
        manager._reduce_ablation_candidates = (
            lambda question, candidates, conversation, question_type: (
                ({"answer": "(B)", "evidence_dialogues": []}, "", {})
                if len(candidates) == 2
                else ({}, "", {"error": "incomplete_candidates"})
            )
        )

        with patch(
            "src.step2_evidence_check.chunk_conversation",
            return_value=chunks,
        ), patch(
            "src.step2_evidence_check.rank_chunks",
            return_value=chunks,
        ):
            records = manager._run_iterative_ablation(
                conversation_item={"session_1": []},
                answer="(A)",
                question="Question?",
                base_evidence_blocks=[],
                question_type="single_choice",
            )

        self.assertEqual("wrong", records[-1]["result"])
        self.assertEqual(
            [["chunk-1", "chunk-2"], ["chunk-2"], ["chunk-2"]],
            scan_batches,
        )

    def test_llm_retry_exhaustion_raises_operational_error(self):
        manager = object.__new__(FullContextManager)
        manager.model_name = "test-model"
        manager.logger = logging.getLogger("step2-llm-failure-test")

        class FailingCompletions:
            def create(self, **kwargs):
                raise TimeoutError("request timed out")

        manager.openai_client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=FailingCompletions()),
            responses=types.SimpleNamespace(),
        )

        with self.assertRaisesRegex(EvidenceProcessingError, "LLM"):
            manager._call_llm("prompt", max_retries=1)

    def test_json_parse_exhaustion_raises_operational_error(self):
        manager = object.__new__(FullContextManager)
        manager.logger = logging.getLogger("step2-json-failure-test")
        manager._build_prompt = lambda **kwargs: "prompt"
        manager._call_llm = lambda prompt: ("not json", 0.01, 0)

        with self.assertRaisesRegex(EvidenceProcessingError, "JSON"):
            manager._request_json_answer(
                conversation_item={},
                question="Question?",
                evidence_blocks=[],
                only_evidence=0,
                except_evidence=0,
                max_json_retries=2,
            )

    def test_v2a_malformed_answer_retries_then_stays_pending(self):
        manager = object.__new__(FullContextManager)
        manager.logger = logging.getLogger("step2-v2a-parse-failure-test")
        manager._get_aligned_dialogue_evidence = lambda question, conversation: (
            [],
            {"result": "pass"},
        )
        manager._build_only_evidence_blocks = lambda question, evidence: []
        attempts = []

        def malformed_answer(**kwargs):
            attempts.append(1)
            return "not an option", 0.01, "prompt", 0

        manager._request_text_answer = malformed_answer

        with self.assertRaisesRegex(EvidenceProcessingError, "answer parsing"):
            FullContextManager._process_single_question(
                manager,
                {"conversation": {}},
                {
                    "question_type": "single_choice",
                    "question": "Question?\n(A) one\n(B) two",
                    "option": ["A. one", "B. two"],
                    "answer": "(A)",
                },
                0,
                0,
                None,
                1,
                0,
            )

        self.assertEqual(3, len(attempts))

    def test_ablation_needs_rerun_does_not_become_terminal_result(self):
        manager = object.__new__(FullContextManager)
        manager.logger = logging.getLogger("step2-needs-rerun-test")
        manager._get_aligned_dialogue_evidence = lambda question, conversation: (
            [],
            {"result": "pass"},
        )
        manager._build_only_evidence_blocks = lambda question, evidence: []
        manager._run_iterative_ablation = lambda **kwargs: [
            {
                "round": 1,
                "result": "invalid_evidence",
                "needs_rerun": True,
                "stop_reason": "evidence_extraction_failed_after_retries",
            }
        ]

        try:
            result = FullContextManager._process_single_question(
                manager,
                {"conversation": {}},
                {
                    "question_type": "single_choice",
                    "question": "Question?\n(A) one\n(B) two",
                    "option": ["A. one", "B. two"],
                    "answer": "(A)",
                },
                0,
                0,
                None,
                0,
                1,
            )
        except EvidenceProcessingError as exc:
            self.fail(f"needs_rerun diagnostics were discarded: {exc}")

        self.assertEqual(
            "needs_rerun",
            result["iterative_evidence_ablation_summary"]["result"],
        )
        self.assertEqual(
            "invalid_evidence",
            result["iterative_evidence_ablation"][0]["result"],
        )

    def test_needs_rerun_diagnostics_are_checkpointed_and_counted_as_failed(self):
        class NeedsRerunManager(ControlledManager):
            def _process_single_question(
                self,
                conversation_item,
                question_item,
                record_idx,
                qa_idx,
                pbar,
                only_evidence,
                except_evidence,
            ):
                result = copy.deepcopy(question_item)
                result["iterative_evidence_ablation"] = [
                    {
                        "round": 1,
                        "result": "invalid_evidence",
                        "needs_rerun": True,
                        "failed_chunks": [
                            {"chunk_id": "chunk-2", "error": "invalid_json"}
                        ],
                    }
                ]
                result["iterative_evidence_ablation_summary"] = {
                    "result": "needs_rerun",
                    "needs_rerun": True,
                    "reason": "incomplete_chunk_coverage",
                }
                return result

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "v2a.json"
            output = Path(temp_dir) / "v2b.json"
            records = [make_record("retry")]
            records[0]["qa"][0]["only_evidence_check"] = {"result": "right"}
            write_json_file(records, source)
            manager = NeedsRerunManager(output)

            manager.process_data_file(
                str(source),
                only_evidence=0,
                except_evidence=1,
                max_workers=1,
                checkpoint_every_questions=1,
            )

            saved = load_json_file(output)[0]["qa"][0]
            self.assertEqual(1, manager.failed_questions)
            self.assertEqual(
                "chunk-2",
                saved["iterative_evidence_ablation"][0]["failed_chunks"][0]["chunk_id"],
            )

    def test_default_mode_stops_before_v2b_when_v2a_has_failed_questions(self):
        calls = []

        class FailingPhaseManager:
            def __init__(self, output_path, **kwargs):
                self.output_path = output_path
                self.failed_questions = 0

            def process_data_file(self, **kwargs):
                calls.append(kwargs)
                self.failed_questions = 1
                return 2

        args = Namespace(
            answer_llm_model="test",
            answer_llm_base_url="",
            answer_llm_api_key="test",
            max_workers=1,
            only_evidence_max_workers=1,
            iterative_ablation_max_workers=1,
            checkpoint_every_questions=7,
        )
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "src.step2_evidence_check.FullContextManager",
            FailingPhaseManager,
        ):
            with self.assertRaisesRegex(EvidenceProcessingError, "v2a"):
                evidence_check_main(
                    args,
                    str(Path(temp_dir) / "input.json"),
                    str(Path(temp_dir) / "story_v2b.json"),
                )

        self.assertEqual(1, len(calls))
        self.assertEqual(7, calls[0]["checkpoint_every_questions"])


if __name__ == "__main__":
    unittest.main()
