"""Offline regression and actual-tokenizer integration tests; no paid API calls."""
import copy
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import build_sft_dataset as build
import collect_teacher_data as collect
from shopping_grpo.acceptance.jev_client import JEV_RUBRIC_VERSION, JevApiError
from shopping_grpo.environment.protocol import message_protocol_problem, runtime_protocol
from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS
from shopping_grpo.evaluation.rollout import STUDENT_SYSTEM_PROMPT, collect_for_task
from shopping_grpo.training.sft.preparation import (
    atomic_json, digest, ensure_identity, file_hash, harness_identity, read_jsonl,
    task_index, terminal_problem,
)


def call(name, args, identifier):
    return {"role": "assistant", "content": None, "tool_calls": [{
        "id": identifier, "type": "function", "function": {
            "name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]}


def trajectory(rid="1", alternative=False):
    messages = [{"role": "system", "content": STUDENT_SYSTEM_PROMPT},
                {"role": "user", "content": f"买白色商品，任务{rid}，预算200元"}]
    for index, (name, args) in enumerate([
        ("search_products", {"query": "白色商品"}),
        ("open_product", {"asin": "100000000001"}),
        ("select_option", {"option_id": "opt_1"}), ("buy_now", {})]):
        messages += [call(name, args, str(index)),
                     {"role": "tool", "tool_call_id": str(index), "content": "公开商品信息"}]
    kind = "partial_alternative_purchase" if alternative else "gold_purchase"
    return {"runtime_protocol": runtime_protocol(), "status": "done", "messages": messages,
            "steps": [{}] * 4, "terminal_result": {
                "over": True, "reward_valid": True, "termination_reason": "purchase",
                "reward_detail": {"reward_type": kind, "reward_valid": True},
                "purchase": {"asin": "100000000001", "name": "实际商品", "price": 100,
                             "options": {"颜色": "白色"}, "attributes": ["实际属性"],
                             "instruction_text": "错误的商品自带历史需求"}},
            "meta": {"record_id": rid, "teacher": "glm", "accepted": True,
                     "attempt": 1, "accept_reason": "gold" if not alternative else "jev:fully_satisfies",
                     "difficulty": "easy", "persona_condition": "no_profile", "audit": {}}}


class FakeJev:
    def __init__(self, fail=False):
        self.fail = fail
        self.inputs = []

    def decide_satisfaction(self, state):
        self.inputs.append(json.loads(state))
        if self.fail:
            raise JevApiError("temporary outage")
        return {"choice": "fully_satisfies", "rubric_version": JEV_RUBRIC_VERSION,
                "request_hash": "test-request-hash", "model_reported": "fake-jev"}


class PreparationRegressionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)

    def test_option_parameters_raw_and_normalized(self):
        raw = [call("select_option", {"option_id": "opt_1"}, "c")]
        self.assertIsNone(message_protocol_problem(raw))
        raw[0]["tool_calls"][0]["function"]["arguments"] = {"option_id": "opt_1"}
        self.assertIsNone(message_protocol_problem(raw))
        raw[0]["tool_calls"][0]["function"]["arguments"] = {"value": "white"}
        self.assertEqual(message_protocol_problem(raw), "unsupported_option_arguments")

    def test_jev_uses_original_request_and_purchased_evidence(self):
        t, jev = trajectory(alternative=True), FakeJev()
        ok, reason, audit = collect.judge_trajectory(t, jev, {"attribute": ["Gold专属属性"]}, None)
        self.assertTrue(ok)
        self.assertEqual(jev.inputs[0]["user_request"], t["messages"][1]["content"])
        self.assertEqual(jev.inputs[0]["purchased_product"]["attributes"], ["实际属性"])
        self.assertNotIn("Gold专属属性", json.dumps(jev.inputs, ensure_ascii=False))
        self.assertNotIn("错误的商品自带历史需求", json.dumps(jev.inputs, ensure_ascii=False))
        t["meta"].update(accept_reason=reason, audit=audit)
        self.assertIsNone(build.admission_problem(t))
        t["terminal_result"]["purchase"]["price"] = 101
        self.assertEqual(build.admission_problem(t), "jev_evidence_mismatch")

    def test_missing_or_old_jev_evidence_rejected(self):
        t = trajectory(alternative=True)
        self.assertEqual(build.admission_problem(t), "missing_jev_approval")
        t["meta"]["audit"] = {"jev_verdict": {"choice": "fully_satisfies"}}
        self.assertEqual(build.admission_problem(t), "jev_audit_incomplete")

    def test_schema_extra_parameters_and_broken_pair_rejected(self):
        t = trajectory()
        t["messages"][-2]["tool_calls"][0]["function"]["arguments"] = '{"reason":"done"}'
        self.assertIn("tool_schema_fields", build.admission_problem(t))
        t = trajectory()
        t["messages"][-1]["tool_call_id"] = "other"
        self.assertEqual(build.admission_problem(t), "conversation_tool_id")

    def test_terminal_and_accept_reason_required(self):
        t = trajectory()
        t["meta"]["accept_reason"] = "manual"
        self.assertEqual(build.admission_problem(t), "accept_reason_mismatch")
        t = trajectory()
        t["terminal_result"]["reward_detail"]["reward_type"] = "wrong_purchase"
        self.assertEqual(build.admission_problem(t), "purchase_not_accepted")
        t = trajectory()
        t["messages"] = []
        self.assertEqual(build.admission_problem(t), "conversation_incomplete")

    def test_retryable_statuses_are_not_marked_completed(self):
        rows = [{"record_id": str(i), "final": state, "runtime_protocol": runtime_protocol()}
                for i, state in enumerate(["accepted", "teacher_hard", "env_exhausted", "pending_jev"])]
        (self.directory / "task_status_glm.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        with patch.object(collect, "OUT", self.directory):
            self.assertEqual(collect.collection_completed_ids(), {"0", "1"})

    def test_batch_identity_blocks_changed_inputs(self):
        path = self.directory / "manifest.json"
        ensure_identity(path, {"prompt": "a"})
        ensure_identity(path, {"prompt": "a"})
        with self.assertRaisesRegex(ValueError, "identity changed"):
            ensure_identity(path, {"prompt": "b"})

    def test_duplicate_frozen_task_ids_fail(self):
        path = self.directory / "tasks.jsonl"
        path.write_text('{"record_id":"1"}\n{"record_id":"1"}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            task_index(path)

    def run_task(self, jev, rollout):
        client = type("Client", (), {"total_usage": {}})()
        task = {"record_id": "1", "difficulty": "easy", "persona_condition": "no_profile"}
        with patch.object(collect, "OUT", self.directory), \
             patch.object(collect, "TRAJ", self.directory / "trajectories"), \
             patch.object(collect, "make_client", return_value=client), \
             patch.object(collect, "collect_for_task", side_effect=rollout) as mocked, \
             patch.object(collect.time, "sleep"):
            result = collect.run_task(task, {"1": 0}, {"1": {}}, {}, jev)
        return result, mocked

    def test_teacher_outage_does_not_consume_attempt(self):
        error = {"status": "error", "error": {"type": "TeacherApiError"},
                 "runtime_protocol": runtime_protocol()}
        result, mocked = self.run_task(FakeJev(), [error, trajectory()])
        self.assertEqual(result["final"], "accepted")
        self.assertEqual(result["attempts"], 1)
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(len(list((self.directory / "infra").glob("*.json"))), 1)

    def test_jev_outage_resumes_saved_purchase_without_teacher(self):
        result, mocked = self.run_task(FakeJev(fail=True), [trajectory(alternative=True)])
        self.assertEqual(result["final"], "pending_jev")
        self.assertEqual(mocked.call_count, 1)
        result, mocked = self.run_task(FakeJev(), [])
        self.assertEqual(result["final"], "accepted")
        self.assertEqual(result["attempts"], 1)
        mocked.assert_not_called()

    def test_teacher_failure_attempts_survive_resume(self):
        failed = trajectory()
        failed["terminal_result"]["reward_detail"]["reward_type"] = "wrong_purchase"
        result, mocked = self.run_task(FakeJev(), [copy.deepcopy(failed) for _ in range(3)])
        self.assertEqual(result["final"], "teacher_hard")
        self.assertEqual(mocked.call_count, 3)
        result, mocked = self.run_task(FakeJev(), [])
        self.assertEqual(result["final"], "teacher_hard")
        mocked.assert_not_called()

    def test_current_harness_rollout_and_online_admission(self):
        from shopping_grpo.environment.observation import OBSERVATION_VERSION
        sample = trajectory()
        product = {"asin": "100000000001", "title": "白色商品", "price": 100,
                   "key_attributes": ["实际属性"]}
        def state(page, actions, **extra):
            return {"observation_version": OBSERVATION_VERSION, "page_type": page,
                    "actions": actions, **extra}
        class Env:
            def __init__(self, **kwargs):
                pass
            def reset(self, task_id):
                return {"instruction": sample["messages"][1]["content"],
                        "observation_state": state("search_home", [], search_available=True)}
            def step(self, action):
                if action.startswith("search["):
                    observation = state("search_results", ["100000000001"], products=[product])
                elif action == "click[100000000001]":
                    observation = state("product_detail", ["Buy Now"], product=product,
                                        available_options={"颜色": ["白色"]})
                elif action.startswith("click_option["):
                    observation = state("product_detail", ["Buy Now"], product=product,
                                        available_options={"颜色": ["白色"]}, selected_options={"颜色": "白色"})
                elif action == "click[Buy Now]":
                    return {**sample["terminal_result"], "done": True, "reward": 1,
                            "observation_state": state("terminal", [])}
                else:
                    raise AssertionError(action)
                return {"done": False, "reward": 0, "observation_state": observation}
            def release(self):
                pass
        class Client:
            def __init__(self):
                self.actions = iter(sample["messages"][2::2])
                self.tools = []
            def complete(self, messages, tools):
                self.tools.append(copy.deepcopy(tools))
                return next(self.actions)
        client = Client()
        t = collect_for_task({"task_id": 0}, client, env_factory=Env)
        self.assertEqual(t["status"], "done")
        self.assertEqual(len(client.tools), 4)
        for tools in client.tools:
            self.assertEqual(tools, SHOP_TOOL_SCHEMAS)
        ok, reason, audit = collect.judge_trajectory(t, FakeJev(), {}, None)
        self.assertTrue(ok)
        t["meta"] = {"accept_reason": reason, "audit": audit}
        self.assertIsNone(build.admission_problem(t))


MODEL = os.environ.get("SHOP_MODEL")


@unittest.skipUnless(MODEL and Path(MODEL).is_dir(), "Set SHOP_MODEL to a local tokenizer")
class ActualTokenizerPipelineTest(unittest.TestCase):
    def fixture(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        samples = [build.build_sample(trajectory(str(i)), {}, {}) for i in range(16)]
        # Identical requests with different task IDs must never cross splits.
        samples[1]["messages"][1]["content"] = samples[2]["messages"][1]["content"]
        tasks = [{"record_id": str(i), "split": "train", "tag": "train"} for i in range(16)]
        tasks_path = directory / "tasks.jsonl"
        tasks_path.write_text("\n".join(json.dumps(t) for t in tasks), encoding="utf-8")
        atomic_json(directory / "build_identity.json", {
            "collection_manifest": {"tasks_sha256": file_hash(tasks_path)}})
        return directory, samples, tasks_path

    def write_samples(self, directory, samples):
        (directory / "samples.jsonl").write_text(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in samples) + "\n", encoding="utf-8")
        from token_stats import count_tokens
        count_tokens(directory, MODEL)

    def test_duplicate_requests_grouped_and_overlong_trajectory_dropped(self):
        from export_sft_parquet import export_dataset
        directory, samples, tasks = self.fixture()
        samples[-1]["messages"][3]["content"] = "long observation " * 18000
        self.write_samples(directory, samples)
        report = export_dataset(directory, MODEL, tasks, validation_ratio=0.5)
        self.assertEqual(report["assignments"]["1"], report["assignments"]["2"])
        self.assertEqual([r["record_id"] for r in report["excluded"]], ["15"])
        self.assertEqual(sum(v["rows"] for v in report["splits"].values()), 15)

    def test_eval_task_is_rejected_even_with_consistent_input_hashes(self):
        from export_sft_parquet import export_dataset
        directory, samples, tasks = self.fixture()
        rows = read_jsonl(tasks)
        rows[0]["tag"] = "eval"
        tasks.write_text("\n".join(json.dumps(t) for t in rows), encoding="utf-8")
        atomic_json(directory / "build_identity.json", {
            "collection_manifest": {"tasks_sha256": file_hash(tasks)}})
        self.write_samples(directory, samples)
        with self.assertRaisesRegex(ValueError, "Non-training/frozen eval"):
            export_dataset(directory, MODEL, tasks, validation_ratio=0.5)
        self.assertFalse((directory / "train.parquet").exists())

    def test_failed_full_verification_preserves_previous_output(self):
        from export_sft_parquet import export_dataset
        directory, samples, tasks = self.fixture()
        self.write_samples(directory, samples)
        good = export_dataset(directory, MODEL, tasks, validation_ratio=0.5)
        for message in samples[0]["messages"]:
            message["train"] = 0
        self.write_samples(directory, samples)
        with self.assertRaisesRegex(ValueError, "no supervised"):
            export_dataset(directory, MODEL, tasks, validation_ratio=0.5)
        self.assertEqual(file_hash(directory / "train.parquet"), good["output_sha256"]["train.parquet"])
        self.assertEqual(file_hash(directory / "val.parquet"), good["output_sha256"]["val.parquet"])
        self.assertFalse(list(directory.glob("*.pending")))

    def test_single_teacher_build_tokens_export_readback_and_stale_stats(self):
        from token_stats import count_tokens
        from export_sft_parquet import export_dataset
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collection, output = root / "collection", root / "dataset"
            (collection / "trajectories").mkdir(parents=True)
            tasks, personas = root / "tasks.jsonl", root / "personas.jsonl"
            personas.write_text("", encoding="utf-8")
            rows, status = [], []
            for i in range(16):
                rid = str(i)
                rows.append({"record_id": rid, "split": "train", "tag": "train",
                             "difficulty": "easy", "persona_condition": "no_profile"})
                t = trajectory(rid, alternative=(i % 3 == 0))
                if i % 3 == 0:
                    ok, reason, audit = collect.judge_trajectory(t, FakeJev(), {}, None)
                    self.assertTrue(ok)
                    t["meta"].update(accept_reason=reason, audit=audit)
                atomic_json(collection / "trajectories" / f"{rid}_2_good.json", t)
                status.append({"record_id": rid, "final": "accepted", "runtime_protocol": runtime_protocol()})
            # A bad earlier accepted attempt must not hide a later valid attempt.
            bad = trajectory("0")
            bad["terminal_result"]["reward_valid"] = False
            atomic_json(collection / "trajectories/0_1_bad.json", bad)
            tasks.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            (collection / "task_status_glm.jsonl").write_text(
                "\n".join(json.dumps(r) for r in status), encoding="utf-8")
            atomic_json(collection / "manifest.json", {
                "runtime_protocol": runtime_protocol(), "tasks_sha256": file_hash(tasks),
                "personas_sha256": file_hash(personas), "harness": harness_identity(ROOT)})
            argv = ["build_sft_dataset.py", "--collection-dir", str(collection), "--output-dir", str(output),
                    "--tasks", str(tasks), "--personas", str(personas)]
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                build.main()
            self.assertEqual(len(read_jsonl(output / "samples.jsonl")), 16)
            self.assertEqual(json.loads((output / "build_report.json").read_text())["dropped_admission"], {"reward_invalid": 1})
            count_tokens(output, MODEL)
            report = export_dataset(output, MODEL, tasks, validation_ratio=0.5)
            self.assertEqual(report["status"], "all_rows_verified")
            self.assertEqual(sum(s["rows"] for s in report["splits"].values()), 16)
            train, val = (pd.read_parquet(output / f"{s}.parquet") for s in ("train", "val"))
            self.assertFalse(set(train.record_id) & set(val.record_id))
            self.assertTrue(all(train.terminal_result.notna()))
            self.assertTrue(all(val.runtime_protocol.notna()))
            hashes = report["output_sha256"]
            repeated = export_dataset(output, MODEL, tasks, validation_ratio=0.5)
            self.assertEqual(hashes, repeated["output_sha256"])
            with (output / "samples.jsonl").open("a", encoding="utf-8") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "Stale token"):
                export_dataset(output, MODEL, tasks, validation_ratio=0.5)


if __name__ == "__main__":
    unittest.main()
