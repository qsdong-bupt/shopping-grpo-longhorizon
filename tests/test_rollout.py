import json
import tempfile
import unittest
from http.client import RemoteDisconnected
from pathlib import Path
from urllib.error import URLError
from unittest.mock import patch

from shopping_grpo.environment.actions import action_guard_tool_message
from shopping_grpo.environment.protocol import runtime_protocol
from shopping_grpo.environment.client import ShopEnvironmentError
from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS
from shopping_grpo.evaluation.rollout import (
    CollectionInfrastructureError,
    OpenAIChatClient,
    SYSTEM_PROMPT,
    TEACHER_SYSTEM_PROMPT,
    collect_tasks,
    collect_for_task,
    completed_task_attempts,
    load_tasks,
    rollout_interrupted,
)


class FakeEnv:
    def __init__(self, **kwargs):
        self.actions = []
        self.released = False

    def reset(self, task_id):
        return {"env_idx": 0, "instruction": f"Instruction: task {task_id}"}

    def step(self, action):
        self.actions.append(action)
        if action == "search[乳胶枕]":
            return {
                "instruction": (
                    "[SHOPPING_OBSERVATION_V3]\n"
                    "page_type: search_results\n"
                    "1|100000000001|999.0|测试店|测试类|attr|测试商品\n"
                    "\n"
                    "搜索功能是否可用: True\n"
                    '可点击的按钮: ["100000000001"]'
                ),
                "reward": 0.0,
                "done": False,
            }
        if action == "click[100000000001]":
            return {
                "instruction": 'detail\n\n可点击的按钮: ["Buy Now"]',
                "reward": 0.0,
                "done": False,
            }
        return {
            "instruction": "done page",
            "reward": 1.0,
            "done": True,
            "over": True,
            "purchase": {"asin": "A1"},
            "reward_detail": {"r_type": 1, "r_att": 1, "r_option": 1, "r_price": 1},
        }

    def release(self):
        self.released = True


class FailingEnv(FakeEnv):
    def step(self, action):
        self.actions.append(action)
        raise RuntimeError("env exploded")


class NonTerminalEnv(FakeEnv):
    def step(self, action):
        self.actions.append(action)
        return {"instruction": "keep going", "reward": 0.0, "done": False}


class ReleaseFailingEnv(FakeEnv):
    def release(self):
        raise OSError("ShopSimulator unavailable during release")


class UnavailableEnv(FakeEnv):
    def reset(self, task_id):
        raise ShopEnvironmentError("Unable to get available environment resource, please try again later")


class GuardRecoveryEnv(FakeEnv):
    """用于验证非法尝试被合法工具调用隔开时仍可恢复。"""

    def step(self, action):
        self.actions.append(action)
        if action == "search[乳胶枕]":
            return {
                "instruction": (
                    "[SHOPPING_OBSERVATION_V3]\n"
                    "page_type: search_results\n"
                    "1|100000000001|999.0|测试店|测试类|attr|测试商品\n"
                    "\n"
                    "搜索功能是否可用: True\n"
                    '可点击的按钮: ["100000000001"]'
                ),
                "reward": 0.0,
                "done": False,
            }
        if action == "click[100000000001]":
            return {
                "instruction": 'detail\n\n可点击的按钮: ["back to search", "Buy Now"]',
                "reward": 0.0,
                "done": False,
            }
        if action == "click[Back to Search]":
            return {
                "instruction": '[SHOPPING_OBSERVATION_V3]\npage_type: search_results\n1|100000000001|999.0|测试商品\n\n可点击的按钮: ["100000000001"]',
                "reward": 0.0,
                "done": False,
            }
        if action == "click[Buy Now]":
            return {
                "instruction": "done page",
                "reward": 1.0,
                "done": True,
                "over": True,
                "purchase": {"asin": "A1"},
                "reward_detail": {"r_type": 1, "r_att": 1, "r_option": 1, "r_price": 1},
            }
        raise AssertionError(f"unexpected action: {action}")


class MockClient:
    def __init__(self, messages):
        self.messages = list(messages)
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append({"messages": messages, "tools": tools})
        return self.messages.pop(0)


def assistant_tool(name, arguments, call_id="call_1"):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
            }
        ],
    }


class RolloutTest(unittest.TestCase):
    def test_student_prompt_contains_protocol_only(self):
        """Student 版只含协议层：环境约定齐备，购物决策规则一条不进。"""
        self.assertIn("单轮购物任务", SYSTEM_PROMPT)
        self.assertIn("不得向用户追问", SYSTEM_PROMPT)
        self.assertIn("不要在任务结束前输出最终答复", SYSTEM_PROMPT)
        self.assertIn("当前页面是动作合法性的唯一依据", SYSTEM_PROMPT)
        self.assertNotIn("信息子页", SYSTEM_PROMPT)
        self.assertIn("无参数工具", SYSTEM_PROMPT)
        self.assertIn("历史 observation 可用于记住和比较候选", SYSTEM_PROMPT)
        self.assertIn("不能直接点击历史页面中的 ASIN", SYSTEM_PROMPT)
        for decision_rule in (
            "品类 > 预算 > 品牌",
            "直接调用 `buy_now`",
            "搜索与候选探索",
            "防止循环",
            "多次有实质差异的搜索",
        ):
            self.assertNotIn(decision_rule, SYSTEM_PROMPT)

    def test_teacher_prompt_adds_decision_rules(self):
        """Teacher 版在协议层之上追加决策规则，供采集教材使用。"""
        self.assertIn("软偏好用于比较满足明确要求的候选，不得覆盖硬约束", TEACHER_SYSTEM_PROMPT)
        self.assertIn("必要规格已选齐且实际价格已确认时，直接调用 `buy_now`", TEACHER_SYSTEM_PROMPT)
        self.assertIn("所有影响可购买 variant 的必要规格轴", TEACHER_SYSTEM_PROMPT)
        self.assertNotIn("finish_without_purchase", TEACHER_SYSTEM_PROMPT)
        self.assertNotIn("多次有实质差异的搜索和多个候选核验", TEACHER_SYSTEM_PROMPT)
        self.assertNotIn("是否达到结束资格由环境判断", TEACHER_SYSTEM_PROMPT)
        self.assertIn("不要连续重复同一动作", TEACHER_SYSTEM_PROMPT)
        self.assertIn("未展示的信息不得假定已满足", TEACHER_SYSTEM_PROMPT)
        self.assertIn("如没有合格候选，不得仅为结束任务而购买不符合要求的商品", TEACHER_SYSTEM_PROMPT)
        self.assertNotIn("Reviews", TEACHER_SYSTEM_PROMPT)
        self.assertNotIn("Reviews", SYSTEM_PROMPT)

    def test_guard_lists_only_current_page_targets(self):
        """守卫反馈只提供当前页面目标。"""
        message = action_guard_tool_message(
            assistant_tool("buy_now", {}, "call_buy")["tool_calls"][0],
            "click_not_in_previous_observation",
            '[SHOPPING_OBSERVATION_V3]\npage_type: product_detail\n'
            '可点击的按钮: ["< Prev"]',
        )

        self.assertNotIn("你处于信息子页", message["content"])
        self.assertIn("当前页面：product_detail", message["content"])
        self.assertIn("请求工具：buy_now", message["content"])
        self.assertIn("可用工具：prev_page", message["content"])
        self.assertNotIn("下一步必须调用", message["content"])

    def test_collect_for_task_executes_openai_tool_calls_until_done(self):
        client = MockClient(
            [
                assistant_tool("search_products", {"query": "乳胶枕"}, "call_search"),
                assistant_tool("open_product", {"asin": "100000000001"}, "call_open"),
                assistant_tool("buy_now", {}, "call_buy"),
            ]
        )
        env = FakeEnv()

        traj = collect_for_task(
            {"task_id": 7},
            client=client,
            env_factory=lambda **kwargs: env,
            base_url="http://shop.test",
            max_steps=4,
        )

        self.assertEqual(traj["status"], "done")
        self.assertTrue(traj["trajectory_id"])
        self.assertEqual(env.actions, ["search[乳胶枕]", "click[100000000001]", "click[Buy Now]"])
        self.assertTrue(env.released)
        self.assertEqual(traj["steps"][0]["tool_call"]["function"]["name"], "search_products")
        self.assertEqual(traj["steps"][2]["env_action"], "click[Buy Now]")
        self.assertEqual(traj["terminal_result"]["purchase"]["asin"], "A1")
        self.assertTrue(any(message["role"] == "tool" for message in traj["messages"]))

    def test_rollout_sends_fixed_tool_schema_on_every_page(self):
        class PageEnv(FakeEnv):
            def reset(self, task_id):
                return {
                    "env_idx": 0,
                    "instruction": (
                        "[SHOPPING_OBSERVATION_V3]\npage_type: search_home\n"
                        "搜索功能是否可用: True\n可点击的按钮: []"
                    ),
                }

            def step(self, action):
                self.actions.append(action)
                if action == "search[乳胶枕]":
                    return {"instruction": (
                        "[SHOPPING_OBSERVATION_V3]\npage_type: search_results\n"
                        "1|100000000001|99|店|类|属性|商品\n"
                        "可点击的按钮: [\"100000000001\", \"Next >\", \"Back to Search\"]"
                    ), "reward": 0.0, "done": False}
                if action == "click[100000000001]":
                    return {"instruction": (
                        "[SHOPPING_OBSERVATION_V3]\npage_type: product_detail\n"
                        "option_choices: [{\"option_id\":\"opt_1\",\"axis\":\"颜色\",\"value\":\"红\"}]\n"
                        "可点击的按钮: [\"opt_1\", \"Buy Now\", \"< Prev\", \"Back to Search\"]"
                    ), "reward": 0.0, "done": False}
                if action.startswith("click_option["):
                    return {"instruction": (
                        "[SHOPPING_OBSERVATION_V3]\npage_type: product_detail\n"
                        "option_choices: [{\"option_id\":\"opt_1\",\"axis\":\"颜色\",\"value\":\"红\"}]\n"
                        "可点击的按钮: [\"opt_1\", \"Buy Now\", \"< Prev\", \"Back to Search\"]"
                    ), "reward": 0.0, "done": False}
                return {"instruction": "done", "reward": 1.0, "done": True, "over": True}

        client = MockClient([
            assistant_tool("search_products", {"query": "乳胶枕"}, "call_search"),
            assistant_tool("open_product", {"asin": "100000000001"}, "call_open"),
            assistant_tool("select_option", {"option_id": "opt_1"}, "call_option"),
            assistant_tool("buy_now", {}, "call_buy"),
        ])
        env = PageEnv()
        traj = collect_for_task(
            {"task_id": 7}, client=client, env_factory=lambda **kwargs: env,
            base_url="http://shop.test", max_steps=5,
        )

        self.assertEqual(traj["status"], "done")
        expected_tools = [schema["function"]["name"] for schema in SHOP_TOOL_SCHEMAS]
        self.assertEqual(len(client.requests), 4)
        self.assertEqual(
            [[schema["function"]["name"] for schema in request["tools"]]
             for request in client.requests],
            [expected_tools] * len(client.requests),
        )
        self.assertTrue(all(request["tools"] == SHOP_TOOL_SCHEMAS for request in client.requests))

    def test_environment_does_not_expose_model_stop_tool(self):
        class EnvironmentV21(FakeEnv):
            def reset(self, task_id):
                result = super().reset(task_id)
                result["environment_version"] = "shopsimulator-environment-v2.2"
                return result

        client = MockClient([{"role": "assistant", "content": "stop"}])
        env = EnvironmentV21()
        collect_for_task(
            {"task_id": 7},
            client=client,
            env_factory=lambda **kwargs: env,
            base_url="http://shop.test",
            max_steps=1,
        )
        tool_names = [schema["function"]["name"] for schema in client.requests[0]["tools"]]
        self.assertNotIn("finish_without_purchase", tool_names)
        self.assertNotIn("合理结束", client.requests[0]["messages"][0]["content"])
        self.assertTrue(env.released)

    def test_collect_for_task_blocks_invalid_click_then_keeps_clean_recovery(self):
        client = MockClient(
            [
                assistant_tool("search_products", {"query": "乳胶枕"}, "call_search"),
                assistant_tool("open_product", {"asin": "100000000001"}, "call_open"),
                assistant_tool("view_features", {}, "call_invalid"),
                assistant_tool("buy_now", {}, "call_buy"),
            ]
        )
        env = FakeEnv()

        traj = collect_for_task(
            {"task_id": 10},
            client=client,
            env_factory=lambda **kwargs: env,
            base_url="http://shop.test",
            max_steps=5,
        )

        self.assertEqual(traj["status"], "done")
        self.assertEqual(env.actions, ["search[乳胶枕]", "click[100000000001]", "click[Buy Now]"])
        self.assertEqual(len(traj["blocked_tool_calls"]), 1)
        self.assertEqual(traj["blocked_tool_calls"][0]["reason"], "unknown_or_invalid_tool")
        self.assertNotIn(
            "view_features",
            [step["tool_name"] for step in traj["steps"]],
        )
        self.assertTrue(
            any(
                message.get("role") == "tool"
                and message.get("tool_call_id") == "call_invalid"
                and message.get("runtime_action_guard") is True
                for message in traj["messages"]
            )
        )

    def test_collect_for_task_allows_current_page_navigation_after_option_selection(self):
        """选择规格后仍由环境当前页面决定能否浏览，采集器不另造状态机。"""
        class OptionEnv(FakeEnv):
            def step(self, action):
                self.actions.append(action)
                if action == "search[乳胶枕]":
                    return {"instruction": "[SHOPPING_OBSERVATION_V3]\npage_type: search_results\n1|100000000001|999.0|测试店|测试类|attr|测试商品\n\n搜索功能是否可用: True\n可点击的按钮: [\"100000000001\"]", "reward": 0.0, "done": False}
                if action == "click[100000000001]":
                    return {
                        "instruction": ('page_type: product_detail\n'
                                        'option_choices: [{"option_id":"opt_1","axis":"颜色","value":"满天星"}]\n'
                                        '可点击的按钮: ["opt_1", "back to search", "Buy Now"]'),
                        "reward": 0.0,
                        "done": False,
                    }
                if action.startswith("click_option["):
                    return {
                        "instruction": 'selected\n\n可点击的按钮: ["back to search", "Buy Now"]',
                        "reward": 0.0,
                        "done": False,
                    }
                if action == "click[Back to Search]":
                    return {
                        "instruction": '[SHOPPING_OBSERVATION_V3]\npage_type: search_results\n1|100000000001|999.0|测试商品\n\n可点击的按钮: ["100000000001"]',
                        "reward": 0.0,
                        "done": False,
                    }
                if action == "click[Buy Now]":
                    return {
                        "instruction": "done",
                        "reward": 1.0,
                        "done": True,
                        "over": True,
                        "purchase": {"asin": "A1"},
                        "reward_detail": {"r_type": 1, "r_att": 1, "r_option": 1, "r_price": 1},
                    }
                raise AssertionError(action)

        client = MockClient(
            [
                assistant_tool("search_products", {"query": "乳胶枕"}, "call_search"),
                assistant_tool("open_product", {"asin": "100000000001"}, "call_open"),
                assistant_tool("select_option", {"option_id": "opt_1"}, "call_option"),
                assistant_tool("back_to_search", {}, "call_navigation"),
                assistant_tool("open_product", {"asin": "100000000001"}, "call_reopen"),
                assistant_tool("buy_now", {}, "call_buy"),
            ]
        )
        env = OptionEnv()

        traj = collect_for_task({"task_id": 12}, client=client, env_factory=lambda **kwargs: env)

        self.assertEqual(traj["status"], "done")
        self.assertEqual([step["env_action"] for step in traj["steps"]], [
            "search[乳胶枕]",
            "click[100000000001]",
            'click_option[{"axis": "颜色", "value": "满天星"}]',
            "click[Back to Search]",
            "click[100000000001]",
            "click[Buy Now]",
        ])
        self.assertEqual(traj["blocked_tool_calls"], [])

    def test_collect_for_task_keeps_one_tool_schema_after_option_selection(self):
        """所有阶段使用同一工具 schema，环境 observation 是唯一动作边界。"""
        class OptionEnv(FakeEnv):
            def step(self, action):
                self.actions.append(action)
                if action == "search[乳胶枕]":
                    return {"instruction": "[SHOPPING_OBSERVATION_V3]\npage_type: search_results\n1|100000000001|999.0|测试店|测试类|attr|测试商品\n\n搜索功能是否可用: True\n可点击的按钮: [\"100000000001\"]", "reward": 0.0, "done": False}
                if action == "click[100000000001]":
                    return {
                        "instruction": ('page_type: product_detail\n'
                                        'option_choices: [{"option_id":"opt_1","axis":"颜色","value":"满天星"}]\n'
                                        '可点击的按钮: ["opt_1", "Buy Now"]'),
                        "reward": 0.0,
                        "done": False,
                    }
                if action.startswith("click_option["):
                    return {
                        "instruction": 'selected\n\n可点击的按钮: ["Buy Now"]',
                        "reward": 0.0,
                        "done": False,
                    }
                if action == "click[Buy Now]":
                    return {
                        "instruction": "done",
                        "reward": 1.0,
                        "done": True,
                        "over": True,
                        "purchase": {"asin": "A1"},
                        "reward_detail": {"r_type": 1, "r_att": 1, "r_option": 1, "r_price": 1},
                    }
                raise AssertionError(action)

        client = MockClient(
            [
                assistant_tool("search_products", {"query": "乳胶枕"}, "call_search"),
                assistant_tool("open_product", {"asin": "100000000001"}, "call_open"),
                assistant_tool("select_option", {"option_id": "opt_1"}, "call_option"),
                assistant_tool("buy_now", {}, "call_buy"),
            ]
        )

        trajectory = collect_for_task(
            {"task_id": 13}, client=client, env_factory=OptionEnv, base_url="http://shop.test"
        )

        self.assertEqual(trajectory["status"], "done")
        self.assertTrue(client.requests)
        self.assertTrue(all(request["tools"] == SHOP_TOOL_SCHEMAS for request in client.requests))

    def test_collect_for_task_allows_recovery_after_separated_guard_rejections(self):
        """合法动作应重置守卫计数，避免累计三次历史点击提前中止。"""
        client = MockClient(
            [
                assistant_tool("search_products", {"query": "乳胶枕"}, "call_search"),
                assistant_tool("open_product", {"asin": "999999999999"}, "call_old_asin"),
                assistant_tool("open_product", {"asin": "100000000001"}, "call_open"),
                assistant_tool("next_page", {}, "call_missing_next"),
                assistant_tool("back_to_search", {}, "call_back"),
                assistant_tool("buy_now", {}, "call_buy_on_subpage"),
                assistant_tool("open_product", {"asin": "100000000001"}, "call_return"),
                assistant_tool("buy_now", {}, "call_buy"),
            ]
        )
        env = GuardRecoveryEnv()

        traj = collect_for_task(
            {"task_id": 11},
            client=client,
            env_factory=lambda **kwargs: env,
            base_url="http://shop.test",
            max_steps=8,
        )

        self.assertEqual(traj["status"], "done")
        self.assertEqual(len(traj["blocked_tool_calls"]), 3)
        self.assertEqual(env.actions, [
            "search[乳胶枕]",
            "click[100000000001]",
            "click[Back to Search]",
            "click[100000000001]",
            "click[Buy Now]",
        ])

    def test_collect_for_task_keeps_exception_trajectory_and_releases_env(self):
        client = MockClient([assistant_tool("search_products", {"query": "乳胶枕"})])
        env = FailingEnv()

        traj = collect_for_task(
            {"task_id": 8},
            client=client,
            env_factory=lambda **kwargs: env,
            base_url="http://shop.test",
            max_steps=2,
        )

        self.assertEqual(traj["status"], "error")
        self.assertIn("env exploded", traj["error"]["message"])
        self.assertEqual(traj["steps"][0]["env_action"], "search[乳胶枕]")
        self.assertTrue(env.released)

    def test_rollout_interrupted_raises_keyboard_interrupt_for_finally_release(self):
        with self.assertRaises(KeyboardInterrupt):
            rollout_interrupted(None, None)

    def test_completed_task_attempts_defaults_missing_attempt_index_to_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "raw.jsonl"
            output.write_text(
                json.dumps({"task_id": 1, "trajectory_id": "old", "runtime_protocol": runtime_protocol()}) + "\n"
                + json.dumps({"task_id": 3, "trajectory_id": "old2", "runtime_protocol": runtime_protocol()}) + "\n",
                encoding="utf-8",
            )

            self.assertEqual(completed_task_attempts(output), {(1, 0), (3, 0)})

    def test_collect_tasks_skips_existing_output_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "raw.jsonl"
            output.write_text(json.dumps({"task_id": 1, "trajectory_id": "old", "runtime_protocol": runtime_protocol()}) + "\n", encoding="utf-8")
            client = MockClient([assistant_tool("buy_now", {}, "call_buy")])

            written = collect_tasks(
                [{"task_id": 1}, {"task_id": 2}],
                client=client,
                output_path=output,
                base_url="http://shop.test",
                max_steps=1,
                env_factory=FakeEnv,
            )
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual([row["task_id"] for row in rows], [1, 2])
        self.assertEqual(len(written), 1)

    def test_collect_tasks_resumes_missing_attempts_for_each_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "raw.jsonl"
            output.write_text(
                json.dumps({"task_id": 1, "attempt_index": 0, "trajectory_id": "old", "runtime_protocol": runtime_protocol()}) + "\n",
                encoding="utf-8",
            )
            client = MockClient(
                [
                    assistant_tool("buy_now", {}, "call_1"),
                    assistant_tool("buy_now", {}, "call_2"),
                    assistant_tool("buy_now", {}, "call_3"),
                ]
            )

            written = collect_tasks(
                [{"task_id": 1}, {"task_id": 2}],
                client=client,
                output_path=output,
                base_url="http://shop.test",
                max_steps=1,
                env_factory=FakeEnv,
                attempts_per_task=2,
            )

        self.assertEqual(
            [(row["task_id"], row["attempt_index"]) for row in written],
            [(1, 1), (2, 0), (2, 1)],
        )

    def test_collect_tasks_stops_after_a_release_failure(self):
        """环境租约未释放时，不能继续消耗后续 task。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "raw.jsonl"
            client = MockClient([{"role": "assistant", "content": "stop"}])

            with self.assertRaises(CollectionInfrastructureError):
                collect_tasks(
                    [{"task_id": 1}, {"task_id": 2}],
                    client=client,
                    output_path=output,
                    base_url="http://shop.test",
                    env_factory=ReleaseFailingEnv,
                )

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual([row["task_id"] for row in rows], [1])
        self.assertEqual(rows[0]["status"], "environment_release_failed")
        self.assertEqual(rows[0]["release_error"]["type"], "OSError")

    def test_collect_tasks_stops_after_environment_resource_is_unavailable(self):
        """服务报告无可用环境时，不能把其余 task 误记成失败轨迹。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "raw.jsonl"
            client = MockClient([])

            with self.assertRaises(CollectionInfrastructureError):
                collect_tasks(
                    [{"task_id": 1}, {"task_id": 2}],
                    client=client,
                    output_path=output,
                    base_url="http://shop.test",
                    env_factory=UnavailableEnv,
                )

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual([row["task_id"] for row in rows], [1])
        self.assertEqual(rows[0]["error"]["type"], "ShopEnvironmentError")

    def test_collect_tasks_stops_after_model_api_is_unreachable(self):
        """模型 API 重试后仍断线时，不能把后续 task 批量记成失败。"""
        class DisconnectedClient:
            def complete(self, messages, tools):
                raise URLError("Remote end closed connection without response")

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "raw.jsonl"

            with self.assertRaises(CollectionInfrastructureError):
                collect_tasks(
                    [{"task_id": 1}, {"task_id": 2}],
                    client=DisconnectedClient(),
                    output_path=output,
                    base_url="http://shop.test",
                    env_factory=FakeEnv,
                )

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual([row["task_id"] for row in rows], [1])
        self.assertEqual(rows[0]["error"]["type"], "URLError")

    def test_collect_for_task_serializes_multiple_tool_calls_before_execution(self):
        client = MockClient(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{index}",
                            "type": "function",
                            "function": {
                                "name": "search_products",
                                "arguments": json.dumps({"query": f"乳胶枕{index}"}, ensure_ascii=False),
                            },
                        }
                        for index in range(3)
                    ],
                },
                assistant_tool("search_products", {"query": "第二次观察后搜索"}, "call_after_observation"),
            ]
        )
        env = NonTerminalEnv()

        traj = collect_for_task(
            {"task_id": 9},
            client=client,
            env_factory=lambda **kwargs: env,
            base_url="http://shop.test",
            max_steps=2,
        )

        self.assertEqual(traj["status"], "max_steps")
        self.assertEqual(len(traj["steps"]), 2)
        self.assertEqual(env.actions, ["search[乳胶枕0]", "search[第二次观察后搜索]"])
        self.assertEqual(len(traj["messages"][2]["tool_calls"]), 1)
        self.assertEqual(
            [call["id"] for call in traj["tool_call_truncations"][0]["dropped_tool_calls"]],
            ["call_1", "call_2"],
        )

    def test_load_tasks_reads_interaction_task_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tasks = Path(tmpdir) / "tasks.jsonl"
            tasks.write_text(
                json.dumps(
                    {
                        "prompt": [{"role": "user", "content": "hello"}],
                        "extra_info": {"interaction_kwargs": {"task_id": 42}},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rows = load_tasks(tasks)

        self.assertEqual(rows[0]["task_id"], 42)
        self.assertEqual(rows[0]["prompt"][0]["content"], "hello")

    def test_openai_client_sends_standard_tool_payload(self):
        captured = {}

        def transport(url, payload, headers, timeout):
            captured.update({"url": url, "payload": payload, "headers": headers, "timeout": timeout})
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        client = OpenAIChatClient(
            model="deepseek-chat",
            base_url="https://api.example.test/v1",
            api_key="secret",
            temperature=0.2,
            top_p=0.9,
            timeout=12,
            transport=transport,
        )

        message = client.complete([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])

        self.assertEqual(message["content"], "ok")
        self.assertEqual(captured["url"], "https://api.example.test/v1/chat/completions")
        self.assertEqual(captured["payload"]["model"], "deepseek-chat")
        self.assertEqual(captured["payload"]["tools"], [{"type": "function"}])
        self.assertEqual(captured["payload"]["max_tokens"], 512)
        self.assertEqual(captured["payload"]["temperature"], 0.2)
        self.assertEqual(captured["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(captured["headers"]["User-Agent"], "shopping-grpo-longhorizon/0.1")

    def test_openai_client_supports_responses_tool_payload_and_result(self):
        captured = {}

        def transport(url, payload, headers, timeout):
            captured.update({"url": url, "payload": payload})
            return {
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_search",
                        "name": "search_products",
                        "arguments": '{"query":"乳胶枕"}',
                    }
                ]
            }

        client = OpenAIChatClient(
            model="gpt-5.6-luna",
            base_url="https://opencode.ai/zen/go/v1/responses",
            api_key="secret",
            transport=transport,
        )
        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "买乳胶枕"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_old",
                        "type": "function",
                        "function": {
                            "name": "search_products",
                            "arguments": '{"query":"枕头"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_old",
                "name": "search_products",
                "content": "search result",
            },
        ]
        message = client.complete(
            messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "search_products",
                        "description": "search",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        )

        self.assertEqual(captured["url"], "https://opencode.ai/zen/go/v1/responses")
        self.assertEqual(captured["payload"]["max_output_tokens"], 512)
        self.assertEqual(captured["payload"]["tools"][0]["name"], "search_products")
        self.assertEqual(captured["payload"]["input"][-1]["type"], "function_call_output")
        self.assertEqual(captured["payload"]["input"][-1]["call_id"], "call_old")
        self.assertEqual(message["tool_calls"][0]["id"], "call_search")
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "search_products")

    def test_openai_client_allows_bounded_completion_override(self):
        """本地推理服务必须收到单次生成上限，避免无工具文本耗尽上下文。"""
        captured = {}

        def transport(url, payload, headers, timeout):
            captured.update({"payload": payload})
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        client = OpenAIChatClient(
            model="Qwen/Qwen3.5-2B",
            base_url="http://127.0.0.1:8000/v1",
            api_key="EMPTY",
            max_tokens=256,
            transport=transport,
        )

        client.complete([{"role": "user", "content": "hi"}], tools=[])

        self.assertEqual(captured["payload"]["max_tokens"], 256)

    def test_openai_client_compacts_old_complete_tool_groups_before_request(self):
        captured = {}

        def transport(url, payload, headers, timeout):
            captured.update({"payload": payload})
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "task"},
            assistant_tool("search_products", {"query": "old"}, "old"),
            {"role": "tool", "tool_call_id": "old", "name": "search_products", "content": "old page"},
            assistant_tool("search_products", {"query": "middle"}, "middle"),
            {"role": "tool", "tool_call_id": "middle", "name": "search_products", "content": "middle page"},
            assistant_tool("search_products", {"query": "latest"}, "latest"),
            {"role": "tool", "tool_call_id": "latest", "name": "search_products", "content": "latest page"},
        ]
        client = OpenAIChatClient(
            model="shopping",
            base_url="http://127.0.0.1:8000/v1",
            api_key="EMPTY",
            max_tokens=2,
            context_window=9,
            context_safety_margin=1,
            context_compaction_enable=True,
            token_counter=lambda candidate, tools: len(candidate),
            transport=transport,
        )

        client.complete(messages, tools=[])

        self.assertNotIn("old page", str(captured["payload"]["messages"]))
        self.assertIn("middle page", str(captured["payload"]["messages"]))
        self.assertIn("latest page", str(captured["payload"]["messages"]))
        self.assertEqual(client.last_context_event["removed_groups"], 1)
        self.assertEqual(messages[3]["content"], "old page")

    def test_openai_client_projects_tool_observation_with_serving_tokenizer(self):
        client = OpenAIChatClient(
            model="shopping",
            base_url="http://127.0.0.1:8000/v1",
            api_key="EMPTY",
            observation_token_budget=128,
            observation_generic_token_budget=128,
            observation_token_counter=len,
        )
        raw = (
            "Description " + "x" * 200
            + "\n\n搜索功能是否可用: False"
            + '\n\n可点击的按钮: ["back to search", "< prev"]'
        )

        visible, meta = client.project_observation("prev_page", raw, {})

        self.assertLessEqual(len(visible), 128)
        self.assertTrue(meta["truncated"])
        self.assertTrue(meta["critical_footer_preserved"])

    def test_openai_client_thinking_mode_keeps_reasoning_for_tool_follow_up(self):
        captured = {}

        def transport(url, payload, headers, timeout):
            captured.update({"payload": payload})
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": "先核对规格，再搜索。",
                            "tool_calls": [
                                {
                                    "id": "call_search",
                                    "type": "function",
                                    "function": {"name": "search_products", "arguments": '{"query":"乳胶枕"}'},
                                }
                            ],
                        }
                    }
                ]
            }

        client = OpenAIChatClient(
            model="deepseek-v4-flash",
            base_url="https://api.example.test/v1",
            api_key="secret",
            thinking=True,
            reasoning_effort="high",
            transport=transport,
        )

        message = client.complete([{"role": "user", "content": "买乳胶枕"}], tools=[{"type": "function"}])

        self.assertEqual(captured["payload"]["thinking"], {"type": "enabled"})
        self.assertEqual(captured["payload"]["reasoning_effort"], "high")
        self.assertNotIn("temperature", captured["payload"])
        self.assertNotIn("top_p", captured["payload"])
        self.assertEqual(message["reasoning_content"], "先核对规格，再搜索。")

    def test_openai_client_explicitly_disables_deepseek_v4_thinking(self):
        captured = {}

        def transport(url, payload, headers, timeout):
            captured.update({"payload": payload})
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        client = OpenAIChatClient(
            model="deepseek-v4-flash",
            base_url="https://opencode.ai/zen/go/v1",
            api_key="secret",
            thinking=False,
            reasoning_effort="max",
            transport=transport,
        )

        client.complete([{"role": "user", "content": "继续"}], tools=[])

        self.assertEqual(captured["payload"]["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", captured["payload"])
        self.assertEqual(captured["payload"]["temperature"], 0.0)
        self.assertEqual(captured["payload"]["top_p"], 1.0)

    def test_openai_client_does_not_send_thinking_to_local_non_deepseek_model(self):
        captured = {}

        def transport(url, payload, headers, timeout):
            captured.update({"payload": payload})
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        client = OpenAIChatClient(
            model="Qwen/Qwen3.5-2B",
            base_url="http://127.0.0.1:8000/v1",
            api_key="EMPTY",
            thinking=False,
            transport=transport,
        )

        client.complete([{"role": "user", "content": "继续"}], tools=[])

        self.assertNotIn("thinking", captured["payload"])
        self.assertNotIn("reasoning_effort", captured["payload"])

    def test_openai_client_retries_transient_disconnect_without_replaying_tools(self):
        attempts = []

        def transport(url, payload, headers, timeout):
            attempts.append(payload)
            if len(attempts) == 1:
                raise RemoteDisconnected("connection closed")
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        client = OpenAIChatClient(
            model="deepseek-v4-pro",
            base_url="https://api.example.test/v1",
            api_key="secret",
            transport=transport,
        )

        with patch("shopping_grpo.evaluation.rollout.time.sleep") as sleep:
            message = client.complete([{"role": "user", "content": "继续"}], tools=[])

        self.assertEqual(message["content"], "ok")
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0], attempts[1])
        sleep.assert_called_once()


if __name__ == "__main__":
    unittest.main()
