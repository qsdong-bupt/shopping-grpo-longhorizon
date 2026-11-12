import ast
import copy
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from shopping_grpo.environment.actions import action_guard_tool_message, action_reject_reason, clickable_buttons
from shopping_grpo.environment.observation import render_structured_observation
from shopping_grpo.environment.options import choices_from_observation, option_choices, resolve_option
from shopping_grpo.environment.projection import ObservationProjectionError, project_observation
from shopping_grpo.environment.protocol import message_protocol_problem
from shopping_grpo.environment.tools import tool_call_to_action
from shopping_grpo.evaluation.rollout import _execute_tool_call, _selection_feedback


ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "environments/ShopSimulator/shop_env/web_agent_site"


def state(options=None):
    options = options or {"尺码": ["S", "M"], "颜色": ['粉色 【双 空格  规格】', '[型号]黑色']}
    return {
        "observation_version": "shopping-observation-v3", "page_type": "product_detail",
        "product": {"asin": "818249730776", "title": "测试商品", "price": "89 to 98"},
        "available_options": options, "selected_options": {}, "search_available": False,
        "actions": ["< prev", "buy now", *[v for values in options.values() for v in values]],
    }


def function_from_source(path, name, globals_):
    """Run real parser/environment methods without loading product data or a GPU."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(globals_)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def test_ids_cover_multiple_axes_and_keep_exact_raw_text():
    s = state()
    obs = render_structured_observation(s)
    choices = choices_from_observation(obs)
    assert [c["option_id"] for c in choices] == ["opt_1", "opt_2", "opt_3", "opt_4"]
    assert resolve_option("opt_3", obs)["value"] == '粉色 【双 空格  规格】'
    assert clickable_buttons(obs) == ["< prev", "buy now", "opt_1", "opt_2", "opt_3", "opt_4"]
    assert all(action_reject_reason("select_option", {"option_id": c["option_id"]}, obs) is None for c in choices)
    s["selected_options"] = {"尺码": "M"}
    after = option_choices(s)
    assert [(c["option_id"], c["axis"], c["value"]) for c in after] == [(c["option_id"], c["axis"], c["value"]) for c in choices]
    assert after[1]["selected"] is True


@pytest.mark.parametrize("bad", ["粉色", "opt_999", "OPT_1", "opt_01", "opt_1 ", 1, None, []])
def test_invalid_ids_never_execute(bad):
    obs = render_structured_observation(state())
    assert action_reject_reason("select_option", {"option_id": bad}, obs)


def test_legacy_free_text_arguments_rejected_with_actionable_feedback():
    obs = render_structured_observation(state())
    reason = action_reject_reason("select_option", {"value": "粉色"}, obs)
    assert reason == "schema_extra_arguments:value"
    msg = action_guard_tool_message({"id": "call", "function": {"name": "select_option"}}, reason, obs)
    assert "当前规格选项" in msg["content"]
    assert '"option_id": "opt_1"' in msg["content"]
    assert "prev_page({})" not in msg["content"]
    assert message_protocol_problem([{"tool_calls": [{"function": {"name": "select_option", "arguments": '{"value":"白"}'}}]}]) == "unsupported_option_arguments"


def test_options_unavailable_on_other_pages_or_after_removal():
    assert action_reject_reason("select_option", {"option_id": "opt_1"}, "page_type: search_results\n") == "select_option_not_available_on_current_page"
    s = state({"颜色": ["白", "黑"]})
    s["available_options"] = {"颜色": ["白"]}
    assert action_reject_reason("select_option", {"option_id": "opt_2"}, render_structured_observation(s)) == "option_id_not_on_current_page"


def test_duplicate_labels_on_different_axes_resolve_by_axis():
    obs = render_structured_observation(state({"尺寸": ["标准"], "套餐": ["标准"]}))
    choices = choices_from_observation(obs)
    actions = [tool_call_to_action("select_option", {"option_id": c["option_id"]}, obs) for c in choices]
    assert actions[0] != actions[1]
    assert {json.loads(a[len("click_option["):-1])["axis"] for a in actions} == {"尺寸", "套餐"}


def test_projection_preserves_complete_mapping_or_fails_closed():
    obs = render_structured_observation(state())
    projected, _ = project_observation("open_product", obs, count_tokens=len, detail_token_budget=4096)
    assert choices_from_observation(projected) == choices_from_observation(obs)
    with pytest.raises(ObservationProjectionError):
        project_observation("open_product", obs, count_tokens=len, detail_token_budget=128)


@pytest.mark.parametrize("value", ['[hm3690-001]黑', '型号[a[b]]', '带"引号"\\反斜杠', '双  空格'])
def test_actual_action_parser_round_trips_special_characters(value):
    parse = function_from_source(ENV / "engine/engine.py", "parse_action", {"re": re})
    assert parse(f"click[{value}]") == ("click", value)
    obs = render_structured_observation(state({"颜色": [value]}))
    name, arg = parse(tool_call_to_action("select_option", {"option_id": "opt_1"}, obs))
    assert name == "click_option"
    assert json.loads(arg) == {"axis": "颜色", "value": value}


def make_execution_env(s):
    """Exercise actual get_available_actions, step and parser with a tiny browser."""
    from bs4 import BeautifulSoup
    parse = function_from_source(ENV / "engine/engine.py", "parse_action", {"re": re})
    get_actions = function_from_source(ENV / "envs/web_agent_text_env.py", "get_available_actions", {})
    env_step = function_from_source(ENV / "envs/web_agent_text_env.py", "step", {"json": json, "parse_action": parse})
    soup = BeautifulSoup("<html></html>", "html.parser")
    for axis, values in s["available_options"].items():
        for value in values:
            tag = soup.new_tag("input", attrs={"type": "radio", "name": axis, "value": value})
            soup.html.append(tag)
    env = SimpleNamespace(session="test", prev_actions=[], prev_obs=[], num_prev_actions=0, num_prev_obs=0, observation="detail")
    env._parse_html = lambda: soup
    env.get_available_actions = lambda: get_actions(env)
    def click(value, mapping):
        tag = mapping[value]
        s["selected_options"][tag["name"]] = tag["value"]
        return {"reward": 0.0, "done": False}
    env.browser = SimpleNamespace(click=click, current_url="detail")
    env.server = SimpleNamespace(visible_asins=lambda *a: [], record_progress=lambda *a, **kw: {"termination_reason": None})
    env.step = lambda action: env_step(env, action)
    return env


def test_actual_environment_selects_two_axes_and_replaces_only_same_axis():
    s = state({"尺寸": ["标准", "大号"], "套餐": ["标准", '[套装]  双空格']})
    env = make_execution_env(s)
    for axis, value in [("尺寸", "标准"), ("套餐", "标准"), ("尺寸", "大号"), ("套餐", '[套装]  双空格')]:
        obs = render_structured_observation(s)
        choice = next(c for c in choices_from_observation(obs) if (c["axis"], c["value"]) == (axis, value))
        env.step(tool_call_to_action("select_option", {"option_id": choice["option_id"]}, obs))
    assert s["selected_options"] == {"尺寸": "大号", "套餐": '[套装]  双空格'}


def test_rollout_audits_resolved_axis_and_summarizes_returned_selection():
    s = state()
    previous = copy.deepcopy(s)
    env = make_execution_env(s)
    def execute(action):
        env.step(action)
        return {"observation_state": s, "reward": 0.0, "done": False}
    call = {"id": "test", "function": {"name": "select_option", "arguments": '{"option_id":"opt_3"}'}}
    step = _execute_tool_call(SimpleNamespace(step=execute), call, 0, render_structured_observation(s))
    assert step["parameters"] == {"option_id": "opt_3"}
    assert step["resolved_option"]["value"] == '粉色 【双 空格  规格】'
    assert "- 颜色：粉色 【双 空格  规格】" in _selection_feedback(previous, step)
