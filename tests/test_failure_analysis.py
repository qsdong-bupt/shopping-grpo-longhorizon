from copy import deepcopy

import pytest

from shopping_grpo.evaluation.failure_analysis import analyze_trajectory, summarize_failure_analyses


def trajectory(reward_type="gold_purchase", task_id=1):
    return {
        "task_id": task_id, "status": "done", "done": True, "steps": [],
        "initial_result": {"observation_state": {"page_type": "search_home"}},
        "blocked_tool_calls": [],
        "terminal_result": {
            "done": True, "over": True, "termination_reason": reward_type,
            "reward_detail": {"reward_version": "shopsimulator-reward-v3",
                              "reward_type": reward_type, "termination_reason": reward_type,
                              "reward_valid": True, "purchase_success": reward_type in {"gold_purchase", "valid_alternative_purchase"}},
        },
    }


def product(selected, asin="123456789012"):
    return {"page_type": "product_detail", "product": {"asin": asin}, "selected_options": selected}


def results_page(page, *asins):
    return {"page_type": "search_results", "query": "query", "page": page,
            "products": [{"asin": asin} for asin in asins]}


def step(tool, state, **progress):
    return {"tool_name": tool, "parameters": {},
            "tool_call": {"function": {"name": tool, "arguments": "{}"}},
            "result": {"observation_state": state, "progress": progress}}


def guard(tool="select_option", reason="invalid_option_id", position=0):
    return {"step_index": position, "reason": reason,
            "tool_call": {"function": {"name": tool, "arguments": '{"option_id":"bad"}'}}}


def test_recovered_guard_does_not_make_success_a_failure():
    row = trajectory()
    row["blocked_tool_calls"] = [guard(), guard()]
    result = analyze_trajectory(row)
    assert result["final_category"] == "success"
    assert result["issue_event_counts"]["action_parameter_error"] == 2
    assert result["attribution_status"] == "not_applicable"
    summary = summarize_failure_analyses([result])
    assert summary["issue_counts"][0]["task_count"] == 1
    assert summary["issue_counts"][0]["event_count"] == 2


@pytest.mark.parametrize("reason", ["schema_missing_arguments:option_id", "schema_extra_arguments:value", "invalid_option_id", "option_id_not_on_current_page"])
def test_parameter_reason_is_not_overwritten_by_page_diagnosis(reason):
    row = trajectory()
    row["blocked_tool_calls"] = [guard(reason=reason)]
    result = analyze_trajectory(row)
    assert result["issue_tags"] == ["action_parameter_error"]
    assert result["evidence"][0]["reason"] == reason


@pytest.mark.parametrize("page,tool,expected", [
    ("product_detail", "open_product", "page_tool_mismatch"),
    ("search_results", "open_product", "action_target_not_visible"),
    ("product_detail", "select_option", "action_target_not_visible"),
    (None, "open_product", "action_target_not_visible"),
])
def test_legacy_guard_reason_is_not_assumed_to_be_bad_parameter(page, tool, expected):
    row = trajectory()
    row["initial_result"]["observation_state"] = {"page_type": page}
    row["blocked_tool_calls"] = [guard(tool=tool, reason="click_not_in_previous_observation")]
    assert analyze_trajectory(row)["issue_tags"] == [expected]


def test_missing_guard_position_does_not_invent_page_or_step():
    row = trajectory()
    row["blocked_tool_calls"] = [guard(tool="open_product", reason="click_not_in_previous_observation", position=-1)]
    event = analyze_trajectory(row)["evidence"][0]
    assert event["page_type"] is None
    assert event["step_index"] is None


def test_confirmed_requirement_failure_keeps_comparator_evidence():
    row = trajectory("wrong_purchase")
    check = {"status": "fail", "required": 100, "actual": 120, "source_field": "variant_price"}
    row["terminal_result"]["reward_detail"]["hard_gates"] = {"budget": check}
    result = analyze_trajectory(row)
    assert result["final_category"] == "requirement_mismatch"
    assert result["attribution_status"] == "evidence_supported"
    assert result["evidence"][0]["check"] == check


def test_low_score_without_comparator_is_not_specific_requirement_evidence():
    row = trajectory("partial_alternative_purchase")
    row["terminal_result"]["reward_detail"]["weighted_score"] = .2
    result = analyze_trajectory(row)
    assert result["final_category"] == "requirement_mismatch"
    assert result["attribution_status"] == "needs_review"
    assert "requirement_check_failed" not in result["issue_tags"]


def test_price_unverifiable_is_not_purchase_mistake():
    row = trajectory("reward_unverifiable")
    row["terminal_result"]["reward_detail"]["reward_valid"] = False
    row["terminal_result"]["reward_detail"]["evidence"] = {
        "price_resolution": {"status": "unverifiable", "method": "multiple_effective_price_axes"}
    }
    result = analyze_trajectory(row)
    assert result["final_category"] == "reward_unverifiable"
    assert result["issue_tags"] == ["reward_evidence_unverifiable"]
    assert result["evidence"][0]["check"]["method"] == "multiple_effective_price_axes"


def test_terminal_service_error_has_priority_but_preserves_reward_fact():
    row = trajectory()
    row["release_error"] = {"type": "ShopEnvironmentError", "message": "release failed"}
    result = analyze_trajectory(row)
    assert result["final_category"] == "infrastructure_failure"
    assert result["outcome"] == "gold_purchase"


def test_recovered_step_error_is_a_tag_not_terminal_infrastructure_failure():
    row = trajectory()
    row["steps"] = [step("open_product", product({}))]
    row["steps"][0]["error"] = {"type": "ShopEnvironmentError", "message": "temporary"}
    result = analyze_trajectory(row)
    assert result["final_category"] == "success"
    assert "environment_or_service_error" in result["issue_tags"]


def test_unrecognized_exception_stays_unclassified():
    row = trajectory()
    row.update(status="error", done=False, terminal_result={}, error={"type": "ValueError", "message": "bug"})
    result = analyze_trajectory(row)
    assert result["final_category"] == "review_required"
    assert result["issue_tags"] == ["execution_error_unclassified"]


def test_incomplete_gold_terminal_does_not_become_success():
    row = trajectory()
    row["terminal_result"]["over"] = False
    result = analyze_trajectory(row)
    assert result["final_category"] == "review_required"
    assert "terminal_contract_incomplete" in result["issue_tags"]


def test_max_steps_does_not_infer_failure_to_find_or_buy():
    result = analyze_trajectory(trajectory("max_steps"))
    assert result["final_category"] == "review_required"
    assert result["issue_tags"] == ["max_steps_reached"]


def test_invalid_action_limit_keeps_independent_guard_tags():
    row = trajectory()
    row.update(status="invalid_action_limit", done=False, terminal_result={})
    row["blocked_tool_calls"] = [guard()]
    result = analyze_trajectory(row)
    assert result["outcome"] == "invalid_action_limit"
    assert result["final_category"] == "action_limit"
    assert result["issue_tags"] == ["action_parameter_error", "invalid_action_limit"]


def test_unchanged_option_is_observable_repetition():
    row = trajectory("repeat_loop")
    row["initial_result"]["observation_state"] = product({"颜色": "白"})
    row["steps"] = [step("select_option", product({"颜色": "白"})) for _ in range(2)]
    row["terminal_result"]["progress"] = {"consecutive_repeats": 2, "no_progress_steps": 2, "runtime_progress_added": []}
    result = analyze_trajectory(row)
    assert result["loop_assessment"] == "repeated_unchanged_state"
    assert result["termination_trigger"] == "repeat_loop"
    assert result["trigger_counters"]["consecutive_repeats"] == 2
    assert result["issue_event_counts"]["repeated_selected_option"] == 2


def test_new_product_or_other_axis_selection_is_not_repeated_option():
    row = trajectory()
    row["initial_result"]["observation_state"] = product({"颜色": "白"})
    row["steps"] = [step("select_option", product({"颜色": "白", "尺码": "S"})),
                    step("select_option", product({"颜色": "白"}, asin="999999999999"))]
    assert "repeated_selected_option" not in analyze_trajectory(row)["issue_tags"]


def test_switching_back_to_known_combination_is_distinct_from_unchanged():
    row = trajectory()
    row["initial_result"]["observation_state"] = product({"尺码": "S"})
    row["steps"] = [step("select_option", product({"尺码": "M"})), step("select_option", product({"尺码": "S"}))]
    result = analyze_trajectory(row)
    assert result["issue_tags"] == ["option_oscillation"]
    assert result["evidence_steps"] == [1]


def test_return_after_option_oscillation_needs_review():
    row = trajectory("repeat_loop")
    row["initial_result"]["observation_state"] = product({"尺码": "S"})
    row["steps"] = [step("select_option", product({"尺码": "M"})),
                    step("select_option", product({"尺码": "S"})),
                    step("select_option", product({"尺码": "S"})),
                    step("prev_page", results_page(1, "123456789012"))]
    row["terminal_result"]["progress"] = {"no_progress_steps": 4, "runtime_progress_added": []}
    result = analyze_trajectory(row)
    assert result["attribution_status"] == "needs_review"
    assert "navigation_transition_terminated" in result["issue_tags"]


def test_small_new_page_terminated_is_not_labelled_confirmed_loop():
    row = trajectory("repeat_loop")
    row["steps"] = [step("next_page", results_page(2, "123456789012"), new_asin_count=1, runtime_progress_added=[])]
    row["terminal_result"]["progress"] = {"no_progress_steps": 4, "new_asin_count": 1, "runtime_progress_added": []}
    result = analyze_trajectory(row)
    assert result["loop_assessment"] == "progress_at_termination"
    assert "low_yield_discovery" in result["issue_tags"]
    assert "progress_at_loop_termination" in result["issue_tags"]


def test_missing_state_or_progress_does_not_prove_loop():
    result = analyze_trajectory(trajectory("repeat_loop"))
    assert result["loop_assessment"] == "insufficient_evidence"
    assert result["attribution_status"] == "needs_review"


def test_input_immutable_and_summary_exclusive_with_empty_categories():
    row = trajectory("wrong_purchase")
    original = deepcopy(row)
    analyses = [analyze_trajectory(row), analyze_trajectory(trajectory(task_id=2))]
    assert row == original
    summary = summarize_failure_analyses(analyses)
    assert sum(summary["category_counts"].values()) == summary["tasks"] == 2
    assert summarize_failure_analyses([])["tasks"] == 0
    with pytest.raises(ValueError, match="Duplicate task IDs"):
        summarize_failure_analyses([analyses[0], analyses[0]])
