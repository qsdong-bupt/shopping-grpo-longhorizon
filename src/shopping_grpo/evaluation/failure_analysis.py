"""Deterministic post-run diagnostics; observed problems are not inferred causes.

One final category is exclusive. Issue tags can overlap, including on successful
tasks. Evidence uses zero-based source step indexes and never changes rewards,
agent messages, or runtime termination decisions.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import json

from shopping_grpo.evaluation.metrics import compute_deterministic_metrics
from shopping_grpo.evaluation.trajectory import normalize_trajectory


FAILURE_ANALYSIS_VERSION = "shopping-failure-analysis-v1"
CATEGORY_LABELS = {
    "success": "成功购买",
    "requirement_mismatch": "需求匹配未通过",
    "loop_termination": "循环规则终止",
    "action_limit": "连续动作被拒绝终止",
    "infrastructure_failure": "环境或服务故障",
    "reward_unverifiable": "奖励不可验证",
    "review_required": "其他 / 待复核",
}
ISSUE_LABELS = {
    "requirement_check_failed": "需求检查未通过",
    "reward_evidence_unverifiable": "奖励证据不可验证",
    "action_parameter_error": "工具参数错误",
    "page_tool_mismatch": "工具不适用当前页面",
    "action_target_not_visible": "目标不在当前页面（原因待查）",
    "invalid_tool_call": "工具调用不合法",
    "guard_rejection_unclassified": "其他守卫拒绝",
    "environment_or_service_error": "环境或服务异常",
    "execution_error_unclassified": "执行异常（类型待查）",
    "context_budget_exceeded": "上下文预算超限",
    "repeated_selected_option": "重复选择已选规格",
    "option_oscillation": "切换回已试规格组合",
    "loop_rule_triggered": "循环规则触发",
    "progress_at_loop_termination": "有新进展时触发循环终止",
    "navigation_transition_terminated": "页面导航中触发循环终止",
    "low_yield_discovery": "发现少量新商品但未计进展",
    "max_steps_reached": "达到步数上限",
    "invalid_action_limit": "连续拒绝达到上限",
    "terminal_contract_incomplete": "终局记录不完整或不一致",
}
_SERVICE_ERRORS = {
    "ShopEnvironmentError", "ShopHttpError", "TimeoutError", "ReadTimeout",
    "ConnectTimeout", "RemoteDisconnected", "URLError", "ConnectionError",
    "ConnectionResetError", "HTTPError", "APIConnectionError", "APITimeoutError",
}
_PAGE_TOOLS = {
    "search_home": {"search_products"},
    "search_results": {"open_product", "next_page", "prev_page", "back_to_search"},
    "product_detail": {"select_option", "buy_now", "prev_page", "back_to_search"},
}
_NAVIGATION = {"next_page", "prev_page", "back_to_search"}


def _mapping(value):
    return value if isinstance(value, Mapping) else {}


def _state(result):
    return _mapping(_mapping(result).get("observation_state"))


def _state_key(state):
    """Only public state, with page context so page changes are not ignored."""
    page = state.get("page_type")
    if page == "search_results":
        products = state.get("products")
        if not isinstance(products, list):
            return None
        payload = [page, state.get("normalized_query") or state.get("query"),
                   state.get("page"), [str(_mapping(p).get("asin")) for p in products]]
    elif page == "product_detail":
        asin = _mapping(state.get("product")).get("asin")
        if not asin or not isinstance(state.get("selected_options"), Mapping):
            return None
        payload = [page, str(asin), dict(state["selected_options"]), state.get("selected_price"),
                   _mapping(state.get("product")).get("price")]
    elif page == "search_home":
        payload = [page]
    else:
        return None
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _guard_tag(reason, tool, state):
    if reason.startswith("schema_") or reason in {"invalid_option_id", "option_id_not_on_current_page"}:
        return "action_parameter_error"
    if reason in {"search_not_available_on_current_page", "select_option_not_available_on_current_page"}:
        return "page_tool_mismatch"
    allowed = _PAGE_TOOLS.get(state.get("page_type"))
    if reason == "click_not_in_previous_observation":
        return "page_tool_mismatch" if allowed is not None and tool not in allowed else "action_target_not_visible"
    if reason == "unknown_or_invalid_tool":
        return "invalid_tool_call"
    return "guard_rejection_unclassified"


def analyze_trajectory(trajectory: Mapping, *, metrics: Mapping | None = None) -> dict:
    """Analyze one raw collector record without a model/Judge call or mutation."""
    if not isinstance(trajectory, Mapping):
        raise TypeError("trajectory must be an object")
    if metrics is None:
        metrics = compute_deterministic_metrics(normalize_trajectory(trajectory))
    outcome_metrics = _mapping(metrics.get("reward_and_outcome"))
    validity = _mapping(metrics.get("validity"))
    terminal = _mapping(trajectory.get("terminal_result"))
    detail = _mapping(terminal.get("reward_detail"))
    reward_type = str(outcome_metrics.get("reward_type") or "unknown")
    outcome = reward_type if reward_type != "unknown" else str(trajectory.get("status") or "unknown")
    steps = [step if isinstance(step, Mapping) else {} for step in trajectory.get("steps") or []]
    events = []

    def add(tag, *, source, step_index=None, **evidence):
        events.append({"tag": tag, "source": source, "step_index": step_index, **evidence})

    # Keep original guard reasons; the same legacy reason can mean either a
    # page error or an unavailable target. Do not guess whitespace mistakes.
    for index, blocked in enumerate(trajectory.get("blocked_tool_calls") or []):
        if not isinstance(blocked, Mapping):
            continue
        position = blocked.get("step_index")
        known_position = isinstance(position, int) and not isinstance(position, bool) and 0 <= position <= len(steps)
        before = _state(steps[position - 1].get("result")) if known_position and position else (
            _state(trajectory.get("initial_result")) if known_position else {}
        )
        function = _mapping(_mapping(blocked.get("tool_call")).get("function"))
        tool = function.get("name") or blocked.get("tool_name") or "unknown"
        reason = str(blocked.get("reason") or "unknown")
        add(_guard_tag(reason, tool, before), source=f"blocked_tool_calls[{index}]",
            step_index=position if known_position else None, tool_name=tool,
            reason=reason, page_type=before.get("page_type"), arguments=function.get("arguments"))

    for field in ("error", "release_error"):
        error = _mapping(trajectory.get(field))
        if not error:
            continue
        error_type = str(error.get("type") or "unknown")
        tag = "context_budget_exceeded" if error_type == "ContextBudgetError" else (
            "environment_or_service_error" if error_type in _SERVICE_ERRORS else "execution_error_unclassified"
        )
        add(tag, source=field, error_type=error_type, message=error.get("message"))
    for index, step in enumerate(steps):
        error = _mapping(step.get("error"))
        if error:
            error_type = str(error.get("type") or "unknown")
            add("environment_or_service_error" if error_type in _SERVICE_ERRORS else "execution_error_unclassified",
                source=f"steps[{index}].error", step_index=index,
                error_type=error_type, message=error.get("message"))

    # A score below one alone is not an independently established mismatch.
    # Extract concrete comparator failures/unknowns and preserve requirements.
    def inspect_check(check, source, dimension):
        check = _mapping(check)
        status = check.get("status")
        tag = {"fail": "requirement_check_failed", "unverifiable": "reward_evidence_unverifiable"}.get(status)
        if tag:
            add(tag, source=source, dimension=dimension, check=dict(check))

    for dimension, check in _mapping(detail.get("hard_gates")).items():
        inspect_check(check, f"terminal_result.reward_detail.hard_gates.{dimension}", dimension)
    preferences = _mapping(_mapping(detail.get("evidence")).get("preference_scoring"))
    for dimension, value in _mapping(preferences.get("dimensions")).items():
        for index, check in enumerate(_mapping(value).get("results") or []):
            inspect_check(check, f"terminal_result.reward_detail.evidence.preference_scoring.dimensions.{dimension}.results[{index}]", dimension)
    resolution = _mapping(_mapping(detail.get("evidence")).get("price_resolution"))
    # Budget hard gates often already contain exactly this resolution.
    if resolution.get("status") == "unverifiable" and not any(
        event["tag"] == "reward_evidence_unverifiable" and event.get("dimension") == "budget" for event in events
    ):
        add("reward_evidence_unverifiable", source="terminal_result.reward_detail.evidence.price_resolution",
            dimension="price", check=dict(resolution))

    previous = _state(trajectory.get("initial_result"))
    state_keys = [_state_key(previous)]
    option_states = {}
    for index, step in enumerate(steps):
        current = _state(step.get("result"))
        key = _state_key(current)
        state_keys.append(key)
        asin = _mapping(current.get("product")).get("asin")
        previous_asin = _mapping(previous.get("product")).get("asin")
        selected = current.get("selected_options")
        old_selected = previous.get("selected_options")
        if current.get("page_type") == "product_detail" and asin and isinstance(selected, Mapping):
            seen = option_states.setdefault(str(asin), set())
            if previous_asin == asin and isinstance(old_selected, Mapping):
                seen.add(json.dumps(dict(old_selected), sort_keys=True, ensure_ascii=False))
            option_key = json.dumps(dict(selected), sort_keys=True, ensure_ascii=False)
            if step.get("tool_name") == "select_option" and selected and previous_asin == asin:
                if old_selected == selected:
                    add("repeated_selected_option", source=f"steps[{index}]", step_index=index,
                        asin=asin, selected_options=dict(selected))
                elif option_key in seen:
                    add("option_oscillation", source=f"steps[{index}]", step_index=index,
                        asin=asin, before=old_selected, after=dict(selected))
            seen.add(option_key)
        progress = _mapping(_mapping(step.get("result")).get("progress"))
        new_asins = progress.get("new_asin_count")
        if (step.get("tool_name") in {"search_products", "next_page", "prev_page"}
                and isinstance(new_asins, int) and 0 < new_asins < 3
                and progress.get("runtime_progress_added") == []):
            add("low_yield_discovery", source=f"steps[{index}].result.progress", step_index=index,
                new_asin_count=new_asins, page_type=current.get("page_type"))
        previous = current

    loop_assessment = None
    trigger_counters = {}
    termination_trigger = terminal.get("termination_reason") or outcome_metrics.get("termination_reason")
    if outcome_metrics.get("repeat_loop_termination"):
        last_index = len(steps) - 1 if steps else None
        last = steps[-1] if steps else {}
        progress = _mapping(terminal.get("progress")) or _mapping(_mapping(last.get("result")).get("progress"))
        add("loop_rule_triggered", source="terminal_result.progress", step_index=last_index,
            progress=dict(progress))
        # Historical progress omits configured thresholds and the exact branch.
        # Preserve observable counters; do not invent which threshold fired.
        termination_trigger = progress.get("termination_trigger") or "repeat_loop"
        trigger_counters = {key: progress[key] for key in ("consecutive_repeats", "no_progress_steps", "step_count") if key in progress}
        no_progress = progress.get("no_progress_steps")
        last_tool = last.get("tool_name")
        discovery = last_tool in {"next_page", "prev_page", "search_products"}
        new_asins = progress.get("new_asin_count")
        new_discovery = discovery and isinstance(new_asins, int) and new_asins > 0
        if progress.get("runtime_progress_added") or new_discovery:
            loop_assessment = "progress_at_termination"
            add("progress_at_loop_termination", source="terminal_result.progress", step_index=last_index,
                new_asin_count=new_asins, runtime_progress_added=progress.get("runtime_progress_added"))
        elif len(state_keys) >= 3 and state_keys[-1] is not None and state_keys[-1] == state_keys[-2] == state_keys[-3]:
            loop_assessment = "repeated_unchanged_state"
        elif isinstance(no_progress, int) and no_progress > 0:
            window = state_keys[-(no_progress + 1):]
            if len(window) == no_progress + 1 and all(key is not None for key in window):
                loop_assessment = "revisited_state_cycle" if len(window) != len(set(window)) else "state_changed_without_new_evidence"
            else:
                loop_assessment = "insufficient_evidence"
        else:
            loop_assessment = "insufficient_evidence"
        if (last_tool in _NAVIGATION and len(state_keys) >= 2
                and state_keys[-1] is not None and state_keys[-2] is not None
                and state_keys[-1] != state_keys[-2]):
            add("navigation_transition_terminated", source=f"steps[{last_index}]", step_index=last_index,
                tool_name=last_tool, page_type=_state(last.get("result")).get("page_type"))

    if outcome_metrics.get("max_steps_termination"):
        add("max_steps_reached", source="terminal_result", step_index=len(steps) - 1 if steps else None)
    if trajectory.get("status") == "invalid_action_limit":
        add("invalid_action_limit", source="status")
    if validity.get("trajectory_contract_issues"):
        add("terminal_contract_incomplete", source="metrics.validity",
            issues=validity["trajectory_contract_issues"])

    tags = sorted({event["tag"] for event in events})
    infrastructure = bool(validity.get("infrastructure_invalid") or any(
        _mapping(trajectory.get(field)).get("type") in _SERVICE_ERRORS for field in ("error", "release_error")
    ))
    complete = (trajectory.get("done") is True and terminal.get("done") is True and terminal.get("over") is True)
    if infrastructure:
        category = "infrastructure_failure"
    elif reward_type == "reward_unverifiable":
        category = "reward_unverifiable"
    elif not complete or validity.get("trajectory_contract_issues"):
        category = "action_limit" if trajectory.get("status") == "invalid_action_limit" else "review_required"
    elif outcome_metrics.get("strict_gold_success") or (
        reward_type == "valid_alternative_purchase" and detail.get("reward_valid") is True
        and detail.get("purchase_success") is True
    ):
        category = "success"
    elif reward_type in {"wrong_purchase", "partial_alternative_purchase"} and detail.get("reward_valid") is True:
        category = "requirement_mismatch"
    elif outcome_metrics.get("repeat_loop_termination"):
        category = "loop_termination"
    else:
        category = "review_required"

    attribution = "needs_review"
    if category == "success":
        attribution = "not_applicable"
    elif category == "infrastructure_failure" and "environment_or_service_error" in tags:
        attribution = "evidence_supported"
    elif category == "requirement_mismatch" and "requirement_check_failed" in tags:
        attribution = "evidence_supported"
    elif category == "reward_unverifiable" and "reward_evidence_unverifiable" in tags:
        attribution = "evidence_supported"
    elif category == "loop_termination" and loop_assessment in {"repeated_unchanged_state", "revisited_state_cycle"}:
        attribution = "evidence_supported"
    if "navigation_transition_terminated" in tags or "progress_at_loop_termination" in tags:
        attribution = "needs_review"
    return {
        "schema_version": FAILURE_ANALYSIS_VERSION,
        "task_id": trajectory.get("task_id"),
        "trajectory_id": trajectory.get("trajectory_id"),
        "outcome": outcome,
        "final_category": category,
        "termination_trigger": termination_trigger,
        "trigger_counters": trigger_counters,
        "loop_assessment": loop_assessment,
        "issue_tags": tags,
        "issue_event_counts": dict(sorted(Counter(event["tag"] for event in events).items())),
        "evidence_steps": sorted({event["step_index"] for event in events if event.get("step_index") is not None}),
        "evidence": events,
        "attribution_status": attribution,
    }


def summarize_failure_analyses(analyses) -> dict:
    """Count unique tasks and events separately. Duplicate tasks require review."""
    rows = list(analyses)
    ids = [str(row["task_id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate task IDs; select one attempt per task before failure analysis")
    categories = Counter(row["final_category"] for row in rows)
    task_counts = Counter(tag for row in rows for tag in set(row["issue_tags"]))
    event_counts = Counter()
    for row in rows:
        event_counts.update(row["issue_event_counts"])
    return {
        "schema_version": FAILURE_ANALYSIS_VERSION,
        "tasks": len(rows),
        "category_counts": {key: categories[key] for key in CATEGORY_LABELS},
        "attribution_counts": dict(sorted(Counter(row["attribution_status"] for row in rows).items())),
        "issue_counts": [
            {"tag": tag, "label": ISSUE_LABELS.get(tag, tag), "task_count": task_counts[tag], "event_count": event_counts[tag]}
            for tag in sorted(task_counts, key=lambda tag: (-task_counts[tag], tag))
        ],
        "loop_assessment_counts": dict(sorted(Counter(row["loop_assessment"] for row in rows if row["loop_assessment"]).items())),
        "notes": [
            "最终类别互斥；过程标签可重叠，包括成功任务。标签任务数不可直接相加。",
            "evidence_supported 表示有规则证据，不代表证明模型内部原因或修复后的收益。",
            "step_index 与 evidence_steps 均为从0开始的原始执行步骤索引；守卫事件另存原始列表位置。",
        ],
    }
