"""根据最新 observation 守卫模型的工具调用。

ShopSimulator 的按钮和商品只在当前页面有效。动作守卫在请求到达环境前检查
这一点，并把拒绝原因反馈给模型，避免非法点击污染环境状态。
"""

import json
import re

from shopping_grpo.environment.product_id import PRODUCT_ID_CAPTURE
from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS, tool_call_to_action


RUNTIME_GUARD_FIELD = "runtime_action_guard"
NAVIGATION_BUTTONS = {
    "description",
    "features",
    "reviews",
    "attributes",
    "next >",
    "< prev",
    "back to search",
    "buy now",
}
TOOL_ARGUMENT_NAMES = {
    tool["function"]["name"]: set(tool["function"]["parameters"].get("properties", {}))
    for tool in SHOP_TOOL_SCHEMAS
}
TOOL_REQUIRED_ARGUMENT_NAMES = {
    tool["function"]["name"]: set(tool["function"]["parameters"].get("required", []))
    for tool in SHOP_TOOL_SCHEMAS
}


def action_reject_reason(name, arguments, observation):
    """返回动作拒绝原因；``None`` 表示允许执行。

检查顺序很重要：先拦截 schema 外字段，再处理无需页面状态的动作，最后只允许
点击最新 observation 中仍然存在的目标。
"""
    if not isinstance(arguments, dict):
        return "schema_arguments_not_object"
    if name not in TOOL_ARGUMENT_NAMES:
        return "unknown_or_invalid_tool"
    extra_argument_names = _schema_extra_argument_names(name, arguments)
    if extra_argument_names:
        return "schema_extra_arguments:" + ",".join(extra_argument_names)
    missing_argument_names = _schema_missing_argument_names(name, arguments)
    if missing_argument_names:
        return "schema_missing_arguments:" + ",".join(missing_argument_names)
    if name == "search_products":
        if "搜索功能是否可用: False" in observation:
            return "search_not_available_on_current_page"
        return None
    if not observation:
        return "missing_previous_observation"
    if name == "select_option":
        from shopping_grpo.environment.options import OptionResolutionError, resolve_option
        try:
            choice = resolve_option(arguments.get("option_id"), observation)
        except OptionResolutionError as exc:
            return str(exc)
        if choice["option_id"] not in clickable_buttons(observation):
            return "option_id_not_on_current_page"
        return None
    if name == "open_product":
        asin = str(arguments.get("asin", ""))
        if asin not in product_ids(observation):
            return "click_not_in_previous_observation"
        current_buttons = {
            button.casefold() for button in clickable_buttons(observation)
        }
        if asin.casefold() not in current_buttons:
            return "click_not_in_previous_observation"
        return None
    try:
        action = tool_call_to_action(name, arguments)
    except Exception:
        return "unknown_or_invalid_tool"
    if not isinstance(action, str) or not action.startswith("click[") or not action.endswith("]"):
        return "click_not_in_previous_observation"
    target = action[6:-1].casefold()
    if target not in {button.casefold() for button in clickable_buttons(observation)}:
        return "click_not_in_previous_observation"
    return None


def _schema_extra_argument_names(name, arguments):
    """只拒绝 schema 未声明字段；缺少必填字段仍由工具动作转换报错。"""
    allowed_names = TOOL_ARGUMENT_NAMES.get(name)
    if allowed_names is None or not isinstance(arguments, dict):
        return []
    return sorted(set(arguments) - allowed_names)


def _schema_missing_argument_names(name, arguments):
    required_names = TOOL_REQUIRED_ARGUMENT_NAMES.get(name)
    if required_names is None or not isinstance(arguments, dict):
        return []
    return sorted(required_names - set(arguments))


def _page_type(observation):
    match = re.search(r"(?m)^page_type:\s*([a-z_]+)\s*$", observation or "")
    return match.group(1) if match else None


def available_tools_for_observation(observation):
    """Return tools usable on the current page and visible controls."""
    page = _page_type(observation)
    buttons = {button.casefold() for button in clickable_buttons(observation)}
    if page == "search_home":
        return ["search_products"] if "搜索功能是否可用: true" in observation.casefold() else []
    if page == "search_results":
        names = []
        asins = product_ids(observation)
        if any(asin.casefold() in buttons for asin in asins):
            names.append("open_product")
        if "next >" in buttons:
            names.append("next_page")
        if "< prev" in buttons:
            names.append("prev_page")
        if "back to search" in buttons:
            names.append("back_to_search")
        return names
    if page == "product_detail":
        names = []
        if any(button.startswith("opt_") for button in buttons):
            names.append("select_option")
        if "buy now" in buttons:
            names.append("buy_now")
        if "< prev" in buttons:
            names.append("prev_page")
        if "back to search" in buttons:
            names.append("back_to_search")
        return names
    return []


def action_guard_tool_message(tool_call, reason, observation):
    """构造标准 tool error observation，让 Agent 感知调用未被执行。"""
    requested_tool = (tool_call.get("function") or {}).get("name") or "unknown"
    page = _page_type(observation) or "unknown"
    available = available_tools_for_observation(observation)
    reason_text = reason
    # Navigation diagnosis is only a presentation layer for the corresponding
    # page-state rejection.  Preserve schema/parameter errors verbatim so the
    # model can correct the call instead of chasing a misleading page hint.
    page_navigation_reasons = {
        "click_not_in_previous_observation",
        "search_not_available_on_current_page",
        "select_option_not_available_on_current_page",
    }
    if (
        reason in page_navigation_reasons
        and page == "product_detail"
        and requested_tool == "open_product"
    ):
        reason_text = "商品详情页不能打开其他商品"
    elif (
        reason in page_navigation_reasons
        and page == "product_detail"
        and requested_tool == "next_page"
    ):
        reason_text = "商品详情页不能翻页"
    elif (
        reason in page_navigation_reasons
        and page == "product_detail"
        and requested_tool == "search_products"
    ):
        reason_text = "商品详情页不能搜索"
    elif (
        reason in page_navigation_reasons
        and page == "search_results"
        and requested_tool == "select_option"
    ):
        reason_text = "搜索结果页不能选择规格"
    elif (
        reason in page_navigation_reasons
        and page == "search_home"
        and requested_tool in {"open_product", "select_option", "next_page", "prev_page"}
    ):
        reason_text = "搜索首页没有当前商品或结果页目标"
    content = (
        "[动作被拒绝]\n"
        f"当前页面：{page}\n"
        f"请求工具：{requested_tool}\n"
        f"原因：{reason_text}\n"
        "可用工具：" + ("、".join(available) if available else "无")
    )
    if (tool_call.get("function") or {}).get("name") == "select_option":
        from shopping_grpo.environment.options import choices_from_observation
        choices = choices_from_observation(observation)
        if choices:
            content += "\n当前规格选项：" + json.dumps(choices, ensure_ascii=False)
    return {
        "role": "tool",
        "tool_call_id": tool_call.get("id"),
        "name": (tool_call.get("function") or {}).get("name"),
        RUNTIME_GUARD_FIELD: True,
        "content": content,
    }


def product_ids(observation):
    """提取当前 observation 中可打开的商品 ID，并保持出现顺序。"""
    return list(
        dict.fromkeys(
            re.findall(
                rf"(?m)^\d+\|({PRODUCT_ID_CAPTURE})\|",
                observation,
            )
        )
    )


def clickable_buttons(observation):
    """读取 observation footer 中当前页面实际可点击的按钮。"""
    match = re.search(r"可点击的按钮:\s*(\[[^\n]*\])", observation)
    if not match:
        return []
    try:
        buttons = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []
    return [button for button in buttons if isinstance(button, str)]
