"""Identity and admission checks for the single active shopping protocol."""

import hashlib
import json
import re

from shopping_grpo.environment.observation import OBSERVATION_VERSION
from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS, TOOL_VERSION

ENVIRONMENT_VERSION = "shopsimulator-environment-v2.2"
RETIRED_TOOLS = frozenset(
    {"view_description", "view_features", "view_reviews", "view_attributes"}
)
RETIRED_BUTTONS = frozenset({"description", "features", "reviews", "attributes"})


def runtime_protocol():
    encoded = json.dumps(SHOP_TOOL_SCHEMAS, ensure_ascii=False, sort_keys=True).encode()
    return {
        "environment_version": ENVIRONMENT_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "tool_version": TOOL_VERSION,
        "option_mapping_version": "current-page-option-ids-v1",
        "tool_schema_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def message_protocol_problem(messages):
    """Reject retired calls/pages rather than rewriting recorded evidence."""
    allowed = {tool["function"]["name"] for tool in SHOP_TOOL_SCHEMAS}
    for message in messages:
        for call in message.get("tool_calls") or []:
            name = (call.get("function") or {}).get("name")
            if name not in allowed:
                return f"unsupported_tool:{name}"
            if name == "select_option":
                try:
                    raw = (call.get("function") or {}).get("arguments", "{}")
                    arguments = json.loads(raw) if isinstance(raw, str) else raw
                except (ValueError, TypeError):
                    return "invalid_option_arguments"
                if not isinstance(arguments, dict) or set(arguments) != {"option_id"}:
                    return "unsupported_option_arguments"
                if not isinstance(arguments["option_id"], str) or not re.fullmatch(r"opt_[1-9][0-9]*", arguments["option_id"]):
                    return "invalid_option_id"
        content = str(message.get("content") or "")
        if "[SHOPPING_OBSERVATION_V2]" in content or "page_type: information_subpage" in content:
            return "retired_observation"
        match = re.search(r"可点击的按钮:\s*(\[[^\n]*\])", content)
        if match:
            try:
                buttons = json.loads(match.group(1))
            except json.JSONDecodeError:
                return "invalid_action_footer"
            if any(str(button).casefold() in RETIRED_BUTTONS for button in buttons):
                return "retired_action_footer"
    return None
