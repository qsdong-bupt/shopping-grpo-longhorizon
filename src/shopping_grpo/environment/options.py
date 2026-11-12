"""Current-page option IDs; raw axis/value strings remain execution authority."""

import json
import re


class OptionResolutionError(ValueError):
    pass


def option_choices(state):
    if state.get("page_type") != "product_detail":
        return []
    selected = state.get("selected_options") or {}
    choices = []
    for axis, values in sorted((state.get("available_options") or {}).items()):
        for value in values:
            if not isinstance(axis, str) or not isinstance(value, str):
                raise OptionResolutionError("invalid_option_table")
            choices.append({
                "option_id": f"opt_{len(choices) + 1}",
                "axis": axis,
                "value": value,
                "selected": selected.get(axis) == value,
            })
    return choices


def choices_from_observation(observation):
    match = re.search(r"(?m)^option_choices: (\[.*\])$", observation or "")
    if not match:
        return []
    try:
        choices = json.loads(match.group(1))
        if not isinstance(choices, list):
            return []
        ids = set()
        for choice in choices:
            if not isinstance(choice, dict) or not all(
                isinstance(choice.get(key), str) for key in ("option_id", "axis", "value")
            ) or not re.fullmatch(r"opt_[1-9][0-9]*", choice["option_id"]):
                return []
            if choice["option_id"] in ids:
                return []
            ids.add(choice["option_id"])
        return choices
    except (ValueError, TypeError):
        return []


def resolve_option(option_id, observation):
    if not re.search(r"(?m)^page_type: product_detail$", observation or ""):
        raise OptionResolutionError("select_option_not_available_on_current_page")
    if not isinstance(option_id, str) or not re.fullmatch(r"opt_[1-9][0-9]*", option_id):
        raise OptionResolutionError("invalid_option_id")
    for choice in choices_from_observation(observation):
        if choice["option_id"] == option_id:
            return dict(choice)
    raise OptionResolutionError("option_id_not_on_current_page")
