"""Shared, CPU-only contracts for reproducible SFT preparation."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from shopping_grpo.acceptance.jev_client import JEV_RUBRIC_VERSION
from shopping_grpo.environment.protocol import runtime_protocol
from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS

PREPARATION_VERSION = "sft-preparation-v1"
ALTERNATIVES = {"valid_alternative_purchase", "partial_alternative_purchase"}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def ensure_identity(path, identity):
    """Create once; changing inputs requires a new batch directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    except FileExistsError:
        if json.loads(path.read_text(encoding="utf-8")) != identity:
            raise ValueError(f"Batch identity changed: {path}; use a new directory")


def task_index(path):
    tasks = {}
    for task in read_jsonl(path):
        rid = str(task["record_id"])
        if rid in tasks:
            raise ValueError(f"Duplicate frozen task ID: {rid}")
        tasks[rid] = task
    return tasks


def training_task(task):
    return task.get("split") == "train" and task.get("tag") == "train"


def user_request(trajectory):
    # The first user message is the actual reset-time request, never a product's
    # instruction_text (which belongs to that product's own source task).
    return next((m.get("content") for m in trajectory.get("messages", [])
                 if m.get("role") == "user"), None)


def purchase_problem(purchase):
    if not isinstance(purchase, dict) or not purchase.get("asin") or not purchase.get("name"):
        return "purchase_incomplete"
    price = purchase.get("price")
    if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
        return "purchase_price_invalid"
    if not isinstance(purchase.get("options"), dict):
        return "purchase_options_invalid"
    return None


def jev_evidence(trajectory):
    purchase = (trajectory.get("terminal_result") or {}).get("purchase") or {}
    query = user_request(trajectory)
    if not isinstance(query, str) or not query.strip():
        raise ValueError("missing_original_request")
    problem = purchase_problem(purchase)
    if problem:
        raise ValueError(problem)
    if not isinstance(purchase.get("attributes"), list):
        raise ValueError("missing_purchased_attributes")
    return {"user_request": query, "purchased_product": {
        key: purchase.get(key) for key in
        ("asin", "name", "category", "product_category", "attributes", "price", "options")}}


def terminal_problem(trajectory, require_jev=True):
    terminal = trajectory.get("terminal_result") or {}
    detail = terminal.get("reward_detail") or {}
    if not isinstance(detail, dict) or not detail.get("reward_type"):
        return "terminal_incomplete"
    if terminal.get("over") is not True:
        return "terminal_not_over"
    if terminal.get("reward_valid") is not True or detail.get("reward_valid") is not True:
        return "reward_invalid"
    if not terminal.get("termination_reason"):
        return "terminal_incomplete"
    kind = detail["reward_type"]
    if kind not in ALTERNATIVES | {"gold_purchase"}:
        return "purchase_not_accepted"
    problem = purchase_problem(terminal.get("purchase"))
    if problem:
        return problem
    if kind in ALTERNATIVES and require_jev:
        audit = (trajectory.get("meta") or {}).get("audit") or {}
        verdict = audit.get("jev_verdict") or {}
        if verdict.get("choice") != "fully_satisfies":
            return "missing_jev_approval"
        if verdict.get("rubric_version") != JEV_RUBRIC_VERSION or not verdict.get("request_hash"):
            return "jev_audit_incomplete"
        try:
            if audit.get("jev_input") != jev_evidence(trajectory):
                return "jev_evidence_mismatch"
        except ValueError as exc:
            return str(exc)
    return None


def argument_problem(call):
    function = call.get("function") or {}
    name, raw = function.get("name"), function.get("arguments")
    if not isinstance(raw, str):
        return "arguments_not_json_string"
    try:
        args = json.loads(raw)
    except ValueError:
        return "arguments_invalid_json"
    if not isinstance(args, dict):
        return "arguments_not_object"
    for key, value in args.items():
        if value is None:
            return f"argument_value_null:{key}"
        if isinstance(value, str) and not value.strip():
            return f"argument_value_empty:{key}"
    schema = next((t["function"]["parameters"] for t in SHOP_TOOL_SCHEMAS
                   if t["function"]["name"] == name), None)
    if schema is None:
        return f"unsupported_tool:{name}"
    properties = schema.get("properties", {})
    if set(args) - set(properties) or set(schema.get("required", [])) - set(args):
        return "tool_schema_fields"
    if any(not isinstance(value, str) for value in args.values()):
        return "tool_schema_type"
    return None


def conversation_problem(messages):
    if len(messages) < 4 or messages[0].get("role") != "system" or messages[1].get("role") != "user":
        return "conversation_incomplete"
    if not isinstance(messages[1].get("content"), str) or not messages[1]["content"].strip():
        return "missing_original_request"
    used = set()
    if len(messages) % 2:
        return "conversation_tool_pair"
    for i in range(2, len(messages), 2):
        assistant = messages[i]
        calls = assistant.get("tool_calls") or []
        if assistant.get("role") != "assistant" or len(calls) != 1 or i + 1 >= len(messages):
            return "conversation_tool_pair"
        call_id = calls[0].get("id")
        tool = messages[i + 1]
        if not call_id or call_id in used or tool.get("role") != "tool" or tool.get("tool_call_id") != call_id:
            return "conversation_tool_id"
        used.add(call_id)
    if messages[-2]["tool_calls"][0].get("function", {}).get("name") != "buy_now":
        return "conversation_missing_purchase"
    return None


def tokenizer_identity(model):
    path = Path(model).resolve()
    files = {name: file_hash(path / name) for name in
             ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json")
             if (path / name).is_file()}
    if "tokenizer_config.json" not in files:
        raise ValueError("Use a local model directory containing tokenizer_config.json")
    return {"files": files, "enable_thinking": False,
            "runtime_protocol": runtime_protocol()}


def harness_identity(root):
    root = Path(root)
    paths = set((root / "src/shopping_grpo/environment").glob("*.py"))
    for directory in ("environments/ShopSimulator/shop_env/web_agent_site/engine",
                      "environments/ShopSimulator/shop_env/web_agent_site/envs",
                      "environments/ShopSimulator/shop_env/shop_env"):
        paths.update((root / directory).glob("*.py"))
    paths.update(root / name for name in
                 ("src/shopping_grpo/evaluation/rollout.py", "scripts/collect_teacher_data.py",
                  "src/shopping_grpo/acceptance/jev_client.py",
                  "src/shopping_grpo/training/sft/preparation.py", "data/environment.json",
                  "configs/tools.json"))
    return {str(p.relative_to(root)).replace("\\", "/"): file_hash(p)
            for p in sorted(paths) if p.is_file()}
