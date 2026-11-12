# SFT 教材转 veRL 训练格式：samples.jsonl -> train.parquet
# 列约定见 multiturn_dataset.py 头注：
#   messages 用 arrow struct + map（无参数不补 None），逐消息保留 train 标记
#   （守卫拒绝的 assistant 段 train=0，数据集据此置 loss mask=0）；
#   tools 存完整 schema 的 JSON 字符串（保留 enum）；
#   enable_thinking 恒为 False（全链路关思考）；
#   准入审计列（accept_reason/reward_type/reward_valid/over/termination_reason/
#   jev_verdict/purchase）随行落盘，替代购买的准入依据可独立追溯。
# 一律 fail fast，禁止静默转换：非法 JSON 参数、null/空字符串参数值、
# token 统计缺失都立即报错退出（历史教训：把 None 静默转成空字符串产生了
# view_features({"__dummy__": ""}) 一类参数语义污染）。
# 超长按 token_stats.jsonl 的真实 token 数过滤（默认 16384，禁止截断）。
# 验证与训练走同一条代码路径：ShoppingMultiTurnSFTDataset + veRL load_extern_object。
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS  # noqa: E402
from shopping_grpo.environment.protocol import message_protocol_problem, runtime_protocol  # noqa: E402
from shopping_grpo.training.sft.preparation import (
    atomic_json, digest, file_hash, read_jsonl, task_index, training_task,
    tokenizer_identity,
)

OUT = ROOT / "outputs/sft_dataset"
DEFAULT_MAX_TOKENS = 16384


def arrow_schema():
    arguments = pa.map_(pa.string(), pa.string())
    function = pa.struct([
        ("name", pa.string()), ("arguments", arguments),
    ])
    tool_call = pa.struct([
        ("id", pa.string()), ("type", pa.string()), ("function", function),
    ])
    message = pa.struct([
        ("role", pa.string()),
        ("content", pa.string()),
        ("tool_calls", pa.list_(tool_call)),
        ("tool_call_id", pa.string()),
        ("train", pa.int64()),
    ])
    return pa.schema([
        ("record_id", pa.string()),
        ("teacher", pa.string()),
        ("difficulty", pa.string()),
        ("persona_condition", pa.string()),
        ("tokens", pa.int64()),
        ("messages", pa.list_(message)),
        ("tools", pa.string()),
        ("runtime_protocol", pa.string()),
        ("enable_thinking", pa.bool_()),
        ("accept_reason", pa.string()),
        ("reward_type", pa.string()),
        ("reward_valid", pa.bool_()),
        ("over", pa.bool_()),
        ("termination_reason", pa.string()),
        ("jev_verdict", pa.string()),
        ("purchase", pa.string()),
        ("terminal_result", pa.string()),
        ("jev_input", pa.string()),
        ("source_sha256", pa.string()),
    ])


def parse_arguments(record_id, tool_call):
    """解析并校验一条工具调用的参数；非法立即报错，绝不静默修复。"""
    function = tool_call.get("function") or {}
    name = function.get("name")
    raw = function.get("arguments")
    if not isinstance(raw, str):
        raise ValueError(
            f"{record_id}: {name} arguments must be a JSON string, "
            f"got {type(raw).__name__}"
        )
    try:
        arguments = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{record_id}: {name} arguments is not valid JSON: {raw!r}"
        ) from exc
    if not isinstance(arguments, dict):
        raise ValueError(
            f"{record_id}: {name} arguments must decode to an object, "
            f"got {type(arguments).__name__}"
        )
    for key, value in arguments.items():
        if value is None:
            raise ValueError(f"{record_id}: {name} parameter {key!r} is null")
        if isinstance(value, str) and not value:
            raise ValueError(f"{record_id}: {name} parameter {key!r} is empty")
    return name, arguments


def slim_messages(sample):
    record_id = sample["record_id"]
    if sample.get("runtime_protocol") != runtime_protocol():
        raise ValueError(f"{record_id}: runtime protocol mismatch; rebuild current demonstrations")
    problem = message_protocol_problem(sample["messages"])
    if problem:
        raise ValueError(f"{record_id}: {problem}")
    messages = []
    for m in sample["messages"]:
        role = m["role"]
        if role == "assistant":
            tool_calls = None
            for call in m.get("tool_calls") or []:
                name, arguments = parse_arguments(record_id, call)
                call = dict(call)
                call["function"] = {
                    "name": name,
                    "arguments": arguments,
                }
                (tool_calls := tool_calls or []).append(call)
            messages.append({
                "role": "assistant",
                "content": m.get("content") or "",
                "train": int(m.get("train", 1)),
                **({"tool_calls": tool_calls} if tool_calls else {}),
            })
        elif role == "tool":
            messages.append({
                "role": "tool",
                "tool_call_id": m.get("tool_call_id"),
                "content": m.get("content") or "",
                "train": int(m.get("train", 0)),
            })
        else:
            messages.append({
                "role": role,
                "content": m.get("content") or "",
                "train": int(m.get("train", 0)),
            })
    return messages


def _fill(value, arrow_type):
    if isinstance(arrow_type, pa.MapType):
        value_type = arrow_type.item_type
        result = {}
        for k, v in (value or {}).items():
            if value_type == pa.string():
                text = "" if v is None else str(v)
                if not text:
                    raise ValueError(f"empty map value for key {k!r}")
                result[str(k)] = text
            else:
                result[str(k)] = _fill(v, value_type)
        return result
    if isinstance(arrow_type, pa.StructType):
        result = {}
        value = value or {}
        for field in arrow_type:
            result[field.name] = _fill(value.get(field.name), field.type)
        return result
    if isinstance(arrow_type, pa.ListType):
        return [_fill(item, arrow_type.value_field.type)
                for item in (value or [])]
    return value


def export_dataset(directory, model, tasks_path, max_tokens=DEFAULT_MAX_TOKENS,
                   validation_ratio=0.02, seed=20261109):
    """Write both splits only after every retained row passes the training path."""
    from build_sft_dataset import admission_problem
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verl.utils.import_utils import load_extern_object
    directory = Path(directory)
    if max_tokens < 1 or not 0 < validation_ratio < 1:
        raise ValueError("Invalid max_tokens or validation_ratio")
    samples_path = directory / "samples.jsonl"
    manifest = json.loads((directory / "token_manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("tokenizer") != tokenizer_identity(model)
            or manifest.get("samples_sha256") != file_hash(samples_path)
            or manifest.get("token_stats_sha256") != file_hash(directory / "token_stats.jsonl")):
        raise ValueError("Stale token statistics or tokenizer mismatch; rerun token_stats.py")
    samples = read_jsonl(samples_path)
    if not samples or len({s["record_id"] for s in samples}) != len(samples):
        raise ValueError("Empty samples or duplicate record IDs")
    tasks = task_index(tasks_path)
    build = json.loads((directory / "build_identity.json").read_text(encoding="utf-8"))
    if build["collection_manifest"]["tasks_sha256"] != file_hash(tasks_path):
        raise ValueError("Frozen task split changed")
    token_rows = read_jsonl(directory / "token_stats.jsonl")
    tokens = {r["record_id"]: r["tokens"] for r in token_rows}
    if len(tokens) != len(token_rows) or set(tokens) != {s["record_id"] for s in samples}:
        raise ValueError("Token statistics IDs do not match samples exactly")
    splits = {"train": [], "val": []}
    excluded = []
    assignments = {}
    for sample in samples:
        rid = sample["record_id"]
        if not training_task(tasks.get(rid, {})):
            raise ValueError(f"Non-training/frozen eval task in corpus: {rid}")
        terminal = sample.get("terminal_result") or {}
        problem = admission_problem({"runtime_protocol": sample.get("runtime_protocol"),
            "messages": sample["messages"], "terminal_result": terminal,
            "meta": {"accept_reason": sample.get("accept_reason"),
                     "audit": {"jev_verdict": sample.get("jev_verdict"),
                               "jev_input": sample.get("jev_input")}}})
        if problem:
            raise ValueError(f"{rid}: admission failed: {problem}")
        detail = terminal["reward_detail"]
        if any(sample.get(key) != expected for key, expected in {
                "reward_type": detail["reward_type"], "reward_valid": terminal["reward_valid"],
                "over": terminal["over"], "termination_reason": terminal["termination_reason"],
                "purchase": terminal["purchase"]}.items()):
            raise ValueError(f"{rid}: audit columns disagree with terminal evidence")
        # All rows with an identical normalized user request stay in one split.
        query = next(m["content"] for m in sample["messages"] if m["role"] == "user")
        group = " ".join(query.split())
        score = int(digest([seed, group]), 16) / (1 << 256)
        split = "val" if score < validation_ratio else "train"
        assignments[rid] = split
        count = tokens[rid]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"Invalid token count: {rid}")
        if count > max_tokens:
            excluded.append({"record_id": rid, "reason": "overlength", "tokens": count})
            continue
        row = {key: sample.get(key) for key in
               ("record_id", "teacher", "difficulty", "persona_condition", "accept_reason",
                "reward_type", "reward_valid", "over", "termination_reason", "source_sha256")}
        row.update(tokens=count, messages=slim_messages(sample),
                   tools=json.dumps(SHOP_TOOL_SCHEMAS, ensure_ascii=False),
                   runtime_protocol=json.dumps(runtime_protocol(), sort_keys=True),
                   enable_thinking=False)
        for key in ("purchase", "jev_verdict", "jev_input", "terminal_result"):
            row[key] = json.dumps(sample[key], ensure_ascii=False) if sample.get(key) is not None else None
        splits[split].append(row)
    if not all(splits.values()):
        raise ValueError("Empty train or validation split; collect more independent tasks or adjust validation-ratio")
    dataset_cls = load_extern_object("pkg://shopping_grpo.training.sft.multiturn_dataset",
                                   "ShoppingMultiTurnSFTDataset")
    tokenizer = AutoTokenizer.from_pretrained(str(model), trust_remote_code=True, local_files_only=True)
    config = OmegaConf.create({"max_length": max_tokens, "truncation": "error"})
    schema = arrow_schema()
    types = {f.name: f.type for f in schema}
    temporary_paths = []
    counts = {}
    try:
        for split, rows in splits.items():
            path = directory / f"{split}.parquet.pending"
            temporary_paths.append(path)
            table = pa.Table.from_pylist([{k: _fill(row.get(k), t) for k, t in types.items()}
                                          for row in rows], schema=schema)
            pq.write_table(table, path)
            dataset = dataset_cls([str(path)], tokenizer, config)
            supervised = 0
            for index, row in enumerate(rows):
                item = dataset[index]
                actual = len(item["input_ids"])
                mask = item["loss_mask"]
                if actual != row["tokens"] or int(mask.sum()) <= 0 or not bool(((mask == 0) | (mask == 1)).all()):
                    raise ValueError(f"Token length/loss mask mismatch: {row['record_id']}")
                supervised += int(mask.sum())
            counts[split] = {"rows": len(rows), "supervised_tokens": supervised}
        # Validation succeeded for both files before either active file changes.
        for split, path in zip(splits, temporary_paths):
            path.replace(directory / f"{split}.parquet")
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
    report = {"status": "all_rows_verified", "splits": counts, "excluded": excluded,
              "split_seed": seed, "validation_ratio": validation_ratio, "assignments": assignments,
              "runtime_protocol": runtime_protocol(), "tokenizer": manifest["tokenizer"],
              "samples_sha256": file_hash(samples_path), "tasks_sha256": file_hash(tasks_path),
              "train_val_id_overlap": 0, "train_val_query_overlap": 0, "frozen_eval_id_overlap": 0,
              "output_sha256": {f"{s}.parquet": file_hash(directory / f"{s}.parquet") for s in splits}}
    atomic_json(directory / "export_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("SHOP_MODEL"),
                        required=not bool(os.environ.get("SHOP_MODEL")))
    parser.add_argument("--tasks", type=Path, default=ROOT / "outputs/split/tasks_final.jsonl")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--validation-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20261109)
    args = parser.parse_args()
    print(json.dumps(export_dataset(args.dataset_dir, args.model, args.tasks,
                     args.max_tokens, args.validation_ratio, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
