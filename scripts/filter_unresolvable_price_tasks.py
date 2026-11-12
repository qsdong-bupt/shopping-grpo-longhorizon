import argparse
import gzip
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.acceptance.hard_checks import (  # noqa: E402
    _selected_options_from_instruction,
    resolve_variant_price,
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def iter_json_array(stream):
    """Stream a JSON array without materializing the complete catalog."""
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    eof = False
    while True:
        while position < len(buffer) and buffer[position] in " \t\r\n,[":
            position += 1
        if position < len(buffer) and buffer[position] == "]":
            return
        try:
            record, end = decoder.raw_decode(buffer, position)
        except json.JSONDecodeError:
            if eof:
                raise
            chunk = stream.read(1024 * 1024)
            buffer = buffer[position:] + chunk
            position = 0
            eof = not chunk
            continue
        yield record
        position = end
        if position > 1024 * 1024:
            buffer = buffer[position:]
            position = 0


def price_fingerprint(product):
    fields = {key: product.get(key) for key in (
        "customization_options", "pricing", "variant_combinations"
    )}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def inspect_target(product):
    instructions = product.get("instructions")
    if not isinstance(instructions, list) or not instructions or not isinstance(instructions[0], dict):
        return {"reason": "target_instruction_missing", "price_resolution": {"status": "unverifiable", "method": "target_instruction_missing"}}
    selected, missing, conflicts, required = _selected_options_from_instruction(product)
    resolution = resolve_variant_price(product, selected)
    if not missing and not conflicts and resolution.get("status") == "pass":
        return None
    return {
        "reason": "target_options_unresolved" if (missing or conflicts) else "target_price_unverifiable",
        "required_options": required,
        "selected_options": selected,
        "missing_options": missing,
        "axis_conflicts": conflicts,
        "price_resolution": resolution,
    }


def main():
    parser = argparse.ArgumentParser(description="Exclude tasks with unresolved target variant prices.")
    parser.add_argument("--apply", action="store_true", help="Back up and rewrite active task datasets.")
    args = parser.parse_args()

    split_path = ROOT / "outputs/split/tasks_final.jsonl"
    tasks = read_jsonl(split_path)
    records = {str(task["record_id"]): task for task in tasks}
    target_ids = set(records)

    source = ROOT / "environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz"
    catalog = ROOT / "environments/ShopSimulator/shop_env/data/items_eval_train.json"
    with catalog.open(encoding="utf-8") as stream:
        runtime_fingerprints = {
            str(product["asin"]): price_fingerprint(product)
            for product in iter_json_array(stream)
        }
    excluded = {}
    task_id_by_record = {}
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        for task_id, product in enumerate(iter_json_array(stream)):
            rid = str(product.get("asin") or "")
            if runtime_fingerprints.get(rid) != price_fingerprint(product):
                raise ValueError(f"Runtime/task product pricing differs: {rid}; resolve explicitly before applying")
            task_id_by_record[rid] = task_id
            problem = inspect_target(product)
            if problem is not None:
                excluded[rid] = {
                    "split": records.get(rid, {}).get("split", "outside_active_split"),
                    "task_id": task_id,
                    **problem,
                }
        for rid in target_ids - set(task_id_by_record):
            excluded[rid] = {"split": records[rid].get("split"), "reason": "target_product_missing"}

    excluded_ids = set(excluded)
    excluded_task_ids = {task_id_by_record[rid] for rid in excluded_ids if rid in task_id_by_record}

    changes = []
    candidates = [split_path]
    candidates.extend(sorted((ROOT / "data/grpo").glob("*.jsonl")))
    candidates.extend(sorted((ROOT / "outputs/grpo").glob("*.parquet")))
    candidates.extend(sorted((ROOT / "data/grpo").glob("*.parquet")))
    for directory in sorted((ROOT / "outputs").glob("sft_dataset*")):
        candidates.extend(sorted(directory.glob("*.parquet")))
    candidates.extend([
        ROOT / "outputs/sft_dataset/train.parquet",
        ROOT / "outputs/sft_dataset/val.parquet",
        ROOT / "outputs/sft_dataset/samples.jsonl",
    ])
    for path in dict.fromkeys(candidates):
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            rows = read_jsonl(path)
            if path == split_path:
                kept = [row for row in rows if str(row.get("record_id")) not in excluded_ids]
            else:
                if rows and "record_id" in rows[0]:
                    kept = [row for row in rows if str(row.get("record_id")) not in excluded_ids]
                elif rows and "task_id" in rows[0]:
                    removed_task_ids = {
                        task_id_by_record[rid] for rid in excluded_ids if rid in task_id_by_record
                    }
                    kept = [row for row in rows if int(row["task_id"]) not in removed_task_ids]
                else:
                    continue
            removed = len(rows) - len(kept)
            if removed:
                changes.append({"path": path, "rows_before": len(rows), "rows_after": len(kept), "rows_removed": removed, "format": "jsonl", "kept": kept})
        elif path.suffix == ".parquet":
            import pyarrow.parquet as pq
            schema = pq.read_schema(path)
            if "record_id" in schema.names:
                ids = pq.read_table(path, columns=["record_id"])["record_id"].to_pylist()
                mask = [str(rid) not in excluded_ids for rid in ids]
            elif "extra_info" in schema.names:
                infos = pq.read_table(path, columns=["extra_info"])["extra_info"].to_pylist()
                mask = [not (
                    str(info.get("record_id")) in excluded_ids or info.get("task_id") in excluded_task_ids
                ) for info in infos]
            else:
                raise ValueError(f"Missing stable identity in {path}")
            removed = len(mask) - sum(mask)
            if removed:
                changes.append({"path": path, "rows_before": len(mask), "rows_after": sum(mask), "rows_removed": removed, "format": "parquet", "kept": mask})

    print(json.dumps({
        "mode": "apply" if args.apply else "dry_run",
        "tasks_before": len(tasks),
        "split_before": dict(Counter(task.get("split") for task in tasks)),
        "excluded_tasks": len(excluded_ids & target_ids),
        "excluded_by_split": dict(Counter(item.get("split", "unknown") for rid, item in excluded.items() if rid in target_ids)),
        "excluded_by_reason": dict(Counter(item["reason"] for rid, item in excluded.items() if rid in target_ids)),
        "excluded_price_methods": dict(Counter(item.get("price_resolution", {}).get("method") for rid, item in excluded.items() if rid in target_ids)),
        "datasets_to_change": [{k: str(v) if k == "path" else v for k, v in change.items() if k != "kept"} for change in changes],
    }, ensure_ascii=False, indent=2))

    if not args.apply or not changes:
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = ROOT / "outputs/backups" / f"price_filter_{stamp}"
    report_dir = ROOT / "outputs/analysis" / f"price_filter_{stamp}"
    backup_root.mkdir(parents=True, exist_ok=False)
    report_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"backup_dir": str(backup_root.relative_to(ROOT)), "files": [], "tasks_before": len(tasks), "excluded_tasks": len(excluded_ids & target_ids)}
    manifest["source_sha256"] = {str(path.relative_to(ROOT)): sha256(path) for path in [source, catalog]}
    manifest["split_before"] = dict(Counter(task.get("split") for task in tasks))
    manifest["excluded_by_price_method"] = dict(Counter(
        item.get("price_resolution", {}).get("method") for rid, item in excluded.items() if rid in target_ids
    ))
    sidecars = [ROOT / "outputs/split/report.md", ROOT / "data/grpo/metadata.json", ROOT / "data/README.md"]
    for path in sidecars:
        if path.exists():
            backup = backup_root / path.relative_to(ROOT)
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
            manifest["files"].append({"path": str(path.relative_to(ROOT)), "sha256_before": sha256(path), "sha256_backup": sha256(backup), "kind": "metadata"})
    for change in changes:
        path = change["path"]
        relative = path.relative_to(ROOT)
        backup = backup_root / relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
        manifest["files"].append({"path": str(relative), "sha256_before": sha256(path), "sha256_backup": sha256(backup), "rows_before": change["rows_before"], "rows_after": change["rows_after"], "rows_removed": change["rows_removed"]})
    # Complete every backup before writing any active dataset.
    for change in changes:
        path = change["path"]
        if change["format"] == "jsonl":
            write_jsonl(path, change["kept"])
        else:
            import pyarrow as pa
            import pyarrow.parquet as pq
            temp = path.with_suffix(path.suffix + ".tmp")
            table = pq.read_table(path)
            pq.write_table(table.filter(pa.array(change["kept"])), temp)
            temp.replace(path)
            del table
    write_jsonl(report_dir / "excluded_tasks.jsonl", [{"record_id": rid, **item} for rid, item in sorted(excluded.items())])
    manifest["tasks_after"] = len(tasks) - len(excluded_ids & target_ids)
    manifest["split_after"] = dict(Counter(
        task.get("split") for task in tasks if str(task.get("record_id")) not in excluded_ids
    ))
    manifest["report_dir"] = str(report_dir.relative_to(ROOT))
    metadata_path = ROOT / "data/grpo/metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for split in ("train", "validation"):
            row = metadata[split]
            row["tasks"] = len(read_jsonl(ROOT / row["jsonl"]))
            row["parquet_sha256"] = sha256(ROOT / row["parquet"])
        metadata["target_price_filter"] = {"audit": manifest["report_dir"], "resolver": "variant-price-v1"}
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        readme = ROOT / "data/README.md"
        if readme.exists():
            text = readme.read_text(encoding="utf-8")
            text = text.replace("| 1000 / 50 |", f"| {metadata['train']['tasks']} / {metadata['validation']['tasks']} |")
            readme.write_text(text, encoding="utf-8")
    split_report = ROOT / "outputs/split/report.md"
    report = ["# 当前任务切分统计", "", "已按目标商品目标规格价格可解析性过滤；保持原 split、任务 ID、画像和顺序。", "", "| 集合 | 过滤前 | 移除 | 保留 |", "| --- | ---: | ---: | ---: |"]
    for split, before in manifest["split_before"].items():
        after = manifest["split_after"].get(split, 0)
        report.append(f"| {split} | {before} | {before - after} | {after} |")
    report.extend(["", f"审计：`{manifest['report_dir']}`；原文件备份：`{manifest['backup_dir']}`。", "", "历史评测结果保留原题集分母，后续评测请使用新的运行名称。"])
    split_report.write_text("\n".join(report) + "\n", encoding="utf-8")
    for entry in manifest["files"]:
        entry["sha256_after"] = sha256(ROOT / entry["path"])
    (report_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Applied; backups: {backup_root}; audit: {report_dir}")


if __name__ == "__main__":
    main()
