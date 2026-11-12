# 全量真实 token 统计（整序列渲染一次，用于确定 max_length 与超长过滤）
import json
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from export_sft_parquet import slim_messages  # noqa: E402
from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS  # noqa: E402
from shopping_grpo.training.sft.preparation import (
    atomic_json, file_hash, read_jsonl, tokenizer_identity,
)

from transformers import AutoTokenizer  # noqa: E402

def count_tokens(directory, model):
    directory = Path(directory)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model), trust_remote_code=True, local_files_only=True)
    identity = tokenizer_identity(model)
    samples_path = directory / "samples.jsonl"
    samples = read_jsonl(samples_path)
    if not samples or len({s["record_id"] for s in samples}) != len(samples):
        raise ValueError("Empty samples or duplicate record IDs")
    out = directory / "token_stats.jsonl"
    temporary = out.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as f:
        for i, s in enumerate(samples):
            messages = slim_messages(s)
            # 归一化：与训练数据集读回后的形态一致
            for m in messages:
                for call in m.get("tool_calls") or []:
                    fn = call.get("function") or {}
                    if isinstance(fn.get("arguments"), list):
                        fn["arguments"] = dict(fn["arguments"])
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, tools=SHOP_TOOL_SCHEMAS,
                add_generation_prompt=False, enable_thinking=False)
            n = len(tokenizer(text, add_special_tokens=False).input_ids)
            f.write(json.dumps({"record_id": s["record_id"], "tokens": n}) + "\n")
            if (i + 1) % 500 == 0:
                print(f"{i+1}/{len(samples)}", flush=True)
    temporary.replace(out)
    atomic_json(directory / "token_manifest.json", {
        "tokenizer": identity, "samples_sha256": file_hash(samples_path),
        "token_stats_sha256": file_hash(out), "rows": len(samples)})
    print("done ->", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("SHOP_MODEL"),
                        required=not bool(os.environ.get("SHOP_MODEL")))
    args = parser.parse_args()
    count_tokens(args.dataset_dir, args.model)

if __name__ == "__main__":
    main()
