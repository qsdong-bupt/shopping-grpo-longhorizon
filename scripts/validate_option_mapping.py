#!/usr/bin/env python3
"""CPU-only replay of option mappings; this does not run a model evaluation."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.environment.actions import action_reject_reason  # noqa: E402
from shopping_grpo.environment.observation import render_structured_observation  # noqa: E402
from shopping_grpo.environment.options import choices_from_observation  # noqa: E402
from shopping_grpo.environment.tools import tool_call_to_action  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.baseline_run / "trajectories.jsonl"
    identity = json.loads((args.baseline_run / "run_identity.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = {t["task_id"] for t in identity["tasks"]}
    assert len(expected) == 200 and len(rows) == 200
    assert {r["task_id"] for r in rows} == expected
    tree = ast.parse((ROOT / "environments/ShopSimulator/shop_env/web_agent_site/engine/engine.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse_action")
    namespace = {"re": re}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "engine.py", "exec"), namespace)
    parse_action = namespace["parse_action"]
    states = options = 0
    covered = set()
    for row in rows:
        results = [row.get("initial_result") or {}, *[s.get("result") or {} for s in row["steps"]]]
        for result in results:
            state = result.get("observation_state") or {}
            if state.get("page_type") != "product_detail":
                continue
            states += 1
            covered.add(row["task_id"])
            observation = render_structured_observation(state)
            choices = choices_from_observation(observation)
            assert len(choices) == sum(len(v) for v in (state.get("available_options") or {}).values())
            for choice in choices:
                parameters = {"option_id": choice["option_id"]}
                assert action_reject_reason("select_option", parameters, observation) is None
                action = tool_call_to_action("select_option", parameters, observation)
                name, value = parse_action(action)
                assert name == "click_option"
                assert json.loads(value) == {"axis": choice["axis"], "value": choice["value"]}
                assert choice["value"] in state["available_options"][choice["axis"]]
                options += 1
    report = {
        "validation_type": "cpu_only_mapping_replay_not_model_evaluation",
        "baseline_run": str(args.baseline_run),
        "baseline_trajectories_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "task_count": len(rows), "tasks_with_product_details": len(covered),
        "product_detail_states": states, "option_round_trips": options,
        "mapping_failures": 0,
        "limitation": "Verifies stored public states and actual action parser, not model choices, live environment sessions or strict success.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
