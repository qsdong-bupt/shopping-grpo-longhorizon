"""Report integration tests for the current build_data/main API."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("eval_report", Path(__file__).resolve().parents[1] / "scripts/build_eval_report.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def row(task=1, reward_valid=True):
    detail = {"reward_version": "shopsimulator-reward-v3", "reward_type": "gold_purchase",
              "reward_valid": reward_valid, "purchase_success": True, "termination_reason": "gold_purchase"}
    return {"task_id": task, "status": "done", "done": True, "steps": [],
            "messages": [{"role": "user", "content": '</script><img src=x onerror="alert(1)">'}],
            "terminal_result": {"done": True, "over": True, "reward_detail": detail}}


def write_run(path, rows):
    (path / "summary.json").write_text(json.dumps({"strict_successes": 900, "protocol": {"model": "test"}}), encoding="utf-8")
    (path / "trajectories.jsonl").write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")


def test_report_recomputes_success_and_adds_failure_analysis(tmp_path):
    write_run(tmp_path, [row(), row(2, False)])
    data = report.build_data(tmp_path)
    assert data["summary"]["strict"] == 1
    assert data["failure_analysis"]["tasks"] == 2
    assert data["rows"][0]["failure_analysis"]["final_category"] == "success"
    assert data["rows"][1]["failure_analysis"]["final_category"] == "review_required"


def test_duplicate_tasks_and_empty_input_rejected(tmp_path):
    write_run(tmp_path, [row(), row()])
    with pytest.raises(ValueError, match="duplicate task IDs"):
        report.build_data(tmp_path)
    write_run(tmp_path, [])
    with pytest.raises(ValueError, match="Not enough"):
        report.build_data(tmp_path)


def test_report_limit_applies_to_analysis_and_exports(tmp_path):
    write_run(tmp_path, [row(), row(2)])
    output = tmp_path / "limited.html"
    report.main(["--run-dir", str(tmp_path), "--output", str(output), "--limit", "1"])
    summary = json.loads(output.with_suffix(".failure_analysis.summary.json").read_text(encoding="utf-8"))
    details = [json.loads(line) for line in output.with_suffix(".failure_analysis.jsonl").read_text(encoding="utf-8").splitlines()]
    assert summary["tasks"] == len(details) == 1
    assert details[0]["task_id"] == 1
    assert '900' in (tmp_path / "summary.json").read_text(encoding="utf-8")


def test_html_injection_protection_and_analysis_controls(tmp_path):
    write_run(tmp_path, [row()])
    data = report.build_data(tmp_path)
    html = report.render_html(data)
    assert '</script><img' not in html
    assert "failure-category-filter" in html
    assert "failure-issue-filter" in html
    assert "failure-review-filter" in html
    assert "row.failure_analysis" in html
    payload = html.split("const REPORT_DATA = ", 1)[1].split(";\nconst S", 1)[0]
    assert json.loads(payload)["rows"][0]["trajectory"] == row()


def test_infrastructure_failure_in_report(tmp_path):
    item = row()
    item.update(status="error", done=False, terminal_result={}, error={"type": "TimeoutError", "message": "timeout"})
    write_run(tmp_path, [item])
    analysis = report.build_data(tmp_path)["rows"][0]["failure_analysis"]
    assert analysis["final_category"] == "infrastructure_failure"
    assert analysis["issue_tags"] == ["environment_or_service_error"]
