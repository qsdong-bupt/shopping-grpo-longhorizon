# Teacher 正式采集：成功即停（每题预算 3 次），在线验收，断点续跑，分级落盘
# 用法：
#   python scripts/collect_teacher_data.py --output-dir outputs/collection/<batch> --limit 50
#   同一批次使用同一 output-dir 续采；代码、任务与配置变化需要新批次。
# 产出：
#   outputs/collection/trajectories/<record_id>_<attempt>.json  全部轨迹
#   outputs/collection/task_status.jsonl                         每题终态（断点）
#   outputs/collection/report.json                               汇总
import argparse
import gzip
import json
import os
import random
import sys
import threading
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_grpo.acceptance.jev_client import (  # noqa: E402
    JevApiError,
    JevDecisionsClient,
)
from shopping_grpo.collection.teacher_client import AnthropicTeacherClient  # noqa: E402
from shopping_grpo.evaluation.rollout import (  # noqa: E402
    TEACHER_SYSTEM_PROMPT,
    OpenAIChatClient,
    collect_for_task,
)
from shopping_grpo.persona.render import render_persona  # noqa: E402
from shopping_grpo.environment.protocol import runtime_protocol  # noqa: E402
from shopping_grpo.environment.protocol import message_protocol_problem
from shopping_grpo.training.sft.preparation import (
    PREPARATION_VERSION, argument_problem, atomic_json, conversation_problem,
    ensure_identity, file_hash, harness_identity, jev_evidence, read_jsonl,
    task_index, terminal_problem, training_task,
)

OUT = ROOT / "outputs/collection"
TRAJ = OUT / "trajectories"
BUDGET = 3
RNG = random.Random(20260926)
STRUCTURAL_FAIL = {"max_steps", "invalid_action_limit", "error", "model_no_tool_call"}
TASKS = ROOT / "outputs/split/tasks_final.jsonl"
PERSONAS = ROOT / "outputs/persona/persona_pool.jsonl"
BASE_URL = "http://127.0.0.1:5700"
MAX_STEPS = 30
INFRA_ERRORS = {"TeacherApiError", "ShopEnvironmentError", "ShopHttpError",
                "URLError", "TimeoutError", "ConnectionError", "RemoteDisconnected"}


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() and key.strip() not in os.environ:
            os.environ[key.strip()] = value.strip()


def collection_completed_ids():
    done = set()
    for status_path in sorted(OUT.glob("task_status_*.jsonl")):
        for row in read_jsonl(status_path):
            if row.get("runtime_protocol") != runtime_protocol():
                raise ValueError("Teacher collection protocol mismatch; move old collection outputs out of the active directory before recollecting")
            if row.get("final") in {"accepted", "teacher_hard"}:
                done.add(row["record_id"])
    return done


def collection_summary():
    latest = {}
    usage = {"input": 0, "output": 0, "cached": 0}
    for path in sorted(OUT.glob("task_status_*.jsonl")):
        for row in read_jsonl(path):
            rid = row["record_id"]
            if latest.get(rid, {}).get("final") != "accepted":
                latest[rid] = row
            for key in usage:
                usage[key] += (row.get("usage") or {}).get(key, 0)
    return {"tasks": dict(Counter(row["final"] for row in latest.values())),
            "usage_total": usage}


def pick_tasks(limit, shard=0, shards=1):
    if limit < 1 or shards < 1 or not 0 <= shard < shards:
        raise ValueError("Invalid limit/shard/shards")
    tasks = list(task_index(TASKS).values())
    pool = sorted(
        (t for t in tasks if training_task(t)),
        key=lambda t: t["record_id"],
    )
    # 按 record_id 稳定分片：同一个任务永远落在同一个分片，双 Teacher 不重叠。
    pool = pool[shard::shards]
    done = collection_completed_ids()
    remaining = [t for t in pool if t["record_id"] not in done]
    # 难度分层：hard 至少 30%，其余近似源分布
    by_diff = defaultdict(list)
    for t in remaining:
        by_diff[t["difficulty"]].append(t)
    for members in by_diff.values():
        RNG.shuffle(members)
    n = min(int(limit), len(remaining))
    n_hard = min(len(by_diff.get("hard", [])), round(n * 0.30))
    n_rest = n - n_hard
    non_hard = by_diff.get("easy", []) + by_diff.get("medium", [])
    source_rest = len(non_hard)
    n_easy = round(n_rest * len(by_diff.get("easy", [])) / max(source_rest, 1))
    selected = (by_diff.get("hard", [])[:n_hard]
                + by_diff.get("easy", [])[:n_easy]
                + by_diff.get("medium", [])[:n_rest - n_easy])
    selected_ids = {t["record_id"] for t in selected}
    # Fill shortages in any stratum instead of silently returning fewer tasks.
    selected.extend(t for t in remaining if t["record_id"] not in selected_ids)
    selected = selected[:n]
    return selected


def make_client(name):
    load_env_file(ROOT / ".env")
    if name == "glm":
        return AnthropicTeacherClient(
            api_key=os.environ["TEACHER_API_KEY"],
            base_url=os.environ["TEACHER_BASE_URL"],
            model=os.environ["TEACHER_MODEL"],
        )
    client = OpenAIChatClient(
        model=os.environ["TEACHER2_MODEL"],
        base_url=os.environ["TEACHER2_BASE_URL"],
        api_key=os.environ["TEACHER2_API_KEY"],
        temperature=1.0, top_p=0.95, max_tokens=2048,
        timeout=120, thinking=False,
    )
    client.extra_headers = {"x-opencode-session": str(uuid.uuid4())}
    return client


def judge_trajectory(trajectory, jev, record, thread_clients):
    """在线验收：显式校验终止审计字段，替代购买必须经 Jev 判定。

    返回 (accepted: bool, reason: str, audit: dict)。audit 携带终止审计与
    （若有）JEV 结论，由 run_task 持久化到轨迹 meta 与任务状态，使每条
    轨迹的准入依据可独立追溯。
    """
    status = trajectory.get("status")
    terminal = trajectory.get("terminal_result") or {}
    reward_detail = terminal.get("reward_detail") or {}
    reward_type = reward_detail.get("reward_type")
    audit = terminal_audit(trajectory)
    if status != "done":
        return False, f"structural:{status}", audit
    # 准入硬性前置：完整终止状态 + reward_valid=true + over=true。
    if not terminal or not isinstance(reward_detail, dict) or not reward_type:
        return False, "terminal_incomplete", audit
    if terminal.get("over") is not True:
        return False, "terminal_not_over", audit
    if terminal.get("reward_valid") is not True or reward_detail.get("reward_valid") is not True:
        return False, "reward_invalid", audit
    if not terminal.get("termination_reason"):
        return False, "terminal_incomplete", audit
    problem = terminal_problem(trajectory, require_jev=False)
    if not problem:
        problem = message_protocol_problem(trajectory.get("messages") or [])
    if not problem:
        problem = conversation_problem(trajectory.get("messages") or [])
    if not problem:
        problems = (argument_problem(c) for m in trajectory.get("messages", [])
                    for c in m.get("tool_calls") or [])
        problem = next((value for value in problems if value), None)
    if problem:
        return False, problem, audit
    if reward_type == "gold_purchase":
        return True, "gold", audit
    if reward_type in {"valid_alternative_purchase", "partial_alternative_purchase"}:
        try:
            evidence = jev_evidence(trajectory)
        except ValueError as exc:
            return False, str(exc), audit
        audit["jev_input"] = evidence
        try:
            verdict = jev.decide_satisfaction(json.dumps(evidence, ensure_ascii=False, sort_keys=True))
        except JevApiError as exc:
            return False, f"jev_error:{str(exc)[:40]}", audit
        audit["jev_verdict"] = dict(verdict)
        if verdict["choice"] == "fully_satisfies":
            return True, "jev:fully_satisfies", audit
        return False, f"jev:{verdict['choice']}", audit
    return False, f"env:{reward_type or status}", audit


def terminal_audit(trajectory):
    """从终止结果提取审计快照：reward_valid/over/termination_reason/购买摘要。"""
    terminal = trajectory.get("terminal_result") or {}
    purchase = terminal.get("purchase") or {}
    return {
        "reward_type": (terminal.get("reward_detail") or {}).get("reward_type"),
        "reward_valid": terminal.get("reward_valid"),
        "over": terminal.get("over"),
        "termination_reason": terminal.get("termination_reason"),
        "purchase": purchase or None,
    }


def run_task(task, index_of, records_by_id, persona_pool, jev, teacher="glm"):
    """Claim a task across processes; POSIX locks release on process exit."""
    rid = str(task["record_id"])
    if not rid or any(c in rid for c in "/\\") or rid in {".", ".."}:
        raise ValueError("Invalid task ID")
    claim = OUT / "claims" / (rid + ".lock")
    claim.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        import fcntl
        with claim.open("a+", encoding="utf-8") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"Task already claimed: {rid}")
            try:
                return _run_task(task, index_of, records_by_id, persona_pool, jev, teacher)
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        # Keep the inode: unlinking a flock file can create two live locks.
    try:
        with claim.open("x", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
    except FileExistsError:
        raise RuntimeError(f"Task already claimed: {rid}; inspect {claim} before retry")
    try:
        return _run_task(task, index_of, records_by_id, persona_pool, jev, teacher)
    finally:
        claim.unlink()


def _run_task(task, index_of, records_by_id, persona_pool, jev, teacher="glm"):
    client = make_client(teacher)
    record = records_by_id[task["record_id"]]
    system = TEACHER_SYSTEM_PROMPT
    if task.get("persona_ref"):
        entry = persona_pool.get(task["persona_ref"])
        if not entry:
            raise ValueError(f"Missing persona: {task['persona_ref']}")
        system += ("\n\n以下是该用户的画像，属于软偏好参考，"
                   "与当前请求冲突时以请求为准：\n\n"
                   + render_persona(entry["persona"]))
    env_task = {"task_id": index_of[task["record_id"]],
                "prompt": [{"role": "system", "content": system}]}
    attempt = 0
    env_failures = 0
    TRAJ.mkdir(parents=True, exist_ok=True)
    # Persisted attempts survive interruption. JEV outages resume the saved
    # purchase rather than paying the Teacher to perform it again.
    pending = []
    for path in sorted(TRAJ.glob(f"{task['record_id']}_*.json")):
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("runtime_protocol") != runtime_protocol():
            raise ValueError("Saved trajectory protocol mismatch")
        meta = saved.get("meta") or {}
        attempt = max(attempt, meta.get("attempt", 0))
        if meta.get("accepted") and terminal_problem(saved) is None:
            return {"record_id": task["record_id"], "final": "accepted", "attempts": attempt,
                    "reason": meta["accept_reason"], "audit": meta["audit"],
                    "usage": dict(client.total_usage)}
        if (str(meta.get("accept_reason", "")).startswith("jev_error:")
                or meta.get("accept_reason") == "pending_review"):
            pending.append((path, saved))
    def verify_and_save(trajectory, number, path):
        accepted, reason, audit = judge_trajectory(trajectory, jev, record, client)
        trajectory["meta"] = {
            "record_id": task["record_id"], "teacher": teacher,
            "difficulty": task["difficulty"], "persona_condition": task["persona_condition"],
            "attempt": number, "accepted": accepted, "accept_reason": reason,
            "reward_type": (trajectory.get("terminal_result") or {}).get("reward_detail", {}).get("reward_type"),
            "steps": len(trajectory.get("steps") or []), "status": trajectory.get("status"), "audit": audit,
        }
        atomic_json(path, trajectory)
        if accepted or reason.startswith("jev_error:"):
            return {"record_id": task["record_id"], "final": "accepted" if accepted else "pending_jev",
                    "attempts": number, "reason": reason, "audit": audit,
                    "usage": dict(client.total_usage)}
        return None
    for path, saved in pending:
        result = verify_and_save(saved, saved["meta"]["attempt"], path)
        if result:
            return result
    while attempt < BUDGET:
        try:
            trajectory = collect_for_task(
                env_task, client=client,
                base_url=BASE_URL, max_steps=MAX_STEPS)
        except Exception as exc:
            if exc.__class__.__name__ not in INFRA_ERRORS:
                raise
            # Teacher/网络层异常属于基础设施故障，不消耗采样预算。
            env_failures += 1
            if env_failures > 12:
                return {"record_id": task["record_id"], "final": "env_exhausted",
                        "attempts": attempt, "usage": dict(client.total_usage)}
            time.sleep(min(60, 2 ** env_failures))
            continue
        error_type = ((trajectory.get("error") or {}).get("type") or "")
        if trajectory.get("status") == "error" and error_type in INFRA_ERRORS:
            # 基础设施故障不消耗采样预算；指数退避，最多容忍 12 次。
            env_failures += 1
            atomic_json(OUT / "infra" / f"{task['record_id']}_{uuid.uuid4().hex}.json", trajectory)
            if env_failures > 12:
                return {"record_id": task["record_id"], "final": "env_exhausted",
                        "attempts": attempt, "usage": dict(client.total_usage)}
            time.sleep(min(60, 2 ** env_failures))
            continue
        attempt += 1
        path = TRAJ / f"{task['record_id']}_{attempt}_{uuid.uuid4().hex}.json"
        trajectory["meta"] = {"record_id": task["record_id"], "teacher": teacher,
                              "attempt": attempt, "accepted": False, "accept_reason": "pending_review"}
        atomic_json(path, trajectory)
        result = verify_and_save(trajectory, attempt, path)
        if result:
            return result
    final = "teacher_hard"
    return {"record_id": task["record_id"], "final": final,
            "attempts": attempt,
            "usage": dict(client.total_usage)}


def main():
    global OUT, TRAJ, TASKS, PERSONAS, BASE_URL, MAX_STEPS
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=6000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--teacher", default="glm", choices=["glm", "deepseek"])
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, default=TASKS)
    parser.add_argument("--personas", type=Path, default=PERSONAS)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()
    if args.workers < 1 or args.max_steps < 1:
        parser.error("workers and max-steps must be positive")
    OUT, TASKS, PERSONAS = args.output_dir, args.tasks, args.personas
    TRAJ, BASE_URL, MAX_STEPS = OUT / "trajectories", args.base_url, args.max_steps

    TRAJ.mkdir(parents=True, exist_ok=True)
    if args.status:
        print(json.dumps(collection_summary(), ensure_ascii=False))
        return

    load_env_file(ROOT / ".env")
    source = ROOT / "environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz"
    from check_grpo_runtime import validate_reward_runtime_files
    validate_reward_runtime_files(json.loads((ROOT / "data/environment.json").read_text(encoding="utf-8")), ROOT)
    identity = {"version": PREPARATION_VERSION, "runtime_protocol": runtime_protocol(),
                "harness": harness_identity(ROOT), "tasks_sha256": file_hash(TASKS),
                "personas_sha256": file_hash(PERSONAS), "source_sha256": file_hash(source),
                "max_steps": MAX_STEPS, "budget": BUDGET, "shards": args.shards}
    if not (OUT / "manifest.json").exists() and (any(OUT.glob("task_status_*.jsonl")) or any(TRAJ.glob("*.json"))):
        raise ValueError("Unversioned collection directory; use a new batch directory")
    ensure_identity(OUT / "manifest.json", identity)
    model_key = "TEACHER_MODEL" if args.teacher == "glm" else "TEACHER2_MODEL"
    url_key = "TEACHER_BASE_URL" if args.teacher == "glm" else "TEACHER2_BASE_URL"
    ensure_identity(OUT / f"teacher_{args.teacher}.json",
                    {"model": os.environ[model_key], "base_url": os.environ[url_key],
                     "temperature": 1.0, "top_p": 0.95, "max_tokens": 2048})
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    index_of = {str(r["asin"]): i for i, r in enumerate(data)}
    records_by_id = {str(r["asin"]): r for r in data}
    persona_pool = {row["persona_id"]: row for row in read_jsonl(PERSONAS)}
    selected = pick_tasks(args.limit, shard=args.shard, shards=args.shards)
    print(f"本轮采集 {len(selected)} 题（预算 {BUDGET} 次/题，成功即停，"
          f"{args.workers} 并发）", flush=True)

    jev = JevDecisionsClient(api_key=os.environ.get("LLMOXY_API_KEY") or os.environ["JEVMODEL_API_KEY"])
    write_lock = threading.Lock()
    done_count = 0
    consecutive_infra = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_task, t, index_of, records_by_id,
                               persona_pool, jev, args.teacher): t
                   for t in selected}
        for future in as_completed(futures):
            result = future.result()
            result["runtime_protocol"] = runtime_protocol()
            with write_lock:
                status_path = OUT / f"task_status_{args.teacher}.jsonl"
                with status_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                done_count += 1
                if result["final"] in {"env_exhausted", "pending_jev"}:
                    consecutive_infra += 1
                else:
                    consecutive_infra = 0
                if done_count % 20 == 0:
                    print(f"进度 {done_count}/{len(selected)}", flush=True)
                # 全局熔断：连续 50 题基础设施失败说明外网或环境长时间不可用，
                # 停止烧队列，恢复后重启续跑。
                if consecutive_infra >= 50:
                    print("连续 50 题基础设施失败，熔断退出。恢复后重启续跑。",
                          flush=True)
                    pool.shutdown(wait=False, cancel_futures=True)
                    break
    report = collection_summary()
    atomic_json(OUT / "report.json", report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
