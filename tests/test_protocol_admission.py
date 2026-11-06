import copy
import json

import pytest

from scripts.build_sft_dataset import admission_problem
from scripts.evaluate_model import ensure_run_identity
from shopping_grpo.environment.observation import (
    OBSERVATION_VERSION, StructuredObservationError, render_structured_observation,
)
from shopping_grpo.environment.projection import ObservationProjectionError, project_observation
from shopping_grpo.environment.protocol import message_protocol_problem, runtime_protocol
from shopping_grpo.environment.tools import SHOP_TOOL_SCHEMAS, tool_call_to_action
from shopping_grpo.evaluation.rollout import STUDENT_SYSTEM_PROMPT, completed_task_attempts
from shopping_grpo.training.grpo.adapter.runtime import exploration_score


def test_single_tool_config_matches_runtime_and_has_eight_tools():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    configured = json.loads((root / 'configs/tools.json').read_text(encoding='utf-8'))
    assert [entry['tool_schema'] for entry in configured['tools']] == SHOP_TOOL_SCHEMAS
    assert len(SHOP_TOOL_SCHEMAS) == 8
    assert 'view_' not in json.dumps(SHOP_TOOL_SCHEMAS)
    assert 'view_' not in STUDENT_SYSTEM_PROMPT
    with pytest.raises(KeyError):
        tool_call_to_action('view_attributes', {})


def test_sft_rejects_old_protocol_and_retired_calls():
    trajectory = {
        'runtime_protocol': runtime_protocol(),
        'terminal_result': {'over': True, 'reward_valid': True,
                            'termination_reason': 'gold_purchase',
                            'reward_detail': {'reward_type': 'gold_purchase', 'reward_valid': True}},
        'messages': [],
    }
    assert admission_problem(trajectory) is None
    old = copy.deepcopy(trajectory)
    del old['runtime_protocol']
    assert admission_problem(old) == 'protocol_mismatch'
    trajectory['messages'] = [{'role': 'assistant', 'tool_calls': [
        {'function': {'name': 'view_attributes', 'arguments': '{}'}}]}]
    assert admission_problem(trajectory) == 'unsupported_tool:view_attributes'


def test_resume_requires_current_protocol(tmp_path):
    path = tmp_path / 'trajectories.jsonl'
    path.write_text(json.dumps({'task_id': 1}), encoding='utf-8')
    with pytest.raises(ValueError, match='different shopping protocol'):
        completed_task_attempts(path)
    path.write_text(json.dumps({'task_id': 1, 'runtime_protocol': runtime_protocol()}), encoding='utf-8')
    assert completed_task_attempts(path) == {(1, 0)}


def test_evaluation_identity_blocks_mixed_runs(tmp_path):
    identity = {'runtime_protocol': runtime_protocol(), 'model': 'fixture', 'max_steps': 35}
    ensure_run_identity(tmp_path, identity)
    ensure_run_identity(tmp_path, identity)
    with pytest.raises(ValueError, match='configuration changed'):
        ensure_run_identity(tmp_path, {**identity, 'max_steps': 10})


def test_retired_buttons_and_pages_fail_closed():
    state = {'observation_version': OBSERVATION_VERSION, 'page_type': 'search_home',
             'actions': ['Attributes']}
    with pytest.raises(StructuredObservationError, match='retired navigation'):
        render_structured_observation(state)
    raw = '可点击的按钮: ["Description"]'
    with pytest.raises(ObservationProjectionError, match='retired_action_footer'):
        project_observation('open_product', raw, count_tokens=len)
    assert message_protocol_problem([{'content': '[SHOPPING_OBSERVATION_V2]'}])


def test_detail_projection_preserves_all_evidence_or_errors():
    state = {'observation_version': OBSERVATION_VERSION, 'page_type': 'product_detail',
             'actions': ['buy now'], 'product': {'asin': '111111111111',
             'title': '完整商品型号 A20', 'price': 100,
             'key_attributes': ['属性' + str(i) for i in range(30)]},
             'available_options': {'颜色': ['白色']}}
    raw = render_structured_observation(state)
    visible, _ = project_observation('select_option', raw, count_tokens=len, detail_token_budget=4096)
    assert visible == raw
    assert '属性29' in visible
    with pytest.raises(ObservationProjectionError, match='complete product evidence'):
        project_observation('select_option', raw, count_tokens=len, detail_token_budget=128)


def test_exploration_requires_two_candidates_and_two_queries():
    steps = [{'tool': 'open_product', 'parameters': {'asin': value}} for value in ('A', 'B')]
    steps += [{'tool': 'search_products', 'parameters': {'query': 'one'}}]
    assert exploration_score({'steps': steps})[0] == 1
    # A retired view cannot inflate effort, even in an imported diagnostic trace.
    assert exploration_score({'steps': steps + [{'tool': 'view_features'}]})[0] == 1
    steps.append({'tool': 'search_products', 'parameters': {'query': 'two'}})
    score, detail = exploration_score({'steps': steps})
    assert score == 2
    assert detail == {'opened_candidates': 2, 'distinct_queries': 2}


def test_grpo_preflight_rejects_old_stored_prompt(tmp_path):
    import pandas as pd
    from scripts.check_grpo_runtime import validate_grpo_data_protocol
    path = tmp_path / 'tasks.parquet'
    row = {'prompt': [{'role': 'system', 'content': STUDENT_SYSTEM_PROMPT}],
           'extra_info': {'runtime_protocol': runtime_protocol()}}
    pd.DataFrame([row]).to_parquet(path, index=False)
    validate_grpo_data_protocol([path])
    row['prompt'][0]['content'] = '旧提示词：请查看信息子页'
    pd.DataFrame([row]).to_parquet(path, index=False)
    with pytest.raises(SystemExit, match='stale GRPO prompt'):
        validate_grpo_data_protocol([path])


def test_frozen_runtime_hashes_and_tool_config_pass_preflight(monkeypatch):
    from pathlib import Path
    from scripts.check_grpo_runtime import validate_environment_contract
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv('SHOPPING_ENV_MANIFEST', str(root / 'data/environment.json'))
    monkeypatch.setenv('SHOPPING_ENVIRONMENT_VERSION', runtime_protocol()['environment_version'])
    monkeypatch.setenv('SHOPPING_TOOL_CONFIG', str(root / 'configs/tools.json'))
    validate_environment_contract()


def test_teacher_collection_does_not_skip_old_protocol_tasks(tmp_path, monkeypatch):
    from scripts import collect_teacher_data
    monkeypatch.setattr(collect_teacher_data, 'OUT', tmp_path)
    path = tmp_path / 'task_status_glm.jsonl'
    path.write_text(json.dumps({'record_id': 'one', 'final': 'accepted'}), encoding='utf-8')
    with pytest.raises(ValueError, match='collection protocol mismatch'):
        collect_teacher_data.collection_completed_ids()
    path.write_text(json.dumps({'record_id': 'one', 'final': 'accepted',
                               'runtime_protocol': runtime_protocol()}), encoding='utf-8')
    assert collect_teacher_data.collection_completed_ids() == {'one'}
