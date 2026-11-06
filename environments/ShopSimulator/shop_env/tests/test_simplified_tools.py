"""Exercise real HTML, browser actions, public evidence and terminal purchase."""
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip('gym')
pytest.importorskip('flask')

from shopping_grpo.environment.observation import render_structured_observation
from web_agent_site.engine.config import load_config
from web_agent_site.engine.observation import product_summary
from web_agent_site.engine.reward_features import compile_reward_features
from web_agent_site.envs.web_agent_text_env import SimServer, WebAgentTextEnv


@pytest.fixture
def shop(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv('SHOP_ENV_CONFIG', str(root / 'configs/environment.json'))
    monkeypatch.delenv('SHOP_MAX_STEPS', raising=False)
    attributes = ['属性' + str(i) for i in range(15)]
    product = {
        'asin': '111111111111', 'title': '测试枕头', 'Title': '测试枕头',
        'brand': '测试品牌', 'shop_name': '测试品牌旗舰店', 'category': '家居›枕头',
        'query': '枕头', 'product_category': '枕头', 'instruction_text': '购买白色枕头',
        'attribute': attributes, 'Attributes': attributes, 'Price': 100, 'pricing': [100],
        'options': {'颜色': ['白色', '黑色']},
        'customization_options': {'颜色': [{'value': '白色', 'price': 100},
                                        {'value': '黑色', 'price': 120}]},
        'MainImage': '', 'option_to_price': {'白色': 100, '黑色': 120},
        'option_to_image': {},
    }
    instruction = {'instruction': '购买白色枕头，预算150元',
                   'attributes': [], 'instruction_options': ['白色']}
    goal = {'asin': product['asin'], 'category': product['category'],
            'price_upper': 150, 'instruction_text': instruction['instruction'],
            'instruction_simple': instruction['instruction'], 'goal_options': ['白色'],
            'user_persona': None, 'weight': 1,
            **compile_reward_features(instruction, product)}
    server = SimServer.__new__(SimServer)
    server.base_url = 'http://shop.test'
    server.environment_config = load_config(root / 'configs/environment.json')
    server.max_steps = 35
    server.product_item_dict = {product['asin']: product}
    server.all_products = [product]
    server.product_prices = {product['asin']: 100}
    server.goals = [goal]
    server.user_sessions = {}
    server.assigned_instruction_text = None
    server.search_engine = None
    server.shuffle_goals = server.shift_goals = False
    server.search_time = 0
    server.render_time = 0
    with patch('web_agent_site.envs.web_agent_text_env.get_top_n_product_from_keywords',
               return_value=[product]):
        env = WebAgentTextEnv(server=server, observation_mode='text')
        env.reset(idx=0)
        yield env, product


def test_search_open_select_and_buy_without_information_tools(shop):
    env, product = shop
    env.step('search[枕头]')
    search = env.structured_observation()
    assert len(search['products'][0]['key_attributes']) == 10
    env.step('click[111111111111]')
    html = env.browser.page_source
    for label in ('Description', 'Features', 'Reviews', 'Attributes'):
        assert label not in html
    state = env.structured_observation()
    assert state['page_type'] == 'product_detail'
    assert state['product']['key_attributes'] == product['attribute']
    assert '属性14' in env.observation
    assert '属性14' in render_structured_observation(state)
    assert not {'description', 'features', 'reviews', 'attributes'}.intersection(state['actions'])
    env.step('click[白色]')
    selected = env.structured_observation()
    assert selected['selected_options'] == {'颜色': '白色'}
    assert selected['selected_price'] == 100
    assert selected['product']['key_attributes'] == product['attribute']
    _, terminal, _ = env.step('click[Buy Now]')
    assert terminal['done'] is True
    assert terminal['reward_valid'] is True
    assert terminal['reward_detail']['reward_type'] == 'gold_purchase'
    assert terminal['purchase']['asin'] == product['asin']
    assert terminal['purchase']['options'] == {'颜色': '白色'}
    assert terminal['purchase']['attributes'] == product['attribute']


@pytest.mark.parametrize('label', ['Description', 'Features', 'Reviews', 'Attributes'])
def test_retired_raw_click_cannot_open_a_page(shop, label):
    env, _ = shop
    env.step('search[枕头]')
    env.step('click[111111111111]')
    before = env.browser.page_source, env.browser.current_url
    _, status, _ = env.step('click[' + label + ']')
    assert not status['done']
    assert (env.browser.page_source, env.browser.current_url) == before
    assert not status['progress']['runtime_progress_added']


def test_summary_attribute_limit_is_only_for_search():
    product = {'attribute': ['a' + str(i) for i in range(15)]}
    assert len(product_summary(product)['key_attributes']) == 10
    assert len(product_summary(product, full_attributes=True)['key_attributes']) == 15
