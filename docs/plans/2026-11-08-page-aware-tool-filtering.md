# 2026-11-08：按页面动态限制工具与结构化守卫反馈

## 背景

全量评测 `qwen35_no_model_stop_full_20261108` 已生成 1092 条轨迹。未完成/无终局共 27 条；其中 27 条都包含在商品详情页误调用 `open_product`，累计出现 208 次、涉及 107 条轨迹。详情页误调用会被动作守卫连续拦截，达到 `invalid_action_limit` 后结束，是当前无终局的共同高频原因。

本次修改针对两个问题：一是守卫反馈把某个返回动作写成“下一步必须调用”，容易把模型固定到单一路径；二是模型在每个页面仍能看到全量工具，增加了调用当前页面不可能执行的工具的机会。

## 修改内容

### 1. 结构化动作拒绝反馈

`src/shopping_grpo/environment/actions.py` 的 `action_guard_tool_message` 统一返回：

```text
[动作被拒绝]
当前页面：product_detail
请求工具：open_product
原因：商品详情页不能打开其他商品
可用工具：select_option、buy_now、prev_page、back_to_search
```

反馈只列出当前页面实际可用的工具，不再输出“下一步必须调用”或把 `prev_page({})` 写成强制路径。页面状态错误仍给出页面级原因；schema 或参数错误（例如缺少字段、额外字段、非法 `option_id`）保留原始分类，避免用页面导航信息掩盖参数问题。

### 2. 页面级工具白名单

新增 `available_tools_for_observation(observation)`，并在每次请求模型前根据最新 observation 过滤 function schema：

| 页面 | 可暴露工具 | 动态条件 |
| --- | --- | --- |
| `search_home` | `search_products` | 搜索功能标记为可用 |
| `search_results` | `open_product`、`next_page`、`prev_page`、`back_to_search` | 当前存在商品 ASIN 或对应按钮 |
| `product_detail` | `select_option`、`buy_now`、`prev_page`、`back_to_search` | 当前存在规格 option ID 或对应按钮 |

过滤使用当前 observation 的页面类型和按钮列表；缺少页面类型时保留原有全量 schema 作为兼容回退，避免异常 observation 导致模型无工具可用。守卫反馈中的“可用工具”使用同一函数，保证模型看到的工具列表与错误反馈一致。

### 3. 测试

新增并更新回归测试，覆盖：

- 三种页面的工具白名单和按钮缺失时的动态收缩；
- rollout 每轮根据最新页面只发送允许的 schema；
- 详情页 `open_product`、`next_page`、`search_products` 的页面级原因；
- schema/参数错误不会被错误改写成导航原因；
- 拒绝反馈不包含强制的“下一步必须调用”。

本地相关测试命令：

```bash
PYTHONPATH=source_checkout/src;source_checkout \
python -m pytest -q \
  source_checkout/tests/test_action_validation.py \
  source_checkout/tests/test_rollout.py \
  source_checkout/tests/test_shop_tools.py
```

## 验证与后续实验

本次代码修改不影响已经启动的 `qwen35_pagination_progress_200_20261108` 评测；该进程启动时尚未加载页面级工具过滤。验证本修改时应重启模型/环境服务（若服务加载了旧 checkout），使用新的运行名称执行相同 200 题，并单独比较：

- 详情页 `open_product` 守卫拒绝次数及涉及轨迹数；
- `invalid_action_limit` 未完成数；
- 每轮模型实际收到的工具集合；
- 严格成功、部分购买和重复循环是否发生回退。

页面过滤只能减少不可能的工具调用，不能替代模型对候选、规格、预算和购买时机的判断；参数映射和环境资源错误仍需单独统计。

## 后续校正：搜索结果按钮交集

复核页面过滤实现时发现，`search_results` 不应仅依据正文中出现的 ASIN 暴露 `open_product`；ASIN 还必须出现在当前 observation 的“可点击的按钮”列表中。已在 `available_tools_for_observation` 中加入 ASIN 与当前按钮的交集检查，并新增回归测试。该修正不改变正常结构化 observation 的行为，只避免异常或截断 observation 暴露过期商品目标。

本次页面过滤实验因服务器 GPU/环境服务不可用暂未完成；之前误启动的商品摘要复测与本优化无关，不作为本实验结论。

## 2026-11-08：方案一致性复核

复核确认当前实现与本方案一致：

- `action_guard_tool_message` 返回当前页面、请求工具、拒绝原因和当前可用工具，不再输出强制的“下一步必须调用”。
- 页面级工具过滤已覆盖 `search_home`、`search_results` 和 `product_detail`；缺少 `page_type` 时保留全量工具回退。
- `search_results` 只有在 ASIN 同时出现在商品列表和当前可点击按钮列表中时才暴露 `open_product`。
- rollout 每轮根据最新 observation 过滤发送给模型的 function schema。
- schema/参数错误保留原始错误分类，不被页面导航提示覆盖。

本地相关回归测试共 88 项通过，服务器端 `py_compile` 通过。记录中的 Bash 测试命令已统一使用 `:` 作为 `PYTHONPATH` 分隔符；服务器 GPU 和 ShopSimulator 服务关闭期间不进行效果评测，因此没有把环境连接失败计入本方案效果结论。

## 2026-11-09：页面级工具过滤全量评测

使用新服务器、同一 1092 题冻结评测集和 `qwen35_page_aware_tools_full_20261109` 运行名称完成全量评测。模型和环境均正常，基础设施错误为 0；报告位于 `outputs/evaluation/qwen35_page_aware_tools_full_20261109/report.html`。

| 指标 | 页面过滤前：`qwen35_no_model_stop_full_20261108` | 页面过滤后：`qwen35_page_aware_tools_full_20261109` | 变化 |
| --- | ---: | ---: | ---: |
| 严格成功 | 563/1092（51.56%） | 541/1092（49.54%） | -22，-2.01 个百分点 |
| 部分满足购买 | 341 | 375 | +34 |
| 错误购买 | 86 | 92 | +6 |
| 重复循环 | 30 | 30 | 0 |
| 达到步数上限 | 9 | 4 | -5 |
| 奖励不可验证 | 31 | 42 | +11 |
| 未完成/无终局（unknown） | 27 | 5 | -22 |
| 有效替代购买 | 5 | 3 | -2 |
| 基础设施错误 | 0 | 0 | 0 |

守卫拒绝对比：

- 详情页误调用 `open_product`：208 次/107 条轨迹 → 21 次/8 条轨迹；减少 187 次、99 条涉及轨迹。
- `next_page` 非当前目标：124 次/79 条轨迹（本轮未出现该拒绝类型）。
- `prev_page` 非当前目标：9 次/3 条轨迹（本轮未出现该拒绝类型）。
- `select_option` 页面不可用：3 次/3 条轨迹（本轮未出现该拒绝类型）。

结论：页面级工具过滤显著减少了不可能动作和 `unknown` 无终局，基本解决了详情页误调用 `open_product` 的高频问题；但严格成功率下降 22 题，部分满足和错误购买上升，说明过滤减少非法调用后，模型仍存在候选比较、购买决策和需求满足问题。该修改达成了“减少动作守卫误用”的局部目标，不能单独视为整体购物成功率提升。后续应逐条分析由严格成功转为部分/错误购买的配对轨迹，并检查模型收到的空工具列表或页面状态变化是否造成决策退化。

## 2026-11-09：统一 `open_product` 当前按钮校验

将 `action_reject_reason` 的 `open_product` 校验与页面级工具过滤统一：ASIN 必须同时存在于当前 observation 的商品列表和“可点击的按钮”列表。若只出现在正文商品列表、但当前按钮不可点击，守卫返回 `click_not_in_previous_observation`，不会继续执行环境动作。

该修改主要防止旧 observation、截断 observation 或模型忽略动态 schema 时使用过期商品目标；正常结构化 observation 的合法打开行为不变。新增回归测试覆盖“ASIN 在商品列表但不在当前按钮列表”的情况。本地相关测试为 89 项通过。
