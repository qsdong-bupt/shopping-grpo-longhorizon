# 2026-11-08：选择规格后的商品与规格摘要

## 目的与范围

按用户确认的格式，将正常 select_option 返回中附加的“规格选择反馈”替换为“选择后的商品与规格摘要”，集中展示商品事实与实际选中状态。原始商品详情 observation 继续保留；摘要仍通过 selection_feedback 字段保存，并拼接到模型实际收到的 tool message 中。

本次不加入需求核对清单、自动购买建议、重复选择拦截或循环规则修改。不改工具 schema、option_id 映射、环境执行、价格解析、奖励及模型权重；不改训练适配器的返回内容。本次修改的是评测 Agent 的返回消息组装路径。

## 模型可见格式

```text
[选择后的商品与规格摘要]

商品标题：羽毛球拍袋单肩拍包拍袋仿水牛津布三层加厚单只2支装便携收纳包
品牌／店铺：gumahus古麦仕旗舰店
商品类别：运动/瑜伽/健身/球迷用品›羽毛球›羽毛球包
商品属性：单肩包、仿水、牛津布、便携
当前页面价格：57.0 元
当前已选规格：
- 颜色分类：【三层加厚·粉色】可装2支拍
尚未选择的规格轴：无
```

以上商品内容为格式示例，实际取自本次工具执行返回的公开 observation_state。

- 标题、品牌、类别、属性取自 product；不改写“仿水”等属性词，不推断额外功能。
- 页面价格沿用原 observation 规则：state.get("selected_price", product.get("price"))。展示值可能是规格价、商品默认价或价格区间，不声称已验证对应当前规格；缺失时显示“未提供”。
- 已选规格只取返回的 selected_options，不将请求参数直接当成已成功选中的规格；保留规格原始空白与符号。没有选中值时显示“- 无”。
- 尚未选择的轴由 available_options 的轴减去 selected_options 的轴得出；不额外断言这些轴必选。
- 原有“选择已生效／重复选择／未确认成功”说明和后续决策提示被此摘要替换。重复选择依旧执行现有逻辑，正常返回同一实际状态摘要。
- 其他工具、非商品详情页返回及守卫拒绝反馈不受本次修改影响。

## 修改文件

- src/shopping_grpo/evaluation/rollout.py：替换 _selection_feedback 的呈现内容；保留原消息拼接与审计字段。
- tests/test_selection_feedback.py：更新摘要、真实累计状态、双规格、未生效请求、空白保留、价格区间、零价、缺失信息及实际消息拼接的验证。
- tests/test_option_ids.py：更新选项 ID 执行后的摘要断言，保留 resolved_option 审计与环境选择测试。

服务器改动前备份：/root/autodl-tmp/selection-summary-backup-20261108。

## 验证与后续实验

服务器使用项目 .venv/bin/python 执行：

```bash
PYTHONPATH=/tmp/shopping-navigation-test-deps:src:.:environments/ShopSimulator/.venv-shopsim/lib/python3.12/site-packages:environments/ShopSimulator/shop_env \
  .venv/bin/python -m pytest -q \
  tests/test_selection_feedback.py tests/test_option_ids.py tests/test_rollout.py
```

结果：55 passed，1 个第三方 pytest 弃用警告。

本次未启动模型评测或训练，未重启环境服务，未改写历史轨迹和报告。新启动的评测进程将读取新消息代码，已运行进程不会自动更新。已有 rollout_sha256 身份校验会区分本次修改，后续实验需使用新运行名称；不得续写旧版本实验。

相同 200 题模型实验继续等待 GPU 恢复，与此前 option_id 修改及后续优化统一安排。此摘要替换可能改变重复选择的表现，目前仅验证了返回内容与执行链路，尚无模型效果结论。


## 2026-11-08：finish_without_purchase 工具描述收紧

### 修改内容

将 `finish_without_purchase` 的工具描述改为：

```text
已经充分搜索并检查多个候选商品、仍未找到符合用户核心需求和预算的商品时，结束当前购物任务且不购买任何商品，调用后任务立即结束。
```

本次修改强调该工具只用于充分搜索后的最终不购买决定，并明确调用后任务终止。本次实际修改位于 `src/shopping_grpo/environment/tools.py`。核对发现 `configs/tools.json` 仍为旧描述，尚未同步；评测系统提示保留原有主动结束规则，未随本次改动。工具参数 schema 和 `reason=no_suitable_product` 校验保持不变。运行使用 Python 中的工具定义，运行身份中工具 schema 哈希已随新描述变化。

### 实验

使用与上一轮相同的冻结 200 题、同一 Qwen3.5-4B shopping-agent 配置和 workers=8，使用新运行名称：
`qwen35_4b_finish_description_200_20261108`。

重点比较 `early_abstain`、严格成功、错误购买、repeat_loop、无终局和 `finish_without_purchase` 调用次数。此前基线为 `qwen35_4b_selection_summary_200_20261108`；由于本次运行同时继承 option_id 映射和商品摘要版本，结果用于评估工具描述的增量效果。

本节将在评测完成后补充 summary 和分析，不把单次 200 题变化解释为稳定总体收益。


### 实验完成与结果

200/200 条唯一任务完成，无基础设施错误。报告：outputs/evaluation/qwen35_4b_finish_description_200_20261108/report.html。

与上一轮运行身份逐项核对：200 个任务及完整 prompt 完全一致；模型服务名、地址、temperature=0、max_steps=35、max_tokens=512 一致；rollout、守卫、选项映射、observation 和环境执行文件哈希一致。只有 tools.py 和工具 schema 哈希变化。因此本次测量的是工具描述变化，系统提示并未修改。未验证底层服务权重文件指纹一致，耗时亦受负载影响。

| 指标 | 商品摘要基线 | 新工具描述 | 变化 |
| --- | ---: | ---: | ---: |
| 严格成功 | 103（51.5%） | 102（51.0%） | -1 |
| 过早放弃 | 10 | 16 | +6 |
| 合格主动停止 | 1 | 3 | +2 |
| 错误购买 | 13 | 11 | -2 |
| 部分满足购买 | 51 | 49 | -2 |
| 循环终止 | 7 | 9 | +2 |
| 步数上限 | 4 | 2 | -2 |
| 连续拒绝无终局 | 3 | 1 | -2 |
| 奖励不可验证 | 7 | 6 | -1 |
| 有效替代购买 | 1 | 1 | 0 |
| finish_without_purchase 调用 | 11 | 19 | +8 |
| 守卫拒绝次数 | 55 | 51 | -4 |
| 平均输出 token | 482.545 | 506.685 | +5.0% |
| 平均耗时（秒） | 30.064 | 29.826 | -0.8% |

直接核对所有主动停止：基线 11 条均只搜索 1 次，其中 10 条只打开 1 个商品；新版 19 条同样均只搜索 1 次，其中 14 条只打开 1 个商品，其余打开 5–8 次商品。打开次数未去重，不能等同于独立候选数。新版的一次打开即停止示例包括 task 337、955、243、100、1452、153。

结论：本次描述没有达到减少过早停止的目的；过早放弃增加 6 条，严格成功减少 1 条。即使描述包含“充分搜索并检查多个候选”，模型仍有 14 条只打开 1 个商品就停止，说明该自然语言前置条件并未可靠落实。不能由此认定单次变化稳定或显著，也不能证明所有过早停止都只源于描述。后续应检查环境结束资格的具体未满足条件、各轨迹当前候选与停止理由，再决定调整描述或增加状态相关反馈，避免通过强制购买消除放弃。
