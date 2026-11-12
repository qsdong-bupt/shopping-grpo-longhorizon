# 2026-11-07：当前页面规格 ID 映射

状态：程序修改和 CPU 验证已完成；模型 200 题评测待 GPU 恢复后，与后续优化一起安排。用户明确要求暂缓模型评测。

## 问题与目标

原 select_option 要求模型输出完整 value 字符串。空格、符号、转义变化可能导致守卫拒绝，或者守卫通过而环境没有选择成功。此前全量 544 条快照中，规格匹配问题涉及 24 条轨迹、58 次拦截；其中 5 条为纯空格差异、2 条为原始合法文本与守卫表示不一致，其余包含不存在或被改写的规格。并非这些问题都能靠映射解决。

本次把规格文本还原交给程序：模型从当前页面选择短 ID，程序按规格轴和原始选项值执行。目标是消除长规格字符串复制造成的错误，保留模型对商品和规格的选择责任。

## 接口与状态

```text
option_choices:
opt_1 → 尺码=S
opt_2 → 尺码=M
opt_3 → 颜色=粉色 【尼龙面料 独立鞋仓拍仓】
opt_4 → 颜色=[型号]黑色
```

```json
{"name":"select_option","arguments":{"option_id":"opt_3"}}
```

- 同一页面跨规格轴统一编号，ID 从 opt_1 开始。
- ID 表由当前公开 observation_state 中的 available_options 确定；保持原始规格轴和值，不进行语义猜测或模糊匹配。
- 规格轴按稳定顺序排列，同一轴保持环境提供的选项顺序；只改变 selected_options 时编号保持不变。
- 不同轴累积选择，同轴新值替换旧值；selected 标记随返回状态更新。
- 商品详情的可点击目标显示规格 ID，规格原文保留在 option_choices 中。导航按钮仍使用原导航工具。
- value 参数、多余字段、未知 ID、错误类型和非详情页选择均会被拒绝；详情页参数错误反馈直接返回可选 ID 表。
- ID 仅按最新页面解释。不同商品可以复用 opt_1，因此无法识别一个碰巧在当前页面也存在的历史 ID；模型仍必须读取最新 observation。
- 同名值可以出现在不同轴，执行请求包含 axis 和 value，避免按文字查找时选错轴。

## 实现

| 文件 | 更改 |
| --- | --- |
| src/shopping_grpo/environment/options.py | 新增统一 ID 表生成、读取和解析。 |
| src/shopping_grpo/environment/tools.py | select_option 改为 option_id；转换为携带原始 axis/value 的 click_option JSON 动作。 |
| src/shopping_grpo/environment/observation.py | 渲染 option_choices、selected 状态及 ID footer。原始环境 observation_state 仍保留。 |
| src/shopping_grpo/environment/actions.py | 按当前 ID 表校验，参数错误反馈提供当前选项，不再混入不相关的详情页导航诊断。 |
| src/shopping_grpo/evaluation/rollout.py | 更新参数约定；记录请求 ID、resolved_option、env_action，并按返回轴和值确认选择效果。 |
| src/shopping_grpo/training/grpo/adapter/tools.py | 训练适配器使用相同映射与结构化动作，保存解析值。未启动训练。 |
| configs/tools.json | 同步唯一工具 schema。 |
| src/shopping_grpo/environment/protocol.py | 工具版本变为 shopping-tools-v4-option-ids；加入 current-page-option-ids-v1，拒绝旧 value 调用进入新协议数据。 |
| src/shopping_grpo/environment/manifest.py、data/environment.json | 同步当前工具协议、映射版本及真实运行文件哈希。 |
| scripts/check_grpo_runtime.py | 将映射和守卫代码纳入运行文件哈希检查。 |
| scripts/evaluate_model.py | 运行身份增加映射、工具、渲染和环境执行代码哈希，防止混写实验。 |
| environments/ShopSimulator/shop_env/web_agent_site/engine/engine.py | 动作名按第一个左方括号解析，保留参数内部方括号、引号及空白。 |
| environments/ShopSimulator/shop_env/web_agent_site/envs/web_agent_text_env.py | 支持 click_option JSON；使用当前 HTML radio 的 name/value 定位，并沿用原有浏览器点击、奖励和进展计算。 |
| scripts/validate_option_mapping.py | 无 GPU 的固定 200 题历史页面映射检查工具。 |

环境执行内部仍将选规格记作 click 进展，因此本次没有调整循环阈值、奖励逻辑或有效翻页判定。

## 已完成验证

1. 相关工具、守卫、观察、投影、rollout、协议、manifest、训练适配器及统计测试：138 项通过，另有 10 个 subtest 通过。
2. 真实 HTML、浏览器、环境执行集成测试：10 项通过，其中新增 4 项覆盖粉色背包空格、双空格、方括号及双轴同名选项/同轴替换。
3. CPU 离线回放使用 outputs/evaluation/qwen35_4b_selection_feedback_200_20261107 的相同 200 条历史轨迹：199 条包含商品详情，639 个详情状态，6,969 次选项 ID → 原始规格轴/值 → 实际动作解析回读，失败 0 次。
4. 当前冻结 eval 数据的前 200 个 record_id 与该基线完全一致；任务 ID 和 record_id 清单已固定在同日期评测计划 JSON。

离线结果位于 outputs/validation/option_id_mapping_20261107/offline_validation.json。测试中的警告来自第三方依赖弃用提示。

以上证明已保存页面中的选项映射和执行解析可以无损往返，并验证了真实浏览器中的关键操作；不代表模型已经完成新版本 200 题，不提供修改后成功率。模型仍可能提交未知 ID、选择错误的有效 ID、误用历史页面 ID或重复选择。

## 待 GPU 恢复：相同 200 题模型验证

计划文件：docs/plans/2026-11-07-option-id-eval-200.json，状态 pending_gpu_model_evaluation。计划保存基线、200 个 task_id/record_id、模型配置、比较指标与待执行事项。

- 基线：qwen35_4b_selection_feedback_200_20261107，严格成功 101/200，循环 8，无终局 9。
- 设置：Qwen3.5-4B / shopping-agent，workers=8，max_steps=35，max_tokens=512，temperature=0。
- 新实验必须使用新名称，不能在旧协议结果中续跑。
- 先重启环境服务以加载 click_option 支持；旧进程不会自动加载修改后的代码。
- 评测前重新核对冻结 200 题清单。若期间继续优化，记录最终代码哈希和全部改动，不能将合并实验的效果全部归因于本次映射。

正式运行示例（本次未执行）：

```bash
python scripts/evaluate_model.py \
  --name qwen35_4b_option_ids_200_20261107 \
  --limit 200 \
  --workers 8

python scripts/build_eval_report.py \
  --run-dir outputs/evaluation/qwen35_4b_option_ids_200_20261107
```

重点检查：旧 value 调用、未知 ID、参数拦截次数、选择未生效、错误规格轴、连续拒绝、重复循环、严格成功、输入输出 token 与耗时。长文本复制错误应由新接口消除；有效 ID 的选择准确性必须由真实模型评测确认。
