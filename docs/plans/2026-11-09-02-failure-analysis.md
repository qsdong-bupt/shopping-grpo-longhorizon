# 2026-11-09-02：评测失败分类与证据分析

## 1. 要解决的问题

原评测报告主要按 Reward 类型和守卫次数展示结果，不能区分需求匹配错误、循环规则终止、动作参数错误、页面工具不匹配、环境故障和奖励不可验证。`repeat_loop` 只说明终止规则触发，不能单独证明模型进行了无效循环；一次可恢复的守卫拒绝也不应被算成整条轨迹失败。

## 2. 希望达到的目的

- 为每条轨迹生成一个互斥的最终分类。
- 保存可重叠的过程问题标签、事件次数、步骤证据和归因状态。
- 明确区分“观测到的规则事实”和“需要人工复核的原因归因”。
- 在 HTML 报告中按最终类别、过程问题和待复核状态筛选，并导出 JSONL 供后续分析。

## 3. 解决方案

新增 `src/shopping_grpo/evaluation/failure_analysis.py`：

- `analyze_trajectory` 只使用标准化轨迹、Reward v3、终局信息、步骤状态和守卫记录，不调用模型或 Judge。
- 最终类别包括：成功购买、需求匹配未通过、循环规则终止、连续动作被拒绝终止、环境或服务故障、奖励不可验证、其他/待复核。
- 过程标签可重叠，包括需求检查未通过、奖励证据不可验证、参数错误、页面工具不匹配、重复规格、规格振荡、循环触发、页面导航终止、少量新商品未计进展、超步和连续拒绝上限。
- 守卫原始原因始终保留；`schema_*`、非法 `option_id` 单独归为参数错误；`click_not_in_previous_observation` 在有页面信息时区分页面工具不匹配与目标不可见，无法确认时标记待查。
- 循环分析保存 `termination_trigger` 和实际计数，避免从历史记录缺失字段推断具体阈值；页面状态变化、重复状态和导航中的终止分别标注。
- `summarize_failure_analyses` 分开计算互斥类别的任务数、可重叠标签的任务数和事件次数，拒绝重复 task_id。

更新 `scripts/build_eval_report.py`：报告数据增加逐条 `failure_analysis`；HTML 增加分类表、问题表、最终类别筛选、过程问题筛选和“仅看归因待复核”筛选；打开任务详情时显示完整分类证据。生成报告时额外写出：

- `<report>.failure_analysis.jsonl`
- `<report>.failure_analysis.summary.json`

本模块为只读诊断，不改变 Reward、循环终止、工具守卫或模型输入。

## 4. 实验验证结果

验证状态：已完成；未启动训练或新模型评测。

本地测试：

```powershell
$env:PYTHONPATH=(Resolve-Path source_checkout/src).Path
python -m pytest -q source_checkout/tests/test_failure_analysis.py source_checkout/tests/test_eval_report.py
```

结果：**31 passed**。

使用历史 `qwen35_page_aware_tools_full_20261109` 的 1,092 条轨迹生成并验证报告：

```powershell
python source_checkout/scripts/build_eval_report.py `
  --run-dir experiment_results/loop_audit_20261109/outputs/evaluation/qwen35_page_aware_tools_full_20261109 `
  --output experiment_results/loop_audit_20261109/report_failure_analysis.html
```

服务器用项目 Python 重新逐条分析并与导出 JSONL 对比，结果 `server_local_analyses_identical=true`；无 pytest 依赖，因此服务器测试未运行。浏览器验证使用无头 Microsoft Edge：总表 1,092 行，循环筛选 30 行，重复规格筛选 42 行，待复核筛选 24 行，task 296 证据弹窗正常，JavaScript 错误 0。

| 最终分类 | 任务数 | 比例 |
| --- | ---: | ---: |
| 成功购买（含有效替代购买） | 544 | 49.8% |
| 需求匹配未通过 | 467 | 42.8% |
| 循环规则终止 | 30 | 2.7% |
| 连续动作被拒绝终止 | 5 | 0.5% |
| 环境或服务故障 | 0 | 0.0% |
| 奖励不可验证 | 42 | 3.8% |
| 其他 / 待复核 | 4 | 0.4% |

过程标签中较多的是：需求检查未通过 485 个任务/1573 个事件；奖励证据不可验证 82 个任务/83 个事件；重复选择已选规格 42 个任务/57 个事件；循环规则触发 30 个任务；页面导航中触发循环终止 15 个任务；目标不在当前页面（原因待查）8 个任务/21 个事件。

## 5. 实验结果分析（是否达到目的）

达到目的。报告现在能把最终结果、过程问题和证据分开呈现；过程问题可以与成功结果共存，避免把“后来恢复的参数错误”算成最终失败。task 296、444 等导航/循环混合轨迹被标为 `needs_review`，没有被自动断言为模型循环原因。

当前限制：分类依据仍是采集到的公开状态和 Reward/Guard 记录，不等于模型内部意图；历史轨迹中没有记录循环阈值配置时，只保留可见计数而不伪造具体触发分支；`click_not_in_previous_observation` 的旧原因无法恢复参数字符串是否存在空格差异。后续如需更精确归因，应在运行时记录页面快照、可用工具集合、状态指纹和守卫结构化原因。

报告及明细位置：`outputs/evaluation/qwen35_page_aware_tools_full_20261109/report_failure_analysis.html`、同名 `.failure_analysis.jsonl` 和 `.failure_analysis.summary.json`；服务器校验记录：`outputs/analysis/failure_analysis_20261109/validation.json`。原 `report.html` 未覆盖。

### 2026-11-09：过滤后全量评测启动记录

用户授权在迁移后的服务器启动全量评测。运行名称为 `qwen35_4b_failure_analysis_full_20261109`，题集为价格过滤后的 eval 全集 **1026 题**；不补回已剔除的66题。模型服务确认加载 `models/Qwen3.5-4B`、名称 `shopping-agent`；workers=8、temperature=0、max_tokens=512、max_steps=35。模型 `/v1/models` 与环境 reset/release 预检均成功，代码及任务主表哈希与已核验版本一致。

```bash
python scripts/evaluate_model.py \
  --name qwen35_4b_failure_analysis_full_20261109 \
  --workers 8
```

已通过独立后台进程启动；首次状态核验已完成5/1026题，均为done。当前状态为**评测中**，尚无最终模型效果结论。失败分析模块不改变模型行为，本轮用于采集当前 agent/harness 在过滤后题集上的结果并验证完整报告流程，不能把结果变化归因于分析模块。

运行目录：`outputs/evaluation/qwen35_4b_failure_analysis_full_20261109/`。

- `evaluation.log`：评测日志；`experiment_metadata.json`：题集、模型服务信息、参数和源文件哈希。
- `run_status.json`：运行阶段、进程号及最终退出码；`run_evaluation_and_report.py`：本次后台执行与报告生成脚本。
- 评测成功后自动执行 `scripts/build_eval_report.py --run-dir <运行目录>`，输出 `report.html`、`report.failure_analysis.jsonl`、`report.failure_analysis.summary.json`。
- 若评测或报告生成失败，状态文件分别标为 `evaluation_failed` 或 `report_failed`，保留日志；不静默重试或更换实验条件。

历史1092题报告保留，严格比较需在前后相同任务交集上重新统计。最终结果应继续补充本节。

用户随后要求完成200题时生成一次中途报告。已在本轮运行目录启动独立监测 `report_at_200.py`，读取完整JSONL记录，达到200条后冻结 `snapshot_200/`（原始轨迹、报告所需协议和任务ID清单），生成 `report_200.html`、`report_200.failure_analysis.jsonl` 和 `report_200.failure_analysis.summary.json`。监测每5秒检查一次，生成报告不停止全量评测；若不足200条时评测失败，则记录 `stopped_before_200`，不生成不足量报告。状态见 `report_200_status.json`，日志见 `report_200.log`。

中途报告使用**最先完成并写入文件的200条轨迹**，并发完成顺序不等同于历史固定200题子集，不能据此直接进行逐题前后对照。

200题中途报告已完成并核对总数和唯一任务数均为200：严格成功104/200（52.0%），有效替代购买1、部分满足购买68、错误购买15、循环终止8、连续拒绝无终局1、超步2、奖励不可验证1、基础设施故障0。报告生成后全量评测继续执行，核验时已超过200题；中途报告使用的200条快照保持固定。本地副本位于工作区 `experiment_results/qwen35_4b_failure_analysis_full_20261109/report_200.html`。

用户随后要求全量结束后删除200题报告，保留全量报告。已启动 `finalize_report.py`：等待全量报告生成成功，核验任务数/唯一任务ID与run_identity均为1026、分类明细ID与轨迹完全一致、HTML包含全量数据后，仅删除 `report_200.html`、`report_200.failure_analysis.jsonl`、`report_200.failure_analysis.summary.json` 三份中途报告文件。原始轨迹、200题快照与历史分析保留。若评测或报告失败则保留中途报告并记录状态；状态见 `final_report_status.json`。当前为等待全量完成。

### 2026-11-09：全量完成与报告替换结果

状态：**已完成**。评测及报告生成均以退出码0结束，最终核验时间为2026-11-09 15:55:26 UTC。1026条轨迹的任务ID唯一，且与本次运行身份中的任务集合完全一致；失败分类明细同样覆盖全部1026题，分类计数之和为1026。

| 指标 | 全量结果 |
| --- | ---: |
| 严格成功 gold_purchase | 525 / 1026（51.17%） |
| 有效替代购买（不计入严格成功） | 4 |
| 部分满足购买 | 354 |
| 错误购买 | 91 |
| 重复循环终止 | 31 |
| 连续动作被拒绝 / invalid_action_limit | 5 |
| 超步 max_steps | 7 |
| 奖励不可验证 | 9 |
| 基础设施错误 | 0 |

互斥最终分类为：成功购买529、需求匹配未通过445、循环终止31、动作拒绝终止5、基础设施故障0、奖励不可验证9、其他/待复核7。成功购买类别含4条有效替代购买，正式严格成功仍为525。

服务器及本地均已保留全量 `report.html`、`report.failure_analysis.jsonl`、`report.failure_analysis.summary.json`，三份文件下载后SHA256与服务器一致。按用户要求，服务器和本地均删除上述三份 `report_200*` 中途报告文件；原始轨迹、`snapshot_200/`、200题需求检查分析及历史统计保留。清理与校验记录见本次运行目录的 `final_report_status.json`。全量HTML的SHA256为 `83f77472790eb4d1774c9d7d6298b3178fb569e033aac73f58349ad98f741183`。

本地全量报告位置：`experiment_results/qwen35_4b_failure_analysis_full_20261109/report.html`；服务器位置：`outputs/evaluation/qwen35_4b_failure_analysis_full_20261109/report.html`。前文的“评测中”“等待全量完成”和中途报告路径描述为历史过程，当前状态以本节为准。

本轮达到全量评测、固定失败分类及报告生成的验收目的；尚不能据此认定模型能力提升。失败分析模块不改变运行策略，且本次题集已过滤到1026题，不能直接将结果与历史1092题总分比较。当前主要剩余问题是需求匹配未通过445题；循环31题、连续动作被拒绝5题及9题奖励不可验证仍需按轨迹证据分别排查。
