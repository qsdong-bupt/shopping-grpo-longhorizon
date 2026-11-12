# 2026-11-10-03：官方 DeepSeek 正式采集 5000 个 SFT 任务并记录预算耗尽失败

## 1. 要解决的问题

原代理教师接口出现超时与断连。切换官方 DeepSeek 后，5题试采及教材构建、token统计、Parquet全量读取验证均已通过。正式采集需要单独记录每题三次有效采样仍未获得Gold完整购买或JEV完全满足购买的情况，并与基础设施故障区分。

## 2. 希望达到的目的

- 按用户指定 `--limit 5000 --workers 8 --max-steps 35` 生成示范。
- 每题最多3次有效采样，成功即停；5000是任务数，不保证产生5000条合格教材。
- 锁定同一5000题供断点续跑；保存每次失败的原始证据。
- 保持固定工具harness及准入口径；自动加工已合格轨迹，不启动SFT训练。

2026-11-10 22:05用户追加要求：**accepted精确达到3200条停止；最终训练3072条、验证128条，按题目难度等比例分层**。该要求替代“全部5000题都要取得终态”的停止条件；原5000题仅作为固定候选任务池。

## 3. 解决方案

运行名：`deepseek_official_train5000_20261110`。

- 教师使用官方 `https://api.deepseek.com` 的 `deepseek-flash`；temperature=1.0、top_p=0.95、max_tokens=2048、thinking=false。
- 沿用 `harness_eval_fixed_tools_20261110`、Teacher prompt、完整7个工具、动作守卫、观察投影、奖励与终止规则。本轮不混入预算解析修复或新的教师策略。
- 批次 `control/collect_official.py` 沿用已实测的官方Flash显式关闭思考适配，记录请求耗时及成功返回模型，不打印凭据。源码与SHA256写入 `formal_batch_identity.json`，冻结harness源文件未修改。
- 首次选择5000个唯一train任务到 `selected_tasks.json`，保存收据；续跑从这个清单取未完成任务，避免再次从全体train中追加其他任务。
- 每题结束写 `task_outcomes/<record_id>.json`，包含原始用户请求、稳定record_id、环境task_id、难度、画像条件及逐次尝试的终局类别、JEV结论、轨迹相对路径和SHA256。
- 只有最终 `teacher_hard` 且实际保存了第1、2、3次未准入轨迹，才记为三次耗尽失败。缺证据的记录单独要求审计。
- `control/run_batch.py` 后台监督采集，每15秒汇总失败与进度。采集结束后自动构建学生prompt教材、使用Qwen3.5-4B tokenizer统计token并导出验证；最大长度16384，超长剔除不截断，train内部验证比例5%、seed=20261109。dev/eval不进入教材。

主要文件均在服务器项目目录下：

| 文件 | 内容 |
| --- | --- |
| `outputs/collection/deepseek_official_train5000_20261110/run.log` | 逐请求与逐任务进度日志 |
| 同目录 `selected_tasks.json`、`selected_tasks_receipt.json` | 固定5000题与清单SHA256 |
| 同目录 `trajectories/` | 所有有效尝试的原始轨迹 |
| 同目录 `exhausted_tasks.jsonl` | 三次有效尝试仍不准入的任务及逐次证据 |
| 同目录 `infrastructure_tasks.jsonl` | pending_jev、env_exhausted，不计模型三次失败 |
| 同目录 `audit_required_tasks.jsonl` | 终态与已保存尝试证据不一致的记录 |
| 同目录 `failure_summary.json` | 最新失败计数与原因汇总 |
| 同目录 `pipeline_status.json` | 采集及后处理阶段状态 |
| `outputs/sft_dataset/deepseek_official_train5000_20261110/` | 自动构建的合格教材及导出报告 |

续跑应使用本批次受审计入口，不能直接重跑原采集CLI，以免丢失固定任务清单和失败记录扩展：

```bash
cd /root/autodl-tmp/shopping-grpo-longhorizon-2.0
export SHOPPING_PROJECT_ROOT="$PWD"
.venv/bin/python -u outputs/collection/deepseek_official_train5000_20261110/control/run_batch.py
```

仅在确认该批次没有活跃进程时使用续跑命令。批次代码、任务清单或教师配置改变后不得继续写入同一批次。

### 22:05 停止条件与精确分层划分修订

- 当前入口改为 `control/run_target3200.py`；它调用 `collect_target3200.py`，复用原始任务池、教师请求适配、每题三次预算及原始manifest，并用 `accepted_target_amendment.json` 单独保存用户授权变更及新增控制脚本指纹。旧身份与旧控制脚本保留，不覆盖历史来源。
- 按剩余accepted名额限制活跃任务数：`in_flight <= min(8, 3200 - accepted)`。每个任务最多贡献一条accepted，达到目标时没有残留采集任务，从而不出现并发超额。
- 切换控制策略时终止旧监督/采集进程，释放已确认无其他客户端使用的环境租约后恢复。已落盘的轨迹、attempt预算、状态及失败记录全部保留；被中断的当前未落盘交互会重做。本次切换前汇总为accepted2784、teacher_hard191。
- 新导出入口 `control/export_exact3200.py` 按最终3200条教材的easy/medium/hard数量计算128条验证配额：先向下取整，再按最大余数分配剩余名额。总数精确为3072/128，各难度比例仅有整数取整误差。
- 相同空白归一化请求为不可拆分组。按seed=20261109排列请求组，通过子集和满足各难度整数配额；跨难度的同请求组一起处理。无法同时满足严格配额和请求隔离时明确失败，不静默打破隔离。
- 使用冻结exporter完成原有准入、tokenizer、字段证据、全量训练类读回和loss mask校验；仅在批次适配器中将已确定的请求分组指派传给exporter。`exact_split_plan.json` 和 `export_report.json` 保存每条ID的去向及难度计数。
- 为保留全部3200条accepted，不按旧16384阈值静默剔除或截断：若有长样本，导出长度上限取至少16384且能容纳最长样本的1024整数倍，实际 `max_tokens` 和最长样本长度写入报告。后续训练必须使用匹配长度配置；本次不启动训练。
- 新续跑命令为：

```bash
export SHOPPING_PROJECT_ROOT="$PWD"
.venv/bin/python -u outputs/collection/deepseek_official_train5000_20261110/control/run_target3200.py
```

后续不再使用上方原 `run_batch.py` 续跑命令。新实时日志为同目录 `run_target3200.log`，精确停止进度为 `accepted_target_progress.json`；完成停止后生成 `accepted_target_complete.json`。

## 4. 实验验证结果

状态：已启动，采集中。启动时间为2026-11-10 21:18（Asia/Shanghai）。

- 服务器与本地失败记录回归检查均5项通过，覆盖三次失败、第三次成功、JEV故障、缺少尝试证据和恢复后覆盖旧pending终态。
- 启动前冻结校验99个文件全部通过，0 problems。
- 已核对5000个record_id唯一且全部 `split=train`、`tag=train`；清单SHA256为 `ec4d8186530dd70c0d5a221db18d897c5ec5ac7af657e432a4c18375d7ec3738`。
- 初始进度快照：2026-11-10 21:19:08，46题已有结果，其中41题accepted、5题三次耗尽；基础设施待处理0题、证据异常0题。该计数是启动核验，不是最终结果。
- 已实际检查一条耗尽记录：record_id=939687533398、环境task_id=4912；第1、2、3次轨迹文件均存在，并记录拒收原因及指纹。
- 正式5000题的最终合格数量、失败分布、token分布和Parquet导出结果尚未产生，以本批次报告为准。
- 修订时本地和服务器相关测试均10项通过，覆盖精确128、按难度最大余数分配、同请求保持同侧、跨难度重复请求及不可行配额报错。新监督进程已启动，2026-11-10 22:06读到accepted2824且仍在推进。最终3200条及3072/128导出尚待完成。

## 5. 实验结果分析（是否达到目的）

正式采集已按指定参数运行，固定任务清单和三次耗尽失败记录已经生效。完整5000题采集及教材导出仍在进行，暂不能给出最终成功率或教材数量。

准入采用完整Gold购买或经冻结JEV标准判为fully_satisfies的替代购买，不能把“某次购买奖励有效”当作已通过SFT验收。接口或环境故障不会消耗三次模型采样预算，独立列出以便恢复；原有服务端模型别名变化风险及预算解析不足仍保留，不在本轮更改。

22:05修订已生效：目标改为3200条accepted和按难度分层的3072/128精确划分；不再追求5000个候选任务全部跑完。最终是否完成以 `accepted_target_complete.json`、`exact_split_plan.json`、`export_report.json` 及 `pipeline_status.json` 联合判断。


### 最终完成结果与质量复核（2026-11-10 追加）

采集22:12:30达到accepted3200且in_flight=0，22:17:16导出完成，pipeline=passed。3406个任务终态包括3200 accepted、206个任务三次耗尽，剩余1594个候选未继续跑，符合修订停止目标。Gold2931、JEV fully_satisfies269。

train3072（easy776/medium976/hard1320）、val128（easy32/medium41/hard55）；train/val ID及请求交叉0，冻结eval ID交叉0。140条超过16384，最长61968；导出上限62464不代表训练显存已经验证。

本轮质量审查未修改原始教材与冻结harness，未启动训练。详细过程分类、风险与使用方案见[质量审查记录](2026-11-10-04-sft-data-quality-audit.md)，报告在本批collection下quality_audit/report.md和report.html。上文仍在采集或等待结果的描述为历史快照。
