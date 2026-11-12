# SFT 阶段使用的冻结 harness 与评测

当前冻结版本：**`harness_eval_fixed_tools_20261110`**。本阶段调整模型权重；保持下面的 harness、数据和评测条件固定。如需改变其中任何一项，另建版本及底座对照实验，不覆盖本冻结快照或已有评测目录。前一版 `harness_eval_20261110` 保留为页面动态 schema 的历史冻结。

## 冻结范围与基线

| 项目 | 固定内容 |
| --- | --- |
| 历史底座参考 | Qwen3.5-4B，`qwen35_4b_failure_analysis_full_20261109`；该运行使用页面动态工具 schema，525/1026（51.17%），**不作为本固定工具版本的直接比较基线** |
| 正式评测 | 过滤后的 eval 全集1026题，任务顺序与逐题系统提示已保存 |
| 数据划分 | train16489 / dev1026 / eval1026；以冻结 record_id 清单为准 |
| 指标 | 完整终局、`reward_valid=true` 且 `gold_purchase` 为严格成功；本固定工具版本的底座基线待重跑 |
| 环境与奖励 | ShopSimulator Environment v2.2、Reward v3；搜索、规格价格解析及循环终止代码固定 |
| 工具与观察 | Observation v3、当前页面option ID、tools v5无模型停止工具；**每轮固定发送完整的7个工具 schema**；动作守卫仍依据当前页面和按钮验证调用；结构化拒绝反馈及选择后商品摘要固定 |
| 采样与边界 | temperature=0、top_p=1、thinking=false、max_tokens=512、max_steps=35、workers=8 |
| 上下文 | 24576窗口；压缩开启，安全余量512；观察预算1536，详情4096，通用768；搜索top_k=20 |
| 报告 | 固定确定性指标、失败分类及报告脚本 |

冻结资料位于项目内 `outputs/frozen/harness_eval_fixed_tools_20261110/`：`manifest.json` 记录harness、代码、数据和模型配置指纹，`snapshot.tar.gz` 保存可恢复的代码与数据快照，`receipt.json` 保存清单和快照SHA256。底座两份权重及索引另存SHA256，不重复打包9GB权重。快照保留基线原始轨迹、运行身份和结果明细；完整HTML保留在原评测目录。前一版快照不覆盖、不删除。

商品全目录保留不变，价格过滤只删除任务；后续仍可能买到价格不可解析的替代商品，所以奖励不可验证不保证归零。

全量评测后Teacher提示第6–8条调整过。Student系统提示、全部1026个任务的提示和元信息沿用历史基线；当前rollout还包含本次恢复固定工具schema的变化，不能再声称除Teacher提示外与历史基线完全一致。冻结快照保留当前Teacher提示和旧rollout证据；SFT样本使用Student提示，不能把Teacher决策提示写入训练样本。

服务器SFT准入客户端当前使用LLMoxy SystemOne的 `jev-latest`，新冻结保留其现有实现和rubric文本；该远端别名可能变动，采集需留存返回模型及endpoint审计。本地checkout另有DeepSeek准入分支，不能与服务器采集代码混用；本冻结以服务器快照为准。

## 每次采集、教材构建与评测前校验

在服务器项目根目录执行：

```bash
source .venv/bin/activate
python scripts/check_frozen_harness.py \
  --manifest outputs/frozen/harness_eval_fixed_tools_20261110/manifest.json
```

必须返回 `status: passed`。它只读校验源文件和数据SHA256，也会拒绝受保护目录里新增的运行代码。它不加载模型、不调用环境、不开始训练。校验是显式前置步骤，现有训练/评测入口尚未自动调用它；不要跳过此步骤。校验失败时先核查差异，不能直接覆盖冻结manifest使其“通过”。

## SFT 数据与检查点选择

- 仅从冻结 `split=train` 任务采集教材；train和SFT内部验证均不得使用dev/eval任务。SFT内部验证从train来源划分，按请求分组保持训练/验证无交叉。
- 使用冻结的采集、构建和导出流程。准入要求：完整购买、奖励可验证；gold购买直接准入，替代购买必须有冻结JEV标准下的 `fully_satisfies` 结论及匹配购买证据。动作守卫拒绝的assistant段不计算训练loss。
- 使用新采集目录和新教材目录，保留collection manifest、build identity、token manifest及export report。新教材需按Qwen3.5-4B tokenizer重新统计token并通过实际训练数据集类验证，禁止静默截断或沿用旧统计。
- **当前旧教材不能直接训练**：旧train.parquet有5783条，val.parquet有108条，虽均属于train任务，但缺少当前协议及完整准入审计列，采集manifest和export report缺失。详见冻结目录 `existing_sft_audit.json`。
- 用冻结dev1026题选择检查点，最终确定模型后再用eval1026题与相同固定工具版本的底座结果正式比较。旧的525/1026结果使用页面动态工具schema，与当前固定工具harness不完全一致；正式归因SFT效果前，应先在当前冻结版本上重跑底座eval。eval轨迹包括成功轨迹都不能回流训练或用于选择检查点。
- `scripts/train_sft.sh` 当前使用Qwen3.5-4B、单进程BF16全参数训练，读取精选1344条教材按20:1划分的1280条训练/64条验证数据。全局batch=8、动态token预算16384、学习率1e-5、训练1个epoch，禁用自动续训；启用console和SwanLab日志，需要设置 `SWANLAB_API_KEY`。`trainer.test_freq=-1` 关闭周期验证。具体训练超参数独立于冻结harness，运行前仍应核对脚本与实际启动参数。

## 训练后评测

模型和环境启动后，先运行冻结校验，再使用新的输出名称。dev检查点评测使用相同参数并加 `--split dev`；正式eval命令如下：

```bash
python scripts/evaluate_model.py \
  --name qwen35_4b_sft_full_RUN_ID \
  --split eval \
  --model shopping-agent \
  --temperature 0 --max-tokens 512 --max-steps 35 --workers 8

python scripts/build_eval_report.py \
  --run-dir outputs/evaluation/qwen35_4b_sft_full_RUN_ID
```

`RUN_ID` 替换为新训练实验标识。不加 `--limit`，不改任务ID子集。每次新模型使用新的评测名称，并记录实际检查点路径及权重SHA256；`shopping-agent`只是服务别名，不能用于识别实际权重。保留服务启动参数及版本，沿用24576上下文和qwen3_coder工具解析器；换GPU时耗时不作模型能力指标。

修改与验证过程见[2026-11-10-01冻结记录](plans/2026-11-10-01-harness-evaluation-freeze.md)。

本版恢复固定工具 schema 的修改记录见[2026-11-10-02](plans/2026-11-10-02-fixed-tool-schema.md)。动态工具加载前后实验报告仍作为历史证据保留，不代表当前冻结行为。
