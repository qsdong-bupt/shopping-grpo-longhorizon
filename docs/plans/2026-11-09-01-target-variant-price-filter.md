# 2026-11-09-01：剔除目标商品目标规格价格不可解析任务

## 1. 要解决的问题

现有验收结果不能保证目标规格价格可验证：历史硬检查只在识别到预算上限时检查价格，且后续合并可让硬检查 unverifiable 的任务通过语义验收。目标商品有多个影响价格的规格轴、又没有完整组合价格时，即使模型选择正确商品与规格，Reward v3 仍无法验证价格。

用户要求从训练集、开发集和测试集中移除这类任务。此次处理任务准入数据，保留商品库和历史评测。

## 2. 希望达到的目的

- 所有保留任务的目标规格均能由当前环境的 `resolve_variant_price` 返回确定价格，不以是否识别到预算为检查前提。
- 保持剩余任务的原 split、record_id、task_id、画像、prompt 与顺序；不重新抽样或补齐数量。
- 同步处理现有训练派生文件，保存原文件备份、删除明细与校验结果。

## 3. 解决方案

新增 `scripts/filter_unresolvable_price_tasks.py`，默认只预览，传入 `--apply` 才备份并过滤。

1. 流式读取 `items_eval_train.json` 与 `fine_items_eval_train_all.json.gz`，逐条核对商品规格与价格字段指纹。两者不一致时停止处理，避免按错误的价格源过滤。
2. 用 `instructions[0].instruction_options` 定位目标规格，向环境原版 `resolve_variant_price` 传入完整商品，保留 `pricing`、`customization_options`、`variant_combinations`。缺少目标规格映射或价格返回非 pass 时排除，不推算最低价、最高价或价格之和。多规格本身不构成排除条件；完整组合价格可解析时保留。
3. 主表为 `outputs/split/tasks_final.jsonl`。按 record_id 过滤；仅有 task_id 的派生集使用原始源数据索引映射。扫描现存 GRPO 和两个 SFT 版本的活动文件，不改写 `_audit`、历史采集轨迹、历史验收结论和评测报告。
4. 所有待修改文件先完整备份，再逐文件替换；同步更新切分统计、GRPO 元数据中的数量/哈希和数据说明。无变化时不再执行写入。

复用命令（项目根目录，已激活项目虚拟环境）：

```bash
python scripts/filter_unresolvable_price_tasks.py
python scripts/filter_unresolvable_price_tasks.py --apply
```

本次没有重写历史 acceptance 判定逻辑。以后从历史验收结果重新构建任务或派生集时，必须再次执行本过滤并确认 dry-run 为零，避免把这些任务重新引入。

## 4. 实验验证结果

状态：数据过滤及验证已完成；未启动模型评测或训练。

执行命令：

```bash
python scripts/filter_unresolvable_price_tasks.py --apply
python scripts/filter_unresolvable_price_tasks.py
python -m pytest -q source_checkout/tests/test_target_price_filter.py
```

前两条在服务器项目根目录执行；最后一条在本地工作区执行，结果为 **8 passed**。覆盖多价格轴、完整组合价、双规格但单价格轴、无预算、缺失目标规格/指令、单一价格、价格区间和流式解析边界。

| 任务集合 | 过滤前 | 移除 | 保留 |
| --- | ---: | ---: | ---: |
| train | 16964 | 475 | 16489 |
| dev | 1050 | 24 | 1026 |
| eval/test | 1092 | 66 | 1026 |
| 合计 | 19106 | 565 | 18541 |

这 565 条的价格解析方法全部为 `multiple_effective_price_axes`。保留任务复扫：不可解析目标价格 **0 条**，派生文件待过滤变化 **0 项**。

| 派生文件 | 过滤前 | 移除 | 保留 |
| --- | ---: | ---: | ---: |
| data/grpo/train.jsonl 与 train.parquet（各自） | 1000 | 52 | 948 |
| data/grpo/validation.jsonl 与 validation.parquet（各自） | 50 | 1 | 49 |
| outputs/grpo/smoke.parquet | 64 | 4 | 60 |
| outputs/grpo/train_1000.parquet | 1000 | 117 | 883 |
| outputs/grpo/train_1000_curriculum.parquet | 1000 | 117 | 883 |
| outputs/sft_dataset/train.parquet | 5852 | 69 | 5783 |
| outputs/sft_dataset/val.parquet | 110 | 2 | 108 |
| outputs/sft_dataset_qwen35_4b_checked_20261105/train.parquet | 5526 | 61 | 5465 |
| outputs/sft_dataset_qwen35_4b_checked_20261105/val.parquet | 109 | 2 | 107 |

派生文件存在重叠，且有历史主表之外的源任务，不能将各行移除数相加作为主表移除数。文件名沿用原路径，实际数量以表格和 parquet 内容为准。

服务器项目内产物：

- 备份：`outputs/backups/price_filter_20261109_162004/`，保留原相对路径。
- 审计：`outputs/analysis/price_filter_20261109_162004/manifest.json`，含主表前后数量、源文件哈希和修改文件哈希。
- 删除明细：同目录 `excluded_tasks.jsonl`，含全源数据中不可解析任务及其 active split/outside_active_split 标记、目标规格、解析结果。主表删除数应只计 train/dev/eval。
- 校验：同目录 `verification.json` 与 `rescan.json`。

独立校验通过：所有备份哈希一致；所有保留行与备份按删除集合过滤后的内容、顺序完全一致；train/dev/eval 无交集；GRPO JSONL/parquet 的任务 ID 和顺序一致；源商品数据未改动。

## 5. 实验结果分析（是否达到目的）

达到本次数据过滤目的：565 条目标规格价格无法确定的任务已从主表移除，活动训练派生文件也完成清理，复扫无残留。此结果验证数据准入条件，不代表模型能力提升。

商品库仍保留原商品，因此模型购买其他价格不可解析的候选时，仍可能产生奖励不可验证。筛掉目标不可解析任务不等于保证所有探索路径都能验证价格。

后续全量评测分母为 1026，必须使用新的运行名称；历史 1092 题评测和 200 题子集保留原结果。直接比较不同题集的成功率不能将变化归因于模型或 harness 改进。需要比较时，应使用前后相同的保留任务交集。
