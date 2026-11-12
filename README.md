# Shopping Agent V2.0

面向长程购物 Agent 的可复现后训练与评测项目。

```text
数据验收 → 画像与切分 → 双 Teacher 教材采集 → veRL 全参 SFT → 在线 GRPO → 冻结评测
```

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![veRL](https://img.shields.io/badge/veRL-0.9.1-0E8A16)](https://github.com/verl-project/verl)
[![vLLM](https://img.shields.io/badge/vLLM-0.25.0-0E6DB8)](https://github.com/vllm-project/vllm)
[![ShopSimulator](https://img.shields.io/badge/Environment-ShopSimulator%20v2.2-4C78A8)](https://arxiv.org/pdf/2601.18225)

## 环境与依赖

- Python 3.12 容器路线，依赖由 `pyproject.toml` / `uv.lock` 固定（uv 管理）
- `verl==0.9.1`、`vllm==0.25.0`、`torch==2.11.0`、`transformers>=5.5.3,<5.11,!=5.6.0`
- 当前 SFT 模型：`models/Qwen3.5-4B/`，单卡 BF16 全参数微调；训练配置见 `scripts/train_sft.sh`
- API Key（Teacher / Jev）在 `.env`，参考 `.env.example`

```bash
bash scripts/setup.sh        # 安装环境依赖与 ShopSimulator
```

## 工作流程

| 阶段 | 说明 | 入口 |
|---|---|---|
| Baseline | 未训练底座在冻结 eval 切分上的基线轨迹 | `bash scripts/serve_model.sh models/Qwen3.5-4B` + `python scripts/evaluate_model.py --name baseline` |
| 数据验收 | 两级验收（本地硬约束 + Jev 语义）+ 复查定案 | `scripts/run_acceptance.py` |
| 画像与切分 | 三条件画像分配、冻结 eval 切分 | `scripts/build_split.py` |
| Teacher 采集 | 双 Teacher 分治，每题预算 3 次成功即停 | `scripts/collect_teacher_data.py` |
| 教材加工 | Student 版提示词移植 + 逐消息 train 标记 + 真实 token 超长过滤 | `scripts/build_sft_dataset.py` + `scripts/export_sft_parquet.py` |
| SFT 训练 | veRL fsdp_sft + 自定义多轮工具数据集 | `bash scripts/train_sft.sh` |
| GRPO | 在线 RL，Reward v3 终局信号 | `scripts/train_grpo.py` |
| Evaluation | 训练后检查点的冻结评测（严格成功指标） | `bash scripts/serve_model.sh <checkpoint>` + `python scripts/evaluate_model.py --name <run>` |

评测入口统一使用 `scripts/evaluate_model.py`：环境由 `scripts/start_environment.sh`
启动，模型经 `scripts/serve_model.sh` 以 OpenAI 兼容接口 serve；轨迹落盘
`outputs/evaluation/<name>/trajectories.jsonl`（支持断点续跑），严格成功等指标
汇总在 `summary.json`。正式评测需用户明确指令后才执行。

设计决策与执行记录见 `docs/plans/`（基线、交接、数据验收记录、工作总结）。

当前交互协议为 Environment v2.2 / observation v3 / tool schema v5，每轮固定发送全部 7 个工具 schema。
规格选择使用当前页面的 `option_id`；动作守卫仍根据页面和按钮检查合法性。模型没有主动停止工具。
商品详情页直接提供全部属性，已移除描述、评论、属性、特点四个信息子页及其按钮。
旧协议的示范、SFT/GRPO 数据和评测输出不能直接混入新运行；需重新采集或构建并使用新的评测名称。
实现与验证范围见 [移除信息页工具的变更记录](docs/plans/2026-11-06-remove-information-tools.md)。
当前冻结规则和评测操作见 [SFT 阶段 harness 与评测](docs/harness-evaluation.md)。

当前训练脚本读取 1280 条训练数据和 64 条内部验证数据，默认训练 1 个 epoch，
全局 batch 为 8、动态 token 预算为 16384。启动前需设置 `SWANLAB_API_KEY`；
训练指标同时输出到控制台和 SwanLab。`trainer.test_freq=-1` 关闭周期验证。
本仓库同步代码、配置和文档；模型权重、完整数据集、冻结快照及运行产物需在运行环境中单独准备。

## 评测契约

- 正式评测严格成功 = 完整 `gold_purchase` 终止结果且 `reward_valid=true`。
- SFT 教材准入 = `reward_valid=true` 且（gold 或经冻结 Jev 判定 `fully_satisfies` 的替代购买）。
- 训练数据与冻结 `tag=eval` 评测集严格去重；旧版 Final-200 已退出 V2，仅存于 Git 历史作迁移核对。

训练、模型合并与正式评测需用户明确指令后才执行。
