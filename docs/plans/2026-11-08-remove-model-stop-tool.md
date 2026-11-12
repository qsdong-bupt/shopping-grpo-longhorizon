# 2026-11-08：移除模型主动停止工具

## 修改内容

为减少模型过早调用 `finish_without_purchase` 结束任务，本次从模型可见工具协议中删除该工具。

- `src/shopping_grpo/environment/tools.py` 删除停止工具 schema 和 `finish[...]` 映射；工具协议更新为 `shopping-tools-v5-no-model-stop`，工具数从 8 个变为 7 个。
- `configs/tools.json` 与 Python 工具 schema 同步。
- `src/shopping_grpo/environment/actions.py` 删除停止工具专用参数放行分支；旧停止调用在新协议中按未知工具处理。
- `src/shopping_grpo/evaluation/rollout.py` 删除 Teacher 的主动停止提示；Student 提示改为任务未购买时由环境运行边界结束。
- `src/shopping_grpo/evaluation/metrics.py`、报告和轨迹分析脚本不再把停止工具作为新版可调用动作统计。
- `data/environment.json` 同步工具版本和工具 schema 运行哈希。
- 更新工具、协议、rollout、数据集和 manifest 相关测试。

Reward v3 的数值、购买质量判定、`early_abstain` 和 `graceful_stop` 历史结果分类均保留，用于兼容历史轨迹分析；本次没有修改奖励计算。新版任务通过 `buy_now` 成功结束，其他任务由步数上限、循环、非法调用上限或环境错误等运行边界结束。

## 验证状态

修改已完成在本地 `source_checkout`，服务器同步和测试待完成。模型评测尚未启动。后续评测必须使用新运行名称，不能续写包含旧工具协议的运行。
