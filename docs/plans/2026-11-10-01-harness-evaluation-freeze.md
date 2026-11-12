# 2026-11-10-01：SFT前冻结harness与评测

## 1. 要解决的问题

准备进入SFT阶段，需要阻止提示、工具、观察、搜索、价格、奖励、循环判定和评测题集继续变化，导致训练前后结果不可比较。现有代码也不能仅凭Git提交辨认版本，需保存实际源码和输入数据指纹。

前序全量基线为 `qwen35_4b_failure_analysis_full_20261109`：1026题，严格成功525题（51.17%），基础设施错误0。参考[失败分类与全量评测记录](2026-11-09-02-failure-analysis.md)。

## 2. 希望达到的目的

保存可恢复、可校验的harness和评测版本；固定train/dev/eval边界；后续主要比较权重变化。明确现有教材是否可直接用于当前SFT协议。此次不启动训练、不再运行正式评测。

## 3. 解决方案

- 新增 `scripts/check_frozen_harness.py`，读取冻结manifest，校验文件SHA256和受保护代码目录的新增文件；有差异即退出1，不调用模型或环境。
- 冻结ID `harness_eval_20261110`，在服务器 `outputs/frozen/harness_eval_20261110/` 保存96份源文件、配置、输入数据、tokenizer/模型配置指纹，任务与提示清单、包版本、底座权重指纹、代码和数据快照。
- 快照包含基线轨迹及统计，不重复包含9GB底座权重和100MB HTML；原权重和完整报告保留原位，SHA256记录于manifest。下载本地副本位于工作区 `experiment_results/harness_eval_freeze_20261110/`。
- 单独保存Teacher/Student提示及collection harness identity。全量之后的rollout差异仅为Teacher提示第6–8条；不撤回已有Teacher改进。核验除Teacher赋值外AST一致，Student提示与全部1026个任务的运行身份完全一致。当前源码的完整哈希与基线rollout哈希不同，不宣称逐字相同。
- 冻结时 `data/environment.json` 已刷新actions.py指纹为当前守卫的 `e9fac906…`，与历史基线清单的 `9303ced3…` 不同；逐项对比仅此审计字段变化，搜索、奖励及运行参数未变。实际actions.py仍与本轮基线run_identity记录完全一致。冻结清单保存当前元数据，不覆盖历史实验中的旧元数据。
- 新增当前操作说明 `docs/harness-evaluation.md`，明确dev选检查点、eval最终对比、训练数据准入和显式前置校验。未改动evaluate_model、rollout、动作守卫、奖励或循环实现。

## 4. 实验验证结果

状态：冻结已完成；没有启动新的训练或评测。

```powershell
$env:PYTHONPATH=((Resolve-Path source_checkout/src).Path + ';' + (Resolve-Path source_checkout).Path)
python -m pytest -q source_checkout/tests/test_frozen_harness.py source_checkout/tests/test_protocol_admission.py source_checkout/tests/test_sft_preparation_pipeline.py
```

结果：28 passed，4 skipped。跳过项依赖实际tokenizer与训练运行环境，本次未作通过声明。新增测试覆盖源文件匹配、修改、缺失、新增运行代码及manifest路径越界。首次执行因本地未包含服务器模板而失败，补齐同版本模板后通过；没有修改断言或跳过失败项。

服务器逐题重建eval任务，与原run_identity完整相等；train16489、dev1026、eval1026，record_id全局唯一。服务器新冻结校验返回passed，96份文件匹配。底座、商品目录和输入数据只读。

现有SFT教材审计：train5783条、val108条，均来自train，但train缺少runtime_protocol等当前审计列，val也缺少当前协议和完整终局证据；collection manifest和export report缺失。保留旧文件并标记须按冻结协议重新采集/构建，不把旧教材当成已通过当前准入。

冻结manifest SHA256：`94f98d2f2d05c20884a4cb3add15e13ca79c9896819ccab795af75bf5cd6118b`。

快照SHA256：`a0fe59024dfe25192109c8bc14732a3f7ab6326137ff63e15c1c3951121eec3b`，大小42022089字节。具体检查结果见冻结目录的 `validation.json`、`receipt.json`、`existing_sft_audit.json`。

本地下载后同时校验manifest和快照总SHA256，并流式读取tar中全部96份冻结文件逐一比较SHA256，全部一致。服务器再次执行独立校验命令，返回 `status=passed`、`checked_files=96`、`problems=[]`。

## 5. 实验结果分析（是否达到目的）

达到harness和评测版本固定、保存及校验的目的。后续训练模型可复用1026题底座基线及相同评测条件；硬约束匹配、奖励不可验证及循环判定的已知问题保留，避免与训练收益混合归因。

尚未达到“可直接启动SFT”的状态：需先生成当前协议的合格教材，再按实际GPU确定4B训练配置。现有train_sft.sh默认9B与4进程，本次未改成一个尚未确认的硬件配置。冻结校验目前是显式命令，不是所有入口内置的自动拦截；操作者需在采集、构建和评测前执行。
