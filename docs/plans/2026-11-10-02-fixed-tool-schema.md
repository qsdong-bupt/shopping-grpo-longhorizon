# 2026-11-10-02：恢复固定工具 schema 并重新冻结

## 1. 要解决的问题

当前 SFT 准备版在每次模型请求前，按 `page_type` 和当前按钮筛除工具 schema。这样会令同一轮对话在不同页面请求不同工具前缀，也使本次 SFT 数据生成和后续评测基于页面动态工具版本。按页面过滤全量实验里，严格成功由563/1092降至541/1092，详情页误调 `open_product` 及无终局明显减少，但该实验同时改变了动作守卫反馈，不能单独归因于工具过滤。

## 2. 希望达到的目的

- 每次模型请求收到相同完整工具 schema，固定工具前缀，适用于冻结 harness 下生成 SFT 示范和评测。
- 保留页面状态、按钮和参数的执行前动作守卫，避免固定加载工具被误解为所有页面都能执行所有工具。
- 新建可校验的harness冻结，不覆盖动态工具历史版本、基线评测或轨迹。

## 3. 解决方案

- `src/shopping_grpo/evaluation/rollout.py` 不再按 observation 过滤 API `tools` 参数；每轮将完整 `SHOP_TOOL_SCHEMAS` 传给模型。删除 rollout 的页面过滤依赖。
- `action_reject_reason` 继续依据最新 observation 校验工具调用；`action_guard_tool_message` 仍按页面返回可用动作作为拒绝后的诊断信息。它不影响正常模型请求中的 `tools` 列表。
- 更新 rollout 测试，跨 search_home、search_results 和 product_detail 检查每轮工具 schema 与完整 `SHOP_TOOL_SCHEMAS` 完全相同。保留守卫页面可用工具诊断测试。
- 创建新冻结ID `harness_eval_fixed_tools_20261110`；沿用相同 eval 1026题、dev 1026题、train 16489题、提示、模型底座及评测参数。旧版 `harness_eval_20261110` 保留，作为动态 schema 历史冻结。
- 冻结前发现SFT准入链路已有JEV服务切换：`jev_client.py` endpoint从OpenRouter改为LLMoxy SystemOne，模型名改为 `jev-latest`；API接入不再显式传choices，增加幂等请求头并在审计中记录endpoint。rubric版本和criteria文本未变。`collect_teacher_data.py` 改为优先读取 `LLMOXY_API_KEY`、回退 `JEVMODEL_API_KEY`。这些服务器现有改动经逐行对比旧快照并纳入新manifest，不保存凭据值。JEV模型别名是远端服务配置，实际返回模型信息需随采集审计记录；冻结本地客户端不能阻止服务端别名将来改变。
- 未重新运行模型效果评测或 SFT 训练。本次没有新的成功率结论。
- 原525/1026底座评测使用页面动态工具 schema；题集和其他参数可复用，但该结果不是当前固定工具 harness 的严格对照。SFT效果正式归因前，需要先用当前固定 schema 重跑底座 eval。

## 4. 实验验证结果

状态：固定工具实现、回归测试、新冻结快照和服务器/本地指纹核验均已完成。以冻结目录 `receipt.json`、`validation.json` 为校验依据。

```powershell
$env:PYTHONPATH=((Resolve-Path source_checkout/src).Path + ';' + (Resolve-Path source_checkout).Path)
python -m pytest -q source_checkout/tests/test_rollout.py source_checkout/tests/test_action_validation.py source_checkout/tests/test_shop_tools.py
```

结果：53 passed。测试直接捕获每轮传给 client 的 schema，断言不仅工具名称一致，完整 schema 内容和顺序也逐轮等于 `SHOP_TOOL_SCHEMAS`。

服务器使用以下命令检查运行时与SFT准入流程：

```bash
PYTHONPATH=src:. .venv/bin/python -m pytest -q \
  tests/test_rollout.py tests/test_action_validation.py tests/test_shop_tools.py \
  tests/test_sft_preparation_pipeline.py tests/test_protocol_admission.py \
  tests/test_frozen_harness.py
```

结果：81 passed、4 skipped、7 subtests passed；跳过项要求额外指定本地tokenizer，未将其记为通过。一次额外指定tokenizer的尝试在结束前连接中断，未取得完整结果，不作为通过证据。另有1项第三方pytest弃用警告。SFT集成测试已删除旧的动态工具列表断言，改为检查完整4轮请求的schema内容及顺序都等于 `SHOP_TOOL_SCHEMAS`。

冻结目录：`outputs/frozen/harness_eval_fixed_tools_20261110/`。包含源文件/配置/数据指纹、固定工具策略、当前采集harness指纹、快照及收据。`receipt.json` 保存最终manifest和snapshot的SHA256，`validation.json`保存只读校验结果。文档记录不纳入运行文件指纹锁定，以便后续追加实验结果；仍随快照保留。定版前的临时产物另存为provisional版本，历史动态工具冻结不覆盖。

本地同组测试结果为82 passed、4 skipped。本地已有DeepSeek准入分支，服务器是JEV准入分支；此前同步服务器采集脚本造成的本地接口不一致已恢复为本地的judge协议，并经上述测试验证。服务器checkout是本冻结的SFT采集权威版本，不能把两者文件混用。旧SFT教材（train5783、val108）缺当前协议与完整审计字段，必须重新采集、构建与导出。本次未采集教材、训练或运行正式模型评测。

最终冻结结果：99个锁定文件、7个固定工具；服务器 `check_frozen_harness.py` 返回 `passed`、0 problems。本地下载快照后核对manifest与snapshot的收据指纹、归档内全部99个锁定文件及38个采集identity文件，均通过。核验结果留存于本地 `experiment_results/harness_eval_fixed_tools_20261110/local_archive_validation.json`。

- manifest SHA256：`44ce34b2459f32e98468c247b116d4d7b403975c39d2778c49ba1aa6f8ff19ae`
- snapshot SHA256：`18ec0eaf958e0c7756ee1d7b559e77078a45edb91de9bb6a0e23cb062421d231`

## 5. 实验结果分析（是否达到目的）

达到恢复固定工具 schema 和建立可校验冻结版本的目的；页面级动作守卫仍提供执行约束。旧动态工具评测不能直接作SFT效果基线。模型效果尚待在此冻结版本下重新生成SFT数据、重跑底座基线并用相同题集评测，不能将回归测试通过等同为成功率提高。
