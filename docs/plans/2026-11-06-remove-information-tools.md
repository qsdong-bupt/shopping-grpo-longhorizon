# 移除四个信息页动作：设计与实现记录

日期：2026-11-06。用户授权：移除四个动作及 observation 中的对应按钮，记录改动并提交到 GitHub `V2.0`。

## 目标与证据

移除 `view_description`、`view_reviews`、`view_attributes`、`view_features`，模型只能通过搜索结果页、商品详情页完成核验与购买。

对当前 23,421 条源商品的检查结果：

- `full_description` 全部等于商品标题，没有独立描述证据。
- loader 的评论来自空的 `all_reviews`，运行时 `Reviews` 为空。
- `small_description` 全部缺失，不能提供有效特点信息。
- 263 件商品的属性超过 10 条，其中 19 件来自源 `tag=eval`。旧详情摘要只显示前 10 条；因此删除属性页必须同时把详情改为完整属性。

此结论只适用于当前冻结商品数据。未来若引入独立描述、真实评论或新特点，应重新设计公开证据字段与版本，不能依赖被删除的页面。

## 交互与观测变化

工具从 12 个收敛为 8 个：`search_products`、`open_product`、`select_option`、`next_page`、`prev_page`、`back_to_search`、`buy_now`、`finish_without_purchase`。

1. Python 唯一工具定义及 `configs/tools.json` 同步删除四个 schema 和动作映射，清理未公开的旧 `think` 配置。
2. Student / Teacher 提示词不再教授信息子页。Teacher 明确商品详情直接提供完整属性。
3. 删除四个 HTML 子页模板、Flask 子页路由、SimServer 导航分支，以及相应状态与访问记录；详情 HTML 不再生成四个按钮。
4. 搜索摘要仍展示前 10 条属性，商品详情、选择规格后的详情均展示全部属性。商品标题、品牌、品类、价格、规格轴与选择状态继续保留。
5. 动作守卫拒绝旧工具名；旧按钮名仍仅作为内部保留导航词，防止被伪装为 `select_option` 参数，不进入模型工具 schema。
6. 结构化渲染器拒绝包含旧按钮的状态；投影器拒绝旧版 observation 或旧按钮 footer。结构化详情超出 token 预算时显式报错，要求调整详情预算，不静默截断购买证据。

## Reward、终止与评测

环境 Reward v3 购买比较逻辑不变：购买时根据 session ASIN 从后端商品字典读取完整记录，使用实际已选规格和价格判分，不要求先访问信息子页。

正式严格成功仍要求完整 `gold_purchase` 终局且 `reward_valid=true`；替代购买不会变成正式严格成功。SFT 的 Gold / 冻结 JEV 替代购买准入规则保持原样，额外增加交互协议校验。

配套变化：

- 删除信息页访问次数指标及其消费者；确定性指标版本升级为 `shopping-deterministic-metrics-v2`。
- 删除信息页的进度信用，不再通过首次访问子页刷新无进展计数；保留 35 步上限、连续无进展 4 步、精确动作重复 2 次的边界。终止追踪版本升级为 v4。
- GRPO 适配层原有的“访问信息页 +1 分”被移除。探索充分度现在为 0–2：打开至少两个不同候选 +1，至少两组不同搜索参数 +1；充分阈值仍为 2。因此必须同时满足两项才能进入尽力买错档。`-0.45 / -1.00` 档位数值保持原样。这是训练塑形条件变化，不是环境 Reward v3 的商品判分变化。
- 两个轨迹分析脚本的探索工具集合同步清理。

## 协议冻结与数据处理

| 层 | 当前版本 |
|---|---|
| Environment | `shopsimulator-environment-v2.2` |
| Observation | `shopping-observation-v3` |
| Tool schema | `shopping-tools-v3` |
| Environment Reward | `shopsimulator-reward-v3` |
| Termination | `shopping-termination-v4` |

新增 `environment/protocol.py`，集中记录版本与工具 schema SHA-256：

- 新采集轨迹保存 `runtime_protocol`；续跑拒绝缺少标识或协议不一致的旧轨迹。
- Teacher 正式采集的任务状态也保存并校验协议，防止旧的 accepted 状态让新采集跳过任务。重新采集前须将旧 collection 输出移出当前活动目录；本次不会搬移或删除用户数据。
- SFT 教材构建、Parquet 导出、训练数据集加载均检查协议；旧动作即使 loss mask 为 0，也不能继续作为上下文混入。合法工具被守卫拒绝的段仍按现有规则屏蔽训练。无可用教材时给出明确错误。
- GRPO 构建使用当前 Student 提示词并写入协议标识；启动前校验 train / val Parquet 的协议和 system 提示词，旧数据必须重新构建。
- 评测保存 `run_identity.json`，包含协议、推理配置、任务及其提示词；不允许更换协议或配置后续写同一评测目录。汇总版本升级为 v2 并保存协议。
- `data/environment.json` 更新版本和运行文件 SHA-256，预检进一步覆盖工具定义、渲染器、投影器、商品模板、终止器、引擎和环境配置；工具 JSON 必须与 Python 定义完全一致。
- 更新当前 README、AGENTS、启动入口与 agent 配置。上游源码来源 commit 继续作为来源元数据保留。历史设计文档不追改。
- `.gitattributes` 为参与运行文件哈希的文本固定 LF，避免 Windows 自动转换行尾导致跨平台哈希不一致。

旧示范需按新动作空间重新采集；旧 SFT/GRPO 数据需要重新构建。已有检查点权重不会自动获得新协议能力，之后若评估它们，应作为新动作空间下的迁移评估。跨协议成功率、步数和探索分数不能直接归因于训练收益。当前没有重新采集、训练、合并模型或运行正式评测。

## 验证

- Python `compileall` 通过。
- 主项目及环境 CPU 测试：193 passed、9 skipped、1 deselected，另有 10 个 subtests 通过；已在用于提交的干净 checkout 上复验。
- 新增测试覆盖工具配置一致性、旧轨迹/SFT 准入拒绝、GRPO 存储提示词拒绝、评测身份检查、完整属性与 token 预算保护、运行文件哈希和探索充分度。
- 6 个真实环境集成测试通过：使用小型固定商品与受控检索结果，实际运行 HTML 模板、SimBrowser、WebAgentTextEnv、选择规格和 Reward v3 终局；验证四个按钮消失、直接旧 click 无效果、15 条属性完整可见，并完成 Gold 购买。它们不调用模型或外部 Judge，也不属于正式评测。
- 完整可运行 CPU 集合中仅 `test_product_sha_mismatch_fails_closed` 失败：Windows 清理临时 SQLite 文件时发生 WinError 32。已在远程原始提交 `e7886e9` 的干净 checkout 复现相同错误，本次未修改搜索模块。上述通过集合显式排除了这一项。
- `test_verl_adapter.py`、`test_sft_trainer_metrics.py` 依赖本机缺少的 veRL，未执行；原有 tokenizer/构建相关条件跳过项保留。没有安装或启动 GPU 训练栈。Flask、Gym、Hydra 等测试依赖仅放在系统临时测试目录。

## 主要代码位置

- 环境工具与观测：`src/shopping_grpo/environment/{tools,actions,observation,projection,protocol,manifest}.py`。
- 环境实现：`environments/ShopSimulator/shop_env/web_agent_site/{app.py,envs/web_agent_text_env.py,engine/*,templates/*}`。
- 轨迹与指标：`src/shopping_grpo/evaluation/{rollout,metrics}.py`、`scripts/{evaluate_model,compare_trajectories}.py`。
- 教材与训练适配：`scripts/{build_sft_dataset,export_sft_parquet,build_grpo_data,check_grpo_runtime}.py`，SFT 数据集与 GRPO runtime/tools 适配器。
- 新回归测试：`tests/test_protocol_admission.py`、`environments/ShopSimulator/shop_env/tests/test_simplified_tools.py`；同步改写已有旧子页断言。
