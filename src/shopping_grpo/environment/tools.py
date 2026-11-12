"""ShopSimulator 对模型公开的唯一工具定义。

这里同时保存两件必须保持同步的内容：模型看到的 JSON Schema，以及把工具调用
转换成环境 ``search[...]``/``click[...]`` 动作的映射。
"""

TOOL_VERSION = "shopping-tools-v5-no-model-stop"

CLICK_TOOL_ACTIONS = {
    "open_product": ("asin", None),
    "next_page": (None, "Next >"),
    "prev_page": (None, "< Prev"),
    "back_to_search": (None, "Back to Search"),
    "buy_now": (None, "Buy Now"),
}


def tool_call_to_action(name, parameters, observation=""):
    """把一个标准 tool call 转成 ShopSimulator 能执行的字符串动作。"""
    parameters = parameters or {}
    if name == "select_option":
        import json
        from shopping_grpo.environment.options import resolve_option
        choice = resolve_option(parameters["option_id"], observation)
        payload = {"axis": choice["axis"], "value": choice["value"]}
        return "click_option[" + json.dumps(payload, ensure_ascii=False) + "]"
    if name == "search_products":
        return f"search[{parameters['query']}]"
    key, fixed_value = CLICK_TOOL_ACTIONS[name]
    value = fixed_value if fixed_value is not None else parameters[key]
    return f"click[{value}]"


def _schema(name, description, properties=None, required=None):
    """生成统一格式的 OpenAI function schema，并禁止额外参数。

    无参数时省略 properties 键（additionalProperties=False 已表达无参数；
    空对象与 null 都无法写入 parquet 的 struct 列）。
    """
    parameters = {
        "type": "object",
        "required": required or [],
        "additionalProperties": False,
    }
    if properties:
        parameters["properties"] = properties
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


_INTERACTION_TOOL_SCHEMAS = [
    _schema(
        "search_products",
        "仅当最新 observation 显示“搜索功能是否可用: True”时搜索。query 应简洁，包含品类和最有区分度的品牌、型号、功能或规格；不得重复相同查询或机械复制整段需求。",
        {"query": {"type": "string"}},
        ["query"],
    ),
    _schema(
        "open_product",
        "打开最新 observation 当前搜索结果中列出的候选商品以核验价格、功能和规格；asin 必须原样取自该页面。",
        {"asin": {"type": "string"}},
        ["asin"],
    ),
    _schema(
        "select_option",
        "为当前候选补齐所有影响可购买 variant 的必要规格轴；option_id 必须取自最新 observation 的 option_choices（如 opt_1），不要复制规格文字；同一规格轴只选一个值，不得选择导航按钮或历史页面中的值。选择后应按完整 variant 的实际价格重新检查预算。",
        {"option_id": {"type": "string", "pattern": "^opt_[1-9][0-9]*$"}},
        ["option_id"],
    ),
    _schema("next_page", "仅当当前页面显示 Next > 且当前页没有合适候选时翻到下一页以发现新商品；无参数，必须传 {}。"),
    _schema("prev_page", "仅当当前页面显示 < Prev 按钮时返回上一页；无参数，必须传 {}。"),
    _schema("back_to_search", "仅当当前页面显示 Back to Search 按钮时返回搜索页；无参数，必须传 {}。"),
    _schema(
        "buy_now",
        "不可撤销的终止动作。仅当最新 observation 显示 Buy Now、品类正确、完整 variant 的实际价格在预算内，且按品牌、型号与核心功能、规格属性比较后是已核验候选中的最佳选择时购买；无参数，必须传 {}。",
    ),
]

SHOP_TOOL_SCHEMAS = list(_INTERACTION_TOOL_SCHEMAS)
