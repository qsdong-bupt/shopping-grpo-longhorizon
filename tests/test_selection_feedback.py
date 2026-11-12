import unittest

from shopping_grpo.evaluation.rollout import _selection_feedback, _tool_message


def state(selected):
    return {
        "page_type": "product_detail",
        "product": {"asin": "123456789012", "title": "羽毛球包", "brand": "示例店铺",
                    "category": "运动›羽毛球包", "price": "50.0 to 60.0",
                    "key_attributes": ["单肩包", "仿水"]},
        "selected_options": selected,
        "available_options": {"尺码": ["S", "M"], "颜色": ["白", "黑"]},
    }


def step(selected):
    return {"tool_name": "select_option", "observation": "original observation",
            "result": {"observation_state": state(selected)}}


class SelectionFeedbackTest(unittest.TestCase):
    def test_summary_contains_current_public_product_and_missing_axis(self):
        text = _selection_feedback({}, step({"尺码": "S"}))
        self.assertEqual(text, "[选择后的商品与规格摘要]\n\n商品标题：羽毛球包\n"
                         "品牌／店铺：示例店铺\n商品类别：运动›羽毛球包\n"
                         "商品属性：单肩包、仿水\n当前页面价格：50.0 to 60.0 元\n"
                         "当前已选规格：\n- 尺码：S\n尚未选择的规格轴：颜色")

    def test_selected_price_preferred_including_zero(self):
        s = step({})
        s["result"]["observation_state"]["selected_price"] = 0
        self.assertIn("当前页面价格：0 元", _selection_feedback({}, s))

    def test_actual_selection_preserves_both_axes_and_whitespace(self):
        text = _selection_feedback({}, step({"尺码": "S", "颜色": "粉色  【套装】"}))
        self.assertIn("- 尺码：S\n- 颜色：粉色  【套装】", text)
        self.assertIn("尚未选择的规格轴：无", text)

    def test_unconfirmed_request_does_not_become_selected(self):
        s = step({})
        s["resolved_option"] = {"axis": "颜色", "value": "白"}
        text = _selection_feedback({}, s)
        self.assertIn("当前已选规格：\n- 无", text)
        self.assertIn("尚未选择的规格轴：尺码、颜色", text)
        self.assertNotIn("- 颜色：白", text)

    def test_repeated_selection_has_same_summary(self):
        s = step({"尺码": "S"})
        self.assertEqual(_selection_feedback(state({}), s),
                         _selection_feedback(state({"尺码": "S"}), s))

    def test_missing_product_information_is_not_invented(self):
        s = step({})
        s["result"]["observation_state"]["product"] = {}
        text = _selection_feedback({}, s)
        self.assertIn("商品标题：未提供", text)
        self.assertIn("当前页面价格：未提供", text)

    def test_other_tools_and_pages_unchanged(self):
        s = step({})
        s["tool_name"] = "open_product"
        self.assertEqual(_selection_feedback({}, s), "")
        self.assertEqual(_tool_message({"id": "x"}, s)["content"], "original observation")
        s["tool_name"] = "select_option"
        s["result"]["observation_state"]["page_type"] = "search_results"
        self.assertEqual(_selection_feedback({}, s), "")

    def test_summary_is_delivered_without_mutating_original_observation(self):
        s = step({"颜色": "白"})
        s["selection_feedback"] = _selection_feedback({}, s)
        msg = _tool_message({"id": "x"}, s)
        self.assertEqual(msg["content"], "original observation\n\n" + s["selection_feedback"])
        self.assertEqual(s["observation"], "original observation")
