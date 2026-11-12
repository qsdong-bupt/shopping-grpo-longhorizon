import unittest

from shopping_grpo.environment.actions import (
    action_reject_reason,
    product_ids,
)


class ActionValidationTest(unittest.TestCase):
    def test_select_option_rejects_navigation_button(self):
        """规格工具不能把页面导航按钮伪装成一个规格值。"""
        observation = ('page_type: product_detail\n'
                       'option_choices: [{"option_id":"opt_1","axis":"颜色","value":"糖果粉"}]\n'
                       '可点击的按钮: ["< Prev", "opt_1"]')

        reason = action_reject_reason("select_option", {"option_id": "< Prev"}, observation)

        self.assertEqual(reason, "invalid_option_id")

    def test_select_option_allows_current_product_option(self):
        observation = ('page_type: product_detail\n'
                       'option_choices: [{"option_id":"opt_1","axis":"颜色","value":"糖果粉"}]\n'
                       '可点击的按钮: ["< Prev", "opt_1"]')

        self.assertIsNone(action_reject_reason("select_option", {"option_id": "opt_1"}, observation))

    def test_post_selection_allows_current_page_navigation(self):
        """规格选择不改变环境的可用动作；只校验当前页面是否真的有按钮。"""
        observation = '商品页\n\n可点击的按钮: ["back to search", "Buy Now"]'

        reason = action_reject_reason("back_to_search", {}, observation)

        self.assertIsNone(reason)

    def test_rejects_schema_extra_argument_before_executing_tool(self):
        """无参数工具携带垃圾字段时，不能静默丢弃字段后继续执行。"""
        observation = '商品页\n\n可点击的按钮: ["Buy Now"]'

        reason = action_reject_reason("buy_now", {"string": "true"}, observation)

        self.assertEqual(reason, "schema_extra_arguments:string")

    def test_rejects_missing_required_argument_before_executing_tool(self):
        observation = "搜索功能是否可用: True"

        reason = action_reject_reason("search_products", {}, observation)

        self.assertEqual(reason, "schema_missing_arguments:query")

    def test_structured_observation_accepts_real_eleven_digit_product_id(self):
        asin = "35842622441"
        observation = (
            "[SHOPPING_OBSERVATION_V3]\n"
            "1|35842622441|158.0|泰国乳胶枕\n"
            '可点击的按钮: ["back to search", "35842622441"]'
        )
        self.assertEqual(product_ids(observation), [asin])
        self.assertIsNone(
            action_reject_reason("open_product", {"asin": asin}, observation)
        )

    def test_open_product_requires_asin_in_current_clickable_buttons(self):
        asin = "35842622441"
        observation = (
            "[SHOPPING_OBSERVATION_V3]\n"
            f"1|{asin}|158.0|泰国乳胶枕\n"
            '可点击的按钮: ["back to search"]'
        )
        self.assertEqual(
            action_reject_reason("open_product", {"asin": asin}, observation),
            "click_not_in_previous_observation",
        )

    def test_unrelated_long_number_is_not_treated_as_product(self):
        observation = (
            "商品描述包含电话号码 13800138000 和价格 12345678"
            '\n\n可点击的按钮: ["back to search"]'
        )
        self.assertEqual(product_ids(observation), [])

    def test_retired_tools_are_rejected(self):
        for name in ("view_description", "view_features", "view_reviews", "view_attributes"):
            self.assertEqual(action_reject_reason(name, {}, ""), "unknown_or_invalid_tool")


class NavigationFeedbackTest(unittest.TestCase):
    def feedback(self, page, name, buttons, search=False, reason="click_not_in_previous_observation"):
        import json
        from shopping_grpo.environment.actions import action_guard_tool_message
        observation = (
            f"[SHOPPING_OBSERVATION_V3]\npage_type: {page}\n"
            f"搜索功能是否可用: {search}\n可点击的按钮: "
            + json.dumps(buttons, ensure_ascii=False)
        )
        call = {"id": "test-call", "function": {"name": name, "arguments": "{}"}}
        message = action_guard_tool_message(call, reason, observation)
        self.assertTrue(message["runtime_action_guard"])
        self.assertEqual(message["tool_call_id"], "test-call")
        return message["content"]

    def test_detail_explains_available_return_routes(self):
        text = self.feedback("product_detail", "next_page", ["< prev", "back to search", "buy now"])
        self.assertIn("商品详情页", text)
        self.assertIn("请求工具：next_page", text)
        self.assertIn("商品详情页不能翻页", text)
        self.assertIn("可用工具：buy_now、prev_page、back_to_search", text)
        self.assertNotIn("下一步必须调用", text)

    def test_does_not_offer_missing_navigation(self):
        text = self.feedback("product_detail", "open_product", ["buy now"])
        self.assertNotIn("prev_page({})", text)
        self.assertNotIn("back_to_search({})", text)

    def test_home_explains_search_after_return(self):
        text = self.feedback("search_home", "next_page", [], search=True)
        self.assertIn("当前页面：search_home", text)
        self.assertIn("请求工具：next_page", text)
        self.assertIn("可用工具：search_products", text)
        self.assertNotIn("prev_page({})", text)

    def test_home_does_not_offer_unavailable_search(self):
        text = self.feedback("search_home", "select_option", [], search=False)
        self.assertNotIn('search_products({"query":', text)

    def test_result_does_not_offer_missing_next_button(self):
        text = self.feedback("search_results", "next_page", ["back to search"])
        self.assertIn("原因：click_not_in_previous_observation", text)
        self.assertIn("可用工具：back_to_search", text)
        self.assertNotIn("next_page", text.split("可用工具：", 1)[1])

    def test_schema_errors_do_not_get_misleading_navigation_diagnosis(self):
        text = self.feedback("product_detail", "next_page", ["< prev"], reason="schema_extra_arguments:x")
        self.assertIn("schema_extra_arguments:x", text)
        self.assertNotIn("不能直接打开其他商品", text)

    def test_detail_search_rejection_has_available_routes(self):
        text = self.feedback("product_detail", "search_products", ["< prev"], reason="search_not_available_on_current_page")
        self.assertIn("原因：商品详情页不能搜索", text)
        self.assertIn("可用工具：prev_page", text)

    def test_dynamic_tools_are_limited_by_page_and_buttons(self):
        from shopping_grpo.environment.actions import available_tools_for_observation

        self.assertEqual(
            available_tools_for_observation(
                '[SHOPPING_OBSERVATION_V3]\npage_type: search_home\n'
                '搜索功能是否可用: True\n可点击的按钮: []'
            ),
            ["search_products"],
        )
        self.assertEqual(
            available_tools_for_observation(
                '[SHOPPING_OBSERVATION_V3]\npage_type: search_results\n'
                '1|12345678901|9|店|类|属性|商品\n'
                '可点击的按钮: ["12345678901", "Next >", "Back to Search"]'
            ),
            ["open_product", "next_page", "back_to_search"],
        )
        self.assertNotIn(
            "open_product",
            available_tools_for_observation(
                '[SHOPPING_OBSERVATION_V3]\npage_type: search_results\n'
                '1|12345678901|9|店|类|属性|商品\n'
                '可点击的按钮: ["Next >", "Back to Search"]'
            ),
        )
        self.assertEqual(
            available_tools_for_observation(
                '[SHOPPING_OBSERVATION_V3]\npage_type: product_detail\n'
                'option_choices: [{"option_id":"opt_1","axis":"颜色","value":"红"}]\n'
                '可点击的按钮: ["opt_1", "Buy Now", "< Prev"]'
            ),
            ["select_option", "buy_now", "prev_page"],
        )

    def test_unknown_page_keeps_tools_available_for_safe_fallback(self):
        from shopping_grpo.environment.actions import available_tools_for_observation

        self.assertEqual(available_tools_for_observation("可点击的按钮: [\"Buy Now\"]"), [])


if __name__ == "__main__":
    unittest.main()
