import copy
import importlib.util
import io
import json
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "target_price_filter", Path(__file__).resolve().parents[1] / "scripts/filter_unresolvable_price_tasks.py"
)
filter_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(filter_module)


def product():
    return {
        "asin": "sample",
        "pricing": [10, 30],
        "instructions": [{"instruction": "买粉色M号", "instruction_options": ["粉色", "M"]}],
        "customization_options": {
            "颜色": [{"value": "粉色", "price": 10}, {"value": "黑色", "price": 20}],
            "尺码": [{"value": "M", "price": 15}, {"value": "L", "price": 30}],
        },
    }


def test_multiple_price_axes_excluded_even_without_budget():
    problem = filter_module.inspect_target(product())
    assert problem["reason"] == "target_price_unverifiable"
    assert problem["price_resolution"]["method"] == "multiple_effective_price_axes"
    assert problem["selected_options"] == {"颜色": "粉色", "尺码": "M"}


def test_complete_combination_kept():
    item = product()
    item["variant_combinations"] = [{"options": {"颜色": "粉色", "尺码": "M"}, "price": 17}]
    assert filter_module.inspect_target(item) is None


def test_two_axes_with_only_one_price_axis_kept():
    item = product()
    for entry in item["customization_options"]["尺码"]:
        entry["price"] = 10
    assert filter_module.inspect_target(item) is None


def test_missing_required_value_excluded_even_if_base_price_known():
    item = product()
    item["instructions"][0]["instruction_options"] = ["不存在"]
    assert filter_module.inspect_target(item)["reason"] == "target_options_unresolved"


def test_plain_single_price_kept_and_input_not_mutated():
    item = {"pricing": [10], "instructions": [{"instruction_options": []}]}
    original = copy.deepcopy(item)
    assert filter_module.inspect_target(item) is None
    assert item == original


def test_price_range_without_resolvable_variant_excluded():
    assert filter_module.inspect_target({"pricing": [10, 20], "instructions": [{}]})["price_resolution"]["method"] == "price_not_deterministic"


def test_missing_target_instruction_excluded():
    assert filter_module.inspect_target({"pricing": [10]})["reason"] == "target_instruction_missing"


def test_stream_array_handles_unicode_and_chunk_boundaries():
    class SmallChunks(io.StringIO):
        def read(self, size=-1):
            return super().read(min(size, 13))

    records = [product(), {"asin": "second", "text": "包含引号\"和括号[]"}]
    assert list(filter_module.iter_json_array(SmallChunks(json.dumps(records, ensure_ascii=False)))) == records
