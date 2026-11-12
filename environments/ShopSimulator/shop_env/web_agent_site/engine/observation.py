"""Structured, answer-free ShopSimulator state for the canonical Agent renderer."""

from __future__ import annotations

OBSERVATION_VERSION = "shopping-observation-v3"


def _compact_list(value, limit=10):
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        value = [value]
    return [str(item) for item in value[:limit] if str(item).strip()]


def product_summary(product: dict, *, rank: int | None = None, full_attributes=False) -> dict:
    pricing = product.get("Price")
    if pricing is None:
        pricing = product.get("pricing")
    result = {
        "asin": str(product.get("asin", "")),
        "title": str(product.get("title") or product.get("Title") or ""),
        "brand": str(product.get("brand") or product.get("shop_name") or ""),
        "category": str(product.get("category") or ""),
        "price": pricing,
        "key_attributes": _compact_list(
            product.get("attribute") or product.get("Attributes") or [],
            limit=None if full_attributes else 10,
        ),
    }
    if rank is not None:
        result["rank"] = int(rank)
    return result


def build_observation_state(
    *,
    page_type: str,
    session: dict,
    product_item_dict: dict,
    available_actions: dict,
) -> dict:
    """Build one public state; the task goal is intentionally not accepted."""
    clickables = [str(value) for value in available_actions.get("clickables", [])]
    state = {
        "observation_version": OBSERVATION_VERSION,
        "page_type": page_type,
        "search_available": bool(available_actions.get("has_search_bar")),
        "actions": clickables,
    }
    if page_type == "search_results":
        all_asins = session.get("search_result_asins") or []
        page_asins = session.get("current_page_asins") or []
        rank_by_asin = {
            asin: index for index, asin in enumerate(all_asins, start=1)
        }
        state.update(
            {
                "query": " ".join(session.get("keywords") or []),
                "normalized_query": session.get("normalized_query") or "",
                "page": int(session.get("page") or 1),
                "total_pages": int(session.get("total_pages") or 1),
                "total_results": int(session.get("total_results") or 0),
                "rank_start": min(
                    (rank_by_asin.get(asin, 0) for asin in page_asins),
                    default=0,
                ),
                "rank_end": max(
                    (rank_by_asin.get(asin, 0) for asin in page_asins),
                    default=0,
                ),
                "products": [
                    product_summary(
                        product_item_dict[asin],
                        rank=rank_by_asin[asin],
                    )
                    for asin in page_asins
                    if asin in product_item_dict and asin in rank_by_asin
                ],
            }
        )
    elif page_type == "product_detail":
        asin = session.get("asin")
        product = product_item_dict.get(asin, {})
        state["product"] = product_summary(product, full_attributes=True)
        state["selected_options"] = dict(session.get("options") or {})
        state["available_options"] = {
            str(key): _compact_list(values, limit=100)
            for key, values in (product.get("options") or {}).items()
        }
        if session.get("selected_price") is not None:
            state["selected_price"] = session["selected_price"]
    return state


def page_type_from_name(page_name: str) -> str:
    return {
        "": "search_home",
        "search_results": "search_results",
        "item_page": "product_detail",
        "done": "terminal",
    }.get(page_name, "unknown")
