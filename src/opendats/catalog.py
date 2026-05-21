from __future__ import annotations

from typing import Any, Dict, List


def extract_datasets(parsed_xml: Dict[str, Any]) -> List[Dict[str, Any]]:
    root = _safe_get(parsed_xml, "catalog", "dataset") or parsed_xml.get("dataset")
    datasets: List[Dict[str, Any]] = []

    def walk(node: Any) -> None:
        if node is None:
            return
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if isinstance(node, dict):
            if any(key.startswith("@") for key in node) or "dataset" in node:
                datasets.append(node)
            walk(node.get("dataset"))

    walk(root)
    return _dedupe_datasets(datasets)


def _safe_get(data: Dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _dedupe_datasets(datasets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique: List[Dict[str, Any]] = []
    for dataset in datasets:
        key = dataset.get("@ID") or dataset.get("@urlPath") or dataset.get("@name") or str(id(dataset))
        if key in seen:
            continue
        seen.add(key)
        unique.append(dataset)
    return unique
