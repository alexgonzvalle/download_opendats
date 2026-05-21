from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Sequence, Union

JsonLike = Dict[str, Any]
ConfigLike = Union[JsonLike, str, os.PathLike]
UrlLike = Union[str, Sequence[str]]


def load_json_config(config: ConfigLike) -> JsonLike:
    if isinstance(config, dict):
        return dict(config)

    with open(os.fspath(config), "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("La configuracion JSON debe ser un objeto.")
    return data


def section(config: JsonLike, section_name: str) -> JsonLike:
    value = config.get(section_name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"La seccion '{section_name}' debe ser un objeto JSON.")
    return value


def first_value(config: JsonLike, section_data: JsonLike, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in section_data:
            return section_data[key]
        if key in config:
            return config[key]
    return default


def urls_from_config(
        config: JsonLike,
        section_data: JsonLike,
        url_keys: Sequence[str],
        template_keys: Sequence[str],
) -> UrlLike:
    url_value = first_value(config, section_data, *url_keys)
    if url_value:
        return url_value

    template = first_value(config, section_data, *template_keys)
    if not template:
        return []

    years = _as_int_list(first_value(config, section_data, "years", "year"), "years")
    months = _as_int_list(first_value(config, section_data, "months", "month"), "months")
    if not years:
        raise ValueError("Falta 'years' para construir URLs desde plantilla.")
    if not months:
        raise ValueError("Falta 'months' para construir URLs desde plantilla.")

    name_catalog = first_value(config, section_data, "name_catalog")
    return [
        _format_url_template(str(template), year, month, name_catalog)
        for year in years
        for month in months
    ]


def format_config_template(value: Any, config: JsonLike, section_data: JsonLike) -> Any:
    if not isinstance(value, str):
        return value

    name_catalog = first_value(config, section_data, "name_catalog")
    return value.format(name_catalog=name_catalog or "")


def normalize_catalog_urls(url_catalog: UrlLike) -> List[str]:
    urls = [url_catalog] if isinstance(url_catalog, str) else list(url_catalog)
    return [url[:-5] + "xml" if url.endswith("/catalog.html") else url for url in urls]


def _as_int_list(value: Any, name: str) -> List[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        return [int(value)]
    if isinstance(value, dict):
        start = value.get("start")
        end = value.get("end")
        if start is None or end is None:
            raise ValueError(f"'{name}' como rango necesita 'start' y 'end'.")
        return list(range(int(start), int(end) + 1))
    if isinstance(value, list):
        return [int(item) for item in value]
    raise ValueError(f"'{name}' debe ser entero, lista o rango.")


def _format_url_template(template: str, year: int, month: int, name_catalog: str | None = None) -> str:
    return template.format(
        name_catalog=name_catalog or "",
        year=year,
        month=month,
        month02=f"{month:02d}",
    )
