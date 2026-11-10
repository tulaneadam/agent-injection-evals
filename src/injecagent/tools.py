"""Convert InjecAgent's tool specifications into Inspect ToolInfo objects.

Mirrors `transform_tool_format_gpt` in the upstream `src/utils.py` so the model
sees the same function schemas as in the paper's function-calling setting.
These tools are never executed: every tool response in InjecAgent is fixed
(the user tool) or simulated from a cache (the attacker tool in step 2).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict

from inspect_ai.tool import ToolInfo, ToolParam, ToolParams
from inspect_ai.util import JSONSchema

from .data import load_json

_ALLOWED_TYPES = {"string", "integer", "number", "boolean", "array", "object", "null"}


def _array_item_description(description: str) -> str:
    for marker in ("array of", "list of"):
        if marker in description:
            return description[description.index(marker) + len(marker):].strip()
    return description


def to_tool_info(spec: Dict[str, Any]) -> ToolInfo:
    properties: Dict[str, ToolParam] = {}
    required = []
    for param in spec.get("parameters", []):
        name = param["name"]
        ptype = param.get("type", "string")
        description = param.get("description", "")
        if ptype == "array":
            # Same (slightly odd) item schema as upstream: an object wrapping one string field.
            properties[name] = ToolParam(
                type="array",
                items=JSONSchema(
                    type="object",
                    properties={name: JSONSchema(type="string", description=_array_item_description(description))},
                ),
            )
        else:
            properties[name] = ToolParam(
                type=ptype if ptype in _ALLOWED_TYPES else "string",
                description=description,
            )
        if param.get("required"):
            required.append(name)
    return ToolInfo(
        name=spec["name"],
        description=spec.get("summary", ""),
        parameters=ToolParams(type="object", properties=properties, required=required),
    )


@lru_cache(maxsize=1)
def tool_specs() -> Dict[str, Dict[str, Any]]:
    """All tool specs keyed by full name (toolkit + tool, e.g. 'GmailSendEmail')."""
    specs: Dict[str, Dict[str, Any]] = {}
    for toolkit in load_json("tools.json"):
        for tool in toolkit["tools"]:
            spec = dict(tool)
            spec["name"] = toolkit["toolkit"] + tool["name"]
            specs[spec["name"]] = spec
    return specs


def tool_info(name: str) -> ToolInfo:
    return to_tool_info(tool_specs()[name])
