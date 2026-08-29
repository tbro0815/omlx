# SPDX-License-Identifier: Apache-2.0
"""Helpers for converting parser-emitted tool calls to OpenAI models."""

import json
import logging
import uuid

from pydantic import ValidationError

from .openai_models import FunctionCall, ToolCall

logger = logging.getLogger(__name__)


def convert_parser_tool_calls(tool_calls: list[dict] | None) -> list[ToolCall]:
    """Convert parser-emitted tool-call dicts into validated OpenAI ToolCalls.

    Parser output comes from model text and can contain malformed JSON
    arguments. Treat those as recoverable parser failures rather than letting
    Pydantic validation abort the response stream.
    """
    converted: list[ToolCall] = []
    for tool_call in tool_calls or []:
        if not isinstance(tool_call, dict):
            continue
        # Two producer shapes reach this converter. Local tool parsers emit
        # the flat parser shape ({"name": ..., "arguments": ...}); the remote
        # engine (RemoteOpenAIEngine._map_tool_calls) emits full OpenAI shape
        # with both fields nested under "function". Reading only the flat
        # shape silently gutted every remote tool call to name="" and
        # arguments="{}" -- the id survived because that field is top-level
        # in both shapes, which made the loss easy to miss.
        function = tool_call.get("function")
        if isinstance(function, dict):
            name = function.get("name", "")
            arguments = function.get("arguments", "{}") or "{}"
        else:
            name = tool_call.get("name", "")
            arguments = tool_call.get("arguments", "{}") or "{}"
        if not isinstance(arguments, str):
            # Some providers return arguments as a decoded object.
            arguments = json.dumps(arguments)
        try:
            converted.append(
                ToolCall(
                    id=tool_call.get("id")
                    or tool_call.get("call_id")
                    or f"call_{uuid.uuid4().hex[:8]}",
                    type="function",
                    function=FunctionCall(
                        name=name,
                        arguments=arguments,
                    ),
                )
            )
        except (TypeError, ValueError, ValidationError) as e:
            snippet = str(arguments)
            if len(snippet) > 120:
                snippet = snippet[:117] + "..."
            logger.warning(
                "Dropping malformed parser tool call %r: %s. arguments=%r",
                name,
                e,
                snippet,
            )
            continue
    return converted
