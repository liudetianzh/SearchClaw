"""Minimal ReAct primitives shared by the higher-level agent harness.

The core loop is deliberately small:

    Reason: ask the model what to do next.
    Act: validate and execute the requested tools.
    Observe: callers append the returned tool results to the conversation.

Policies such as research plans, citation gates, context compaction, sessions,
and UI events belong to ``src.core.loop``. Keeping them out of this module
makes the ReAct mechanism independently understandable and reusable.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, AsyncGenerator

from src.core.tool import ToolRegistry, ToolUseContext
from src.core.types import ContentBlock, EventType, LoopState, Message, StreamEvent, ToolResult

if TYPE_CHECKING:
    from src.llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class ReActTurn:
    """The outcome of one model reasoning step."""

    tool_calls: list[dict] = field(default_factory=list)
    had_error: bool = False


@dataclass(frozen=True)
class ReActContext:
    """Runtime values needed to execute a tool action."""

    session_id: str = ""
    cache_dir: str = "./cache"
    rate_limiter: object | None = None
    allowed_roots: list[str] = field(default_factory=list)


class ReActEngine:
    """Run the model's Reason and Act phases without product policy."""

    def __init__(self, registry: ToolRegistry, llm_client: LLMClient):
        self.registry = registry
        self.llm_client = llm_client

    async def reason(
        self,
        state: LoopState,
        system_prompt: str,
        tools: list[dict] | None,
        session_id: str,
        turn: ReActTurn,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream one LLM response and append its assistant message to state."""
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        reasoning_blocks: list[dict] = []

        try:
            async for event in self.llm_client.stream(
                messages=[message.to_api_dict() for message in state.messages],
                system_prompt=system_prompt,
                tools=tools if tools else None,
                max_tokens=self.llm_client.config.max_tokens,
                session_id=session_id,
            ):
                yield event
                if event.type == EventType.TEXT_DELTA:
                    text_parts.append(event.data.get("text", ""))
                elif event.type == EventType.REASONING_DELTA:
                    reasoning_parts.append(event.data.get("text", ""))
                elif event.type == EventType.REASONING_BLOCKS:
                    reasoning_blocks = event.data.get("blocks", []) or []
                elif event.type == EventType.TOOL_USE:
                    turn.tool_calls.append(event.data)
                elif event.type == EventType.ERROR:
                    logger.error("LLM error: %s", event.data)
                    turn.had_error = True
                    break
        except Exception as exc:
            logger.error("Unexpected error in LLM stream: %s", exc)
            turn.had_error = True
            yield StreamEvent(type=EventType.ERROR, data={"message": f"LLM stream error: {exc}"})
            return

        if not turn.had_error:
            self._record_assistant_message(
                state, "".join(text_parts), "".join(reasoning_parts), reasoning_blocks, turn.tool_calls
            )

    async def act(
        self,
        tool_calls: list[dict],
        state: LoopState,
        context: ReActContext,
    ) -> list[ToolResult]:
        """Validate and execute actions, parallelizing tools that declare it safe."""
        tool_context = ToolUseContext(
            session_id=context.session_id,
            turn_count=state.turn_count,
            cache_dir=Path(context.cache_dir),
            extra={
                "loop_state": state,
                "research_query": self._research_query(state.messages),
                "allowed_roots": context.allowed_roots,
            },
            rate_limiter=context.rate_limiter,
        )
        concurrent_safe = self.registry.get_concurrent_safe()
        parallel_indices = [
            i for i, call in enumerate(tool_calls)
            if call["tool_name"] in concurrent_safe
        ]
        sequential_indices = [
            i for i, call in enumerate(tool_calls)
            if call["tool_name"] not in concurrent_safe
        ]
        results = [ToolResult(data="", is_error=True) for _ in tool_calls]

        if parallel_indices:
            parallel_results = await asyncio.gather(
                *(self._execute_one(tool_calls[i], tool_context) for i in parallel_indices),
                return_exceptions=True,
            )
            for index, result in zip(parallel_indices, parallel_results):
                if isinstance(result, Exception):
                    logger.error("Tool %s failed: %s", tool_calls[index]["tool_name"], result)
                    results[index] = ToolResult(data=f"Tool execution failed: {result}", is_error=True)
                else:
                    results[index] = result

        for index in sequential_indices:
            results[index] = await self._execute_one(tool_calls[index], tool_context)
        return results

    def _record_assistant_message(
        self,
        state: LoopState,
        text: str,
        reasoning: str,
        reasoning_blocks: list[dict],
        tool_calls: list[dict],
    ) -> None:
        if tool_calls:
            blocks = self._history_reasoning_blocks(reasoning, reasoning_blocks)
            if text:
                blocks.append(ContentBlock(type="text", text=text))
            blocks.extend(
                ContentBlock(
                    type="tool_use",
                    tool_use_id=call["tool_use_id"],
                    tool_name=call["tool_name"],
                    tool_input=call["tool_input"],
                )
                for call in tool_calls
            )
            state.messages.append(Message(role="assistant", content=blocks))
        elif text:
            # Reasoning is not needed after a final answer, so retain only user-visible text.
            state.messages.append(Message(role="assistant", content=text))

    @staticmethod
    def _history_reasoning_blocks(reasoning: str, structured_blocks: list[dict]) -> list[ContentBlock]:
        if structured_blocks:
            return [
                ContentBlock(
                    type="reasoning",
                    text=block.get("thinking", "") or "",
                    signature=block.get("signature", "") or None,
                )
                for block in structured_blocks
            ]
        return [ContentBlock(type="reasoning", text=reasoning)] if reasoning else []

    async def _execute_one(self, call: dict, context: ToolUseContext) -> ToolResult:
        tool_name = call["tool_name"]
        tool_input = call.get("tool_input", {})
        tool = self.registry.get(tool_name)
        if tool is None:
            return ToolResult(
                data=f"Error: Unknown tool '{tool_name}'. Available tools: "
                f"{', '.join(item.name for item in self.registry.all_tools())}",
                is_error=True,
            )

        validation = tool.validate_input(tool_input)
        if not validation.valid:
            logger.warning("Tool %s input validation failed: %s", tool_name, validation.message)
            return ToolResult(
                data=(
                    f"Invalid input for {tool_name}: {validation.message}. You sent: "
                    f"{json.dumps(tool_input)}. Please re-call with the correct required parameters."
                ),
                is_error=True,
            )

        for attempt in range(4):
            try:
                result = await tool.call(tool_input, context)
                logger.info("Tool %s completed: %d chars", tool_name, len(result.data))
                return result
            except Exception as exc:
                if "429" in str(exc) and attempt < 3:
                    await asyncio.sleep(2**attempt)
                    continue
                logger.error("Tool %s execution error: %s", tool_name, exc, exc_info=True)
                return ToolResult(
                    data=f"The {tool_name} service is temporarily unavailable. Please try again later.",
                    is_error=False,
                )

        raise AssertionError("unreachable")

    @staticmethod
    def _research_query(messages: list[Message]) -> str:
        for message in messages:
            if message.role == "user" and not message.metadata.get("_tag"):
                return message.text_content
        return ""
