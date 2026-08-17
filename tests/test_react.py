from __future__ import annotations

import asyncio

from src.core.react import ReActContext, ReActEngine, ReActTurn
from src.core.tool import Tool, ToolRegistry, ToolUseContext
from src.core.types import EventType, LoopState, Message, StreamEvent, ToolResult


class FakeLLM:
    class config:
        max_tokens = 100

    async def stream(self, **_kwargs):
        yield StreamEvent(
            type=EventType.TOOL_USE,
            data={
                "tool_use_id": "call-1",
                "tool_name": "echo",
                "tool_input": {"value": "hello"},
            },
        )


class EchoTool(Tool):
    name = "echo"
    description = "Return its input."
    input_schema = {"type": "object"}
    is_concurrency_safe = True

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        return ToolResult(data=f"{context.extra['research_query']}:{args['value']}")


def test_react_reason_then_act_records_action_and_observation(tmp_path):
    asyncio.run(_run_react_turn(tmp_path))


async def _run_react_turn(tmp_path):
    registry = ToolRegistry()
    registry.register(EchoTool())
    engine = ReActEngine(registry, FakeLLM())
    state = LoopState(messages=[Message(role="user", content="original question")])
    turn = ReActTurn()

    events = [event async for event in engine.reason(state, "system", registry.get_api_schemas(), "s1", turn)]
    results = await engine.act(
        turn.tool_calls,
        state,
        ReActContext(session_id="s1", cache_dir=str(tmp_path)),
    )

    assert [event.type for event in events] == [EventType.TOOL_USE]
    assert turn.tool_calls[0]["tool_name"] == "echo"
    assert state.messages[-1].role == "assistant"
    assert state.messages[-1].content[0].tool_name == "echo"
    assert results[0].data == "original question:hello"
