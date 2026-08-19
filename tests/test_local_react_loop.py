from __future__ import annotations

import asyncio
import json

import httpx

from src.core.loop import QueryParams, query_loop
from src.core.tool import build_default_registry
from src.core.types import EventType, StreamEvent


class SearchThenAnswerLLM:
    """A deterministic model used to verify the ReAct search-observe loop."""

    class config:
        max_tokens = 100

    uses_responses_api = False

    def __init__(self):
        self.calls: list[list[dict]] = []

    def reset_response_chain(self, session_id: str = "") -> None:
        pass

    async def stream(self, *, messages: list[dict], **_kwargs):
        self.calls.append(messages)
        if len(self.calls) == 1:
            yield StreamEvent(
                type=EventType.TOOL_USE,
                data={
                    "tool_use_id": "search-1",
                    "tool_name": "search_local",
                    "tool_input": {"query": "capital of France", "top_k": 1},
                },
            )
            return

        observation = messages[-1]
        assert observation["role"] == "tool"
        assert observation["tool_call_id"] == "search-1"
        assert "Paris is the capital of France." in observation["content"]
        yield StreamEvent(type=EventType.TEXT_DELTA, data={"text": "Paris."})


def test_react_search_observe_loop_with_local_corpus(tmp_path):
    asyncio.run(_run_local_react_loop(tmp_path))


async def _run_local_react_loop(tmp_path):
    def retrieval_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/search"
        assert json.loads(request.content) == {"query": "capital of France", "top_k": 1}
        return httpx.Response(200, json={
            "results": [{
                "id": "wiki-paris",
                "title": "Paris",
                "content": "Paris is the capital of France.",
            }]
        })

    registry = build_default_registry({
        "local_retrieval": {"enabled": True, "base_url": "http://retriever.test"},
    })
    search_tool = registry.get("search_local")
    assert search_tool is not None
    search_tool._client = httpx.AsyncClient(
        transport=httpx.MockTransport(retrieval_handler),
        base_url="http://retriever.test",
    )

    llm = SearchThenAnswerLLM()
    events = [
        event async for event in query_loop(QueryParams(
            query="What is the capital of France?",
            system_prompt="Use the available tools.",
            tool_registry=registry,
            llm_client=llm,
            cache_dir=str(tmp_path),
            max_turns=3,
            max_search=2,
            max_fetch=2,
            stream_full_tool_results=True,
        ))
    ]

    assert [tool.name for tool in registry.all_tools()] == ["search_local", "read_local_document"]
    assert [event.type for event in events].count(EventType.TOOL_USE) == 1
    result_events = [event for event in events if event.type == EventType.TOOL_RESULT]
    assert len(result_events) == 1
    assert "Paris is the capital of France." in result_events[0].data["result"]
    assert len(llm.calls) == 2
    assert any(event.type == EventType.TEXT_DELTA and event.data["text"] == "Paris." for event in events)
    await registry.close_all()
