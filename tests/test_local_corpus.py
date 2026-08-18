from __future__ import annotations

import asyncio
import json

import httpx

from src.core.tool import ToolUseContext, build_default_registry
from src.tools.local_corpus import LocalCorpusReadTool, LocalCorpusSearchTool


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://retriever.test")


def test_local_corpus_tools_search_and_read(tmp_path):
    asyncio.run(_search_and_read(tmp_path))


async def _search_and_read(tmp_path):
    requests: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content) if request.content else {}
        requests.append((request.url.path, payload))
        if request.url.path == "/search":
            return httpx.Response(200, json={
                "results": [{
                    "doc_id": "wiki-42",
                    "title": "Answer page",
                    "contents": "A short answer excerpt.",
                    "url": "https://example.test/wiki-42",
                }]
            })
        assert request.url.path == "/document"
        assert payload == {"document_id": "wiki-42"}
        return httpx.Response(200, json={
            "id": "wiki-42",
            "title": "Answer page",
            "content": "The complete local document.",
            "url": "https://example.test/wiki-42",
        })

    search = LocalCorpusSearchTool(base_url="http://retriever.test")
    reader = LocalCorpusReadTool(base_url="http://retriever.test")
    search._client = _client(handler)
    reader._client = _client(handler)
    context = ToolUseContext(cache_dir=tmp_path)

    search_result = await search.call({"query": "answer", "top_k": 1}, context)
    read_result = await reader.call({"document_id": "wiki-42"}, context)

    assert requests == [
        ("/search", {"query": "answer", "top_k": 1}),
        ("/document", {"document_id": "wiki-42"}),
    ]
    assert "Document ID**: `wiki-42`" in search_result.data
    assert search_result.citations[0].url == "https://example.test/wiki-42"
    assert "The complete local document." in read_result.data
    assert read_result.citations[0].source_type.value == "local"


def test_local_retrieval_mode_replaces_external_tools():
    registry = build_default_registry({
        "local_retrieval": {"enabled": True, "base_url": "http://127.0.0.1:8080"},
    })

    assert [tool.name for tool in registry.all_tools()] == ["search_local", "read_local_document"]
    asyncio.run(registry.close_all())


def test_local_retrieval_mode_requires_http_base_url():
    try:
        build_default_registry({"local_retrieval": {"enabled": True, "base_url": "localhost:8080"}})
    except ValueError as exc:
        assert "base_url" in str(exc)
    else:
        raise AssertionError("expected invalid local retrieval configuration to fail")
