"""Tools for a locally served, fixed retrieval corpus.

The service contract is intentionally small and configurable so the same
ReAct baseline can use a FlashRAG index, a Wikipedia index, or a crawled-web
index. By default it expects JSON POST endpoints:

* ``/search``: ``{"query": "...", "top_k": 5}``
* ``/document``: ``{"document_id": "..."}``

Search responses may be a list or an object containing ``results``, ``data``,
``documents``, or ``hits``. Documents may use common field aliases such as
``id``/``doc_id`` and ``content``/``text``.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote, urljoin

import httpx

from src.core.tool import Tool, ToolUseContext
from src.core.types import Citation, SourceType, ToolResult, ValidationResult

logger = logging.getLogger(__name__)

_RESULT_KEYS = ("results", "data", "documents", "hits")
_ID_KEYS = ("document_id", "doc_id", "id", "_id")
_TEXT_KEYS = ("content", "text", "contents", "body", "snippet", "passage")
_TITLE_KEYS = ("title", "name", "heading")
_URL_KEYS = ("url", "source_url", "link")


class _LocalCorpusTool(Tool):
    """Shared HTTP client and response helpers for local-corpus tools."""

    def __init__(
        self,
        base_url: str,
        endpoint: str,
        method: str,
        timeout: int,
        max_result_size_chars: int,
    ):
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint
        self.method = method.upper()
        self.max_result_size_chars = max_result_size_chars
        self._client = httpx.AsyncClient(timeout=float(timeout))

    def _endpoint_url(self) -> str:
        if self.endpoint.startswith(("http://", "https://")):
            return self.endpoint
        return urljoin(f"{self.base_url}/", self.endpoint.lstrip("/"))

    async def _request(self, payload: dict[str, Any]) -> Any:
        try:
            if self.method == "GET":
                response = await self._client.get(self._endpoint_url(), params=payload)
            else:
                response = await self._client.post(self._endpoint_url(), json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            logger.warning("Local retrieval service returned HTTP %s", status)
            raise RuntimeError(f"Local retrieval service returned HTTP {status}") from exc
        except httpx.RequestError as exc:
            logger.warning("Local retrieval service request failed: %s", exc)
            raise RuntimeError(f"Cannot reach local retrieval service: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError("Local retrieval service returned invalid JSON") from exc

    @staticmethod
    def _value(item: dict[str, Any], keys: tuple[str, ...], default: str = "") -> str:
        for key in keys:
            value = item.get(key)
            if value is not None and str(value).strip():
                return str(value)
        return default

    @staticmethod
    def _records(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if not isinstance(payload, dict):
            return []
        for key in _RESULT_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                return [value]
        return [payload]

    @staticmethod
    def _document_url(document_id: str) -> str:
        return f"local://document/{quote(document_id, safe='')}"


class LocalCorpusSearchTool(_LocalCorpusTool):
    """Search a fixed corpus through a locally served retrieval endpoint."""

    name = "search_local"
    description = (
        "Search the configured local document corpus. Returns document IDs, titles, "
        "and excerpts. Use read_local_document with a relevant document_id to read it."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Specific query terms for the local corpus.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default: 5).",
                "default": 5,
            },
        },
        "required": ["query"],
    }
    is_concurrency_safe = True

    def __init__(
        self,
        base_url: str,
        search_path: str = "/search",
        search_method: str = "POST",
        query_field: str = "query",
        top_k_field: str = "top_k",
        default_top_k: int = 5,
        max_top_k: int = 20,
        timeout: int = 30,
        max_result_size_chars: int = 20000,
    ):
        super().__init__(base_url, search_path, search_method, timeout, max_result_size_chars)
        self.query_field = query_field
        self.top_k_field = top_k_field
        self.default_top_k = default_top_k
        self.max_top_k = max_top_k

    def prompt(self) -> str:
        return (
            "Use search_local to find documents in the fixed local corpus. "
            "Then use read_local_document with the returned document ID when more "
            "evidence is needed. Do not use external-web assumptions."
        )

    def validate_input(self, args: dict) -> ValidationResult:
        query = str(args.get("query", "")).strip()
        if len(query) < 2:
            return ValidationResult(valid=False, message="Query must be at least 2 characters")
        if len(query) > 500:
            return ValidationResult(valid=False, message="Query too long (max 500 chars)")
        return ValidationResult(valid=True)

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        query = str(args["query"]).strip()
        top_k = min(max(1, int(args.get("top_k", self.default_top_k))), self.max_top_k)
        try:
            payload = await self._request({self.query_field: query, self.top_k_field: top_k})
        except RuntimeError as exc:
            return ToolResult(data=str(exc), is_error=True)

        records = self._records(payload)
        if not records:
            return ToolResult(data=f"No local documents found for: {query}")

        parts = [f"## Local corpus results for: {query}\n"]
        citations: list[Citation] = []
        for index, record in enumerate(records[:top_k], 1):
            document_id = self._value(record, _ID_KEYS, default=str(index))
            title = self._value(record, _TITLE_KEYS, default=f"Document {document_id}")
            excerpt = self._value(record, _TEXT_KEYS, default="No excerpt returned.")
            url = self._value(record, _URL_KEYS, default=self._document_url(document_id))
            parts.append(
                f"### {index}. {title}\n"
                f"**Document ID**: `{document_id}`\n"
                f"**Source**: {url}\n"
                f"**Excerpt**: {excerpt}\n"
            )
            citations.append(Citation(url=url, title=title, snippet=excerpt, source_type=SourceType.LOCAL))

        data, truncated, cached_path = await self._maybe_truncate("\n".join(parts), query, context)
        return ToolResult(data=data, citations=citations, truncated=truncated, cached_path=cached_path)


class LocalCorpusReadTool(_LocalCorpusTool):
    """Read one document from a fixed local corpus by its stable identifier."""

    name = "read_local_document"
    description = "Read a complete document from the configured local corpus by document ID."
    input_schema = {
        "type": "object",
        "properties": {
            "document_id": {
                "type": "string",
                "description": "Document ID returned by search_local.",
            },
        },
        "required": ["document_id"],
    }
    is_concurrency_safe = True

    def __init__(
        self,
        base_url: str,
        document_path: str = "/document",
        document_method: str = "POST",
        document_id_field: str = "document_id",
        timeout: int = 30,
        max_result_size_chars: int = 50000,
    ):
        super().__init__(base_url, document_path, document_method, timeout, max_result_size_chars)
        self.document_id_field = document_id_field

    def prompt(self) -> str:
        return "Use read_local_document only for a document ID returned by search_local."

    def validate_input(self, args: dict) -> ValidationResult:
        document_id = str(args.get("document_id", "")).strip()
        if not document_id:
            return ValidationResult(valid=False, message="document_id is required")
        if len(document_id) > 1000:
            return ValidationResult(valid=False, message="document_id too long")
        return ValidationResult(valid=True)

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        requested_id = str(args["document_id"]).strip()
        try:
            payload = await self._request({self.document_id_field: requested_id})
        except RuntimeError as exc:
            return ToolResult(data=str(exc), is_error=True)

        records = self._records(payload)
        if not records:
            return ToolResult(data=f"Local document not found: {requested_id}")

        record = records[0]
        document_id = self._value(record, _ID_KEYS, default=requested_id)
        title = self._value(record, _TITLE_KEYS, default=f"Document {document_id}")
        content = self._value(record, _TEXT_KEYS, default="No document content returned.")
        url = self._value(record, _URL_KEYS, default=self._document_url(document_id))
        full_content = f"## {title}\n**Document ID**: `{document_id}`\n**Source**: {url}\n\n{content}"
        data, truncated, cached_path = await self._maybe_truncate(full_content, document_id, context)
        return ToolResult(
            data=data,
            citations=[Citation(url=url, title=title, snippet=content[:500], source_type=SourceType.LOCAL)],
            truncated=truncated,
            cached_path=cached_path,
        )
