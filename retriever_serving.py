from fastapi import FastAPI, HTTPException
import argparse
import os
import sys
from pydantic import BaseModel
from typing import List, Tuple, Union
import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

app = FastAPI()

retriever_list = []
available_retrievers = deque()
retriever_semaphore = None
retriever_lock = None

DEFAULT_CACHE_ROOT = Path(__file__).resolve().parent / "cache"


def _resolve_dir(path: str | None, default: Path) -> Path:
    if not path:
        return default
    return Path(path).expanduser().resolve()


def configure_cache(args) -> None:
    cache_root = _resolve_dir(args.cache_dir, DEFAULT_CACHE_ROOT)
    hf_home = _resolve_dir(args.hf_home, cache_root / "huggingface")
    datasets_cache = _resolve_dir(args.datasets_cache_dir, hf_home / "datasets")
    hub_cache = _resolve_dir(args.hf_hub_cache_dir, hf_home / "hub")
    tmp_dir = _resolve_dir(args.tmp_dir, cache_root / "tmp")

    for path in (hf_home, datasets_cache, hub_cache, tmp_dir):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["TMPDIR"] = str(tmp_dir)
    os.environ["TEMP"] = str(tmp_dir)
    os.environ["TMP"] = str(tmp_dir)

    if "datasets.config" in sys.modules:
        import datasets.config

        datasets.config.HF_DATASETS_CACHE = str(datasets_cache)

    print(f"[cache] HF_HOME={hf_home}")
    print(f"[cache] HF_DATASETS_CACHE={datasets_cache}")
    print(f"[cache] HF_HUB_CACHE={hub_cache}")
    print(f"[cache] TMPDIR={tmp_dir}")


def init_retriever(args):
    global retriever_semaphore
    configure_cache(args)

    from flashrag.config import Config
    from flashrag.utils import get_retriever

    config = Config(args.config)
    for i in range(args.num_retriever):
        print(f"Initializing retriever {i+1}/{args.num_retriever}")
        retriever = get_retriever(config)
        retriever_list.append(retriever)
        available_retrievers.append(i)
    # create a semaphore to limit the number of retrievers that can be used concurrently
    retriever_semaphore = asyncio.Semaphore(args.num_retriever)
    global retriever_lock
    retriever_lock = asyncio.Lock()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "retrievers": {
            "total": len(retriever_list),
            "available": len(available_retrievers)
        }
    }

class QueryRequest(BaseModel):
    query: str
    top_n: int = 10
    return_score: bool = False

class BatchQueryRequest(BaseModel):
    query: List[str]
    top_n: int = 10
    return_score: bool = False

class Document(BaseModel):
    id: str
    contents: str

@app.post("/search", response_model=Union[Tuple[List[Document], List[float]], List[Document]])
async def search(request: QueryRequest):
    query = request.query
    top_n = request.top_n
    return_score = request.return_score

    if not query or not query.strip():
        print(f"Query content cannot be empty: {query}")
        raise HTTPException(
            status_code=400,
            detail="Query content cannot be empty"
        )

    async with retriever_semaphore:
        async with retriever_lock:
            retriever_idx = available_retrievers.popleft()
        try:
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                if return_score:
                    results, scores = await loop.run_in_executor(
                        executor,
                        retriever_list[retriever_idx].search,
                        query, top_n, return_score
                    )
                    return [Document(id=result['id'], contents=result['contents']) for result in results], scores
                else:
                    results = await loop.run_in_executor(
                        executor,
                        retriever_list[retriever_idx].search,
                        query, top_n, return_score
                    )
                    return [Document(id=result['id'], contents=result['contents']) for result in results]
        finally:
            async with retriever_lock:
                available_retrievers.append(retriever_idx)

@app.post("/batch_search", response_model=Union[List[List[Document]], Tuple[List[List[Document]], List[List[float]]]])
async def batch_search(request: BatchQueryRequest):
    query = request.query
    top_n = request.top_n
    return_score = request.return_score

    async with retriever_semaphore:
        async with retriever_lock:
            retriever_idx = available_retrievers.popleft()
        try:
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                if return_score:
                    results, scores = await loop.run_in_executor(
                        executor,
                        retriever_list[retriever_idx].batch_search,
                        query, top_n, return_score
                    )
                    return [[Document(id=result['id'], contents=result['contents']) for result in results[i]] for i in range(len(results))], scores
                else:
                    results = await loop.run_in_executor(
                        executor,
                        retriever_list[retriever_idx].batch_search,
                        query, top_n, return_score
                    )
                    return [[Document(id=result['id'], contents=result['contents']) for result in results[i]] for i in range(len(results))]
        finally:
            async with retriever_lock:
                available_retrievers.append(retriever_idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./retriever_config.yaml")
    parser.add_argument("--num_retriever", type=int, default=1)
    parser.add_argument("--port", type=int, default=80)
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=str(DEFAULT_CACHE_ROOT),
        help="Root cache directory for Hugging Face caches and temporary files.",
    )
    parser.add_argument(
        "--hf_home",
        type=str,
        default="",
        help="HF_HOME directory. Defaults to <cache_dir>/huggingface.",
    )
    parser.add_argument(
        "--datasets_cache_dir",
        type=str,
        default="",
        help="HF datasets Arrow cache directory. Defaults to <hf_home>/datasets.",
    )
    parser.add_argument(
        "--hf_hub_cache_dir",
        type=str,
        default="",
        help="Hugging Face Hub cache directory. Defaults to <hf_home>/hub.",
    )
    parser.add_argument(
        "--tmp_dir",
        type=str,
        default="",
        help="Temporary directory for large local writes. Defaults to <cache_dir>/tmp.",
    )
    args = parser.parse_args()
    
    init_retriever(args)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)
