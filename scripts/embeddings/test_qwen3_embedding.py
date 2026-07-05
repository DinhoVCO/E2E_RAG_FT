"""Test Qwen3-Embedding via vLLM (online API and offline inference).

Mirrors notebooks/embeddings/test_qwen3_embedding.ipynb.

Usage:
    # Start the vLLM server first (from repo root):
    bash jobs/scripts/vllm/serve_embedding_4b.sh

    python scripts/embeddings/test_qwen3_embedding.py online
    python scripts/embeddings/test_qwen3_embedding.py offline
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TASK = "Given a web search query, retrieve relevant passages that answer the query"


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


def build_smoke_inputs() -> tuple[list[str], list[str], list[str]]:
    queries = [
        get_detailed_instruct(TASK, "What is the capital of China?"),
        get_detailed_instruct(TASK, "Explain gravity"),
    ]
    documents = [
        "The capital of China is Beijing.",
        (
            "Gravity is a force that attracts two bodies towards each other. "
            "It gives weight to physical objects and is responsible for the "
            "movement of planets around the sun."
        ),
    ]
    return queries, documents, queries + documents


def similarity_scores(embeddings: torch.Tensor, num_queries: int) -> torch.Tensor:
    return embeddings[:num_queries] @ embeddings[num_queries:].T


def make_client() -> tuple[OpenAI, str]:
    base_url = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
    model = os.getenv("VLLM_MODEL", "Qwen/Qwen3-Embedding-4B")
    api_key = os.getenv("VLLM_API_KEY", "EMPTY")
    client = OpenAI(base_url=base_url, api_key=api_key)
    return client, model


def embed_online(client: OpenAI, model: str, input_texts: list[str]) -> torch.Tensor:
    response = client.embeddings.create(model=model, input=input_texts)
    return torch.tensor([item.embedding for item in response.data])


def run_online() -> None:
    client, model = make_client()
    queries, documents, input_texts = build_smoke_inputs()

    print("mode: online")
    print("base_url:", os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    print("model:", model)
    print(f"input_texts: {len(queries)} queries + {len(documents)} documents")

    embeddings = embed_online(client, model, input_texts)
    scores = similarity_scores(embeddings, len(queries))
    print("similarity scores (queries x documents):")
    print(scores.tolist())


def run_offline() -> None:
    from vllm import LLM

    model_name = os.getenv("VLLM_OFFLINE_MODEL", os.getenv("VLLM_MODEL", "Qwen/Qwen3-Embedding-4B"))
    queries, documents, input_texts = build_smoke_inputs()

    print("mode: offline")
    print("model:", model_name)
    print(f"input_texts: {len(queries)} queries + {len(documents)} documents")

    model = LLM(model=model_name, task="embed")
    outputs = model.embed(input_texts)
    embeddings = torch.tensor([o.outputs.embedding for o in outputs])
    scores = similarity_scores(embeddings, len(queries))
    print("similarity scores (queries x documents):")
    print(scores.tolist())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test Qwen3-Embedding via vLLM (online or offline).",
    )
    parser.add_argument(
        "mode",
        choices=("online", "offline"),
        help="online: API smoke test; offline: in-process LLM.embed",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = _build_parser().parse_args(argv)

    if args.mode == "online":
        run_online()
    else:
        run_offline()


if __name__ == "__main__":
    main(sys.argv[1:])
