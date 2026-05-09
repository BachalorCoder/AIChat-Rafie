from __future__ import annotations

import time
import uuid
from typing import Any

import ollama


class LocalMemory:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.enabled = False
        self.collection = None
        self.embedding_model = config["models"]["embedding"]
        self.top_k = int(config.get("memory", {}).get("top_k", 3))

        try:
            import chromadb

            client = chromadb.PersistentClient(path=config["paths"]["memory"])
            collection_name = config.get("memory", {}).get("collection", "localagent_memory")
            self.collection = client.get_or_create_collection(name=collection_name)
            self.enabled = True
        except Exception as exc:
            print(f"Memory disabled: {exc}")

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        if not self.enabled or not text.strip():
            return

        metadata = metadata or {}
        metadata.setdefault("created_at", time.time())
        doc_id = str(uuid.uuid4())
        embedding = self._embed(text)
        self.collection.add(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata],
            embeddings=[embedding],
        )

    def recall(self, query: str, top_k: int | None = None) -> list[str]:
        if not self.enabled or not query.strip():
            return []

        embedding = self._embed(query)
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k or self.top_k,
        )
        docs = result.get("documents") or []
        return docs[0] if docs else []

    def format_context(self, query: str) -> str:
        memories = self.recall(query)
        if not memories:
            return ""

        lines = "\n".join(f"- {memory}" for memory in memories)
        return f"Relevant memory:\n{lines}"

    def _embed(self, text: str) -> list[float]:
        try:
            response = ollama.embed(model=self.embedding_model, input=[text])
            embeddings = response.get("embeddings") or []
            if embeddings:
                return embeddings[0]
        except Exception:
            pass

        response = ollama.embeddings(model=self.embedding_model, prompt=text)
        return response["embedding"]

