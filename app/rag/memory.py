"""Per-child long-term memory using ChromaDB.

Embeddings run locally with Chroma's built-in MiniLM model (free) because Groq
and Cerebras do not offer an embeddings endpoint. Each child gets their own
collection so retrieved context is personal.
"""
import time
import uuid

import chromadb

from app.config import settings


class Memory:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.chroma_path)

    def _collection(self, user_id: str):
        return self.client.get_or_create_collection(name=f"user_{user_id}")

    def add_turn(self, user_id: str, child_text: str, tutor_text: str) -> None:
        col = self._collection(user_id)
        doc = f"Child said: {child_text}\nTutor replied: {tutor_text}"
        col.add(
            documents=[doc],
            ids=[uuid.uuid4().hex],
            metadatas=[{"ts": time.time()}],
        )

    def retrieve(self, user_id: str, query: str, k: int = 4) -> list[str]:
        col = self._collection(user_id)
        count = col.count()
        if count == 0:
            return []
        res = col.query(query_texts=[query], n_results=min(k, count))
        docs = res.get("documents") or [[]]
        return docs[0]
