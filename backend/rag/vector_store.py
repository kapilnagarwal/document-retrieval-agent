import hashlib
from dataclasses import asdict

import chromadb
from chromadb.config import Settings as ChromaSettings


class VectorStore:
    def __init__(self, path, model_name, chunk_size=800, overlap=150):
        self.client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        # Include model AND chunking settings: a changed configuration creates an isolated index.
        signature = hashlib.sha256(f"{model_name}|{chunk_size}|{overlap}".encode()).hexdigest()[:12]
        self.name = f"minilm-{signature}"
        self.revision = 0
        self._retrieval_cache = None
        self.collection = self.client.get_or_create_collection(
            self.name,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embedding_model": model_name},
        )

    def insert(self, chunks, embeddings):
        batch_size = min(256, self.client.get_max_batch_size())
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            metadata = []
            for chunk in batch:
                item = asdict(chunk)
                item.pop("text")
                item.pop("chunk_id")
                metadata.append({key: value for key, value in item.items() if value is not None})
            self.collection.add(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                embeddings=embeddings[start : start + batch_size],
                metadatas=metadata,
            )
            self.revision += 1
            self._retrieval_cache = None

    def search(self, vector, top_k=5, document_ids=None):
        if not self.count():
            return []
        where = {"document_id": {"$in": document_ids}} if document_ids else None
        result = self.collection.query(
            query_embeddings=[vector],
            n_results=min(top_k, self.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return [
            dict(chunk_id=identifier, text=text, distance=distance, **metadata)
            for identifier, text, distance, metadata in zip(
                result["ids"][0],
                result["documents"][0],
                result["distances"][0],
                result["metadatas"][0],
            )
        ]

    def by_document(self, document_id):
        return self.collection.get(
            where={"document_id": document_id}, include=["documents", "metadatas"]
        )

    def list_document_ids(self):
        return sorted(
            {m["document_id"] for m in self.collection.get(include=["metadatas"])["metadatas"]}
        )

    def delete(self, document_id):
        self.collection.delete(where={"document_id": document_id})
        self.revision += 1
        self._retrieval_cache = None

    def count(self):
        return self.collection.count()

    def retrieval_records(self):
        # A cached exact scan is simple and fast for this local learning corpus (~10k chunks).
        # Keep persisted Chroma vectors as the source of truth; no model re-download or reindex needed.
        if self._retrieval_cache is None:
            result = self.collection.get(include=["documents", "metadatas", "embeddings"])
            self._retrieval_cache = [
                dict(chunk_id=identifier, text=text, embedding=embedding, **metadata)
                for identifier, text, embedding, metadata in zip(
                    result["ids"], result["documents"], result["embeddings"], result["metadatas"]
                )
            ]
        return self._retrieval_cache
