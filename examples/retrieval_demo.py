from tempfile import TemporaryDirectory

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


def main():
    texts = [
        "RAG retrieves reference passages before generating an answer.",
        "A vector database indexes numeric embeddings.",
        "PDF pages contain document text.",
    ]
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    with TemporaryDirectory() as directory:
        client = chromadb.PersistentClient(
            path=directory, settings=Settings(anonymized_telemetry=False)
        )
        collection = client.create_collection(
            "learning-demo", embedding_function=None, configuration={"hnsw": {"space": "cosine"}}
        )
        collection.add(
            ids=[str(i) for i in range(len(texts))],
            documents=texts,
            embeddings=model.encode(texts, normalize_embeddings=True).tolist(),
        )
        result = collection.query(
            query_embeddings=model.encode(
                ["What does RAG do?"], normalize_embeddings=True
            ).tolist(),
            n_results=2,
        )
        for text, distance in zip(result["documents"][0], result["distances"][0]):
            print(f"Cosine distance {distance:.4f} (lower = closer): {text}")
        print("The temporary collection is separate from your real index.")


if __name__ == "__main__":
    main()
