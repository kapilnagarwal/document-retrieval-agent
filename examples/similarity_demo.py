import numpy as np
from sentence_transformers import SentenceTransformer


def cosine(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def main():
    texts = [
        "The network encrypts traffic.",
        "A cake bakes in the oven.",
        "TLS protects data in transit.",
    ]
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    query = model.encode("How is network communication secured?")
    ranked = sorted(
        ((cosine(query, vector), text) for text, vector in zip(texts, model.encode(texts))),
        reverse=True,
    )
    for score, text in ranked:
        print(f"{score:.3f}  {text}")
    print("Cosine similarity is a vector comparison, not a probability of relevance.")


if __name__ == "__main__":
    main()
