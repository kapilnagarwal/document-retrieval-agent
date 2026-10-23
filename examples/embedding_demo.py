"""Run from any directory: python /path/to/examples/embedding_demo.py"""

import numpy as np
from sentence_transformers import SentenceTransformer


def main():
    sentences = [
        "A cat is sleeping on the sofa.",
        "A kitten naps on a couch.",
        "A router forwards network packets.",
    ]
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    vectors = model.encode(sentences, normalize_embeddings=True)
    print("Shape:", vectors.shape, "— one 384-dimensional vector per sentence")
    # Unit-length vectors make their dot product equal cosine similarity.
    print("Cosine similarities:\n", np.round(vectors @ vectors.T, 3))
    print(
        "Related meanings usually have closer vectors, even with different words. This is learned, not guaranteed."
    )


if __name__ == "__main__":
    main()
