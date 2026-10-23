import time

import numpy as np

from backend.rag.lexical import (
    KeywordIndex,
    contains_literal,
    matching_excerpt,
    query_tokens,
    required_literals,
)


class Retriever:
    def __init__(self, embedder, store, threshold=None):
        self.embedder, self.store, self.threshold = embedder, store, threshold
        self._keyword_index = None
        self._indexed_revision = None

    def search(self, query, top_k=5, document_ids=None, allow_semantic=True, keyword_query=None):
        start = time.perf_counter()
        if self._keyword_index is None or self._indexed_revision != self.store.revision:
            self._keyword_index = KeywordIndex(self.store.retrieval_records())
            self._indexed_revision = self.store.revision
        records = self._keyword_index.records
        allowed = set(document_ids or [])
        literals = required_literals(query)
        eligible = {
            i
            for i, record in enumerate(records)
            if (not allowed or record["document_id"] in allowed)
            and all(contains_literal(record["text"], term) for term in literals)
        }

        def response(chunks, dimension, message=None):
            return dict(
                query=query,
                embedding_dimension=dimension,
                chunks=chunks,
                total_matches=len({records[i]["text"] for i in eligible}),
                retrieval_seconds=time.perf_counter() - start,
                required_terms=literals,
                retrieval_method="literal_first_bm25_cosine",
                message=message,
            )

        if not eligible:
            message = (
                "No passages contain the exact requested version or phrase: "
                + ", ".join(literals)
                + ". Other versions have not been substituted."
                if literals
                else "No indexed passages match the selected documents."
            )
            return response([], 0, message)
        terms = query_tokens(keyword_query if keyword_query is not None else query)
        phrase = " ".join(terms)
        phrase_matches = {
            i for i in eligible if phrase and contains_literal(records[i]["text"], phrase)
        }
        # Literal evidence must not be outranked by a semantically similar but different command.
        # Do not pad an exact match with unrelated nearest neighbours just to fill top-k.
        if phrase_matches:
            eligible, match_type = phrase_matches, "exact_phrase"
        else:
            all_terms = {
                i
                for i in eligible
                if terms and all(i in self._keyword_index.postings.get(term, {}) for term in terms)
            }
            if all_terms:
                eligible, match_type = all_terms, "all_keywords"
            else:
                match_type = "semantic"
        if match_type == "semantic" and not allow_semantic:
            eligible = set()
            return response(
                [],
                0,
                "No supported exact or complete keyword match was found in the selected sources. Try fewer subject words, check source coverage, or explicitly choose Semantic suggestions.",
            )
        vector = np.asarray(self.embedder.query(query), dtype=float)
        distances = {}
        for i in eligible:
            stored = np.asarray(records[i]["embedding"], dtype=float)
            denominator = np.linalg.norm(vector) * np.linalg.norm(stored)
            distances[i] = float(1 - np.dot(vector, stored) / denominator) if denominator else 1.0
        if self.threshold is not None:
            eligible = {i for i in eligible if distances[i] <= self.threshold}
        lexical = self._keyword_index.scores(
            keyword_query if keyword_query is not None else query, eligible
        )
        dense_ranking = sorted(eligible, key=lambda i: (distances[i], records[i]["chunk_id"]))
        sparse_ranking = sorted(lexical, key=lambda i: (-lexical[i], records[i]["chunk_id"]))
        # Reciprocal-rank fusion combines incomparable BM25 and cosine scales without calling them probabilities.
        fusion = {i: 1 / (60 + rank) for rank, i in enumerate(dense_ranking, 1)}
        for rank, i in enumerate(sparse_ranking, 1):
            fusion[i] = fusion.get(i, 0) + 1 / (60 + rank)
        ranking = sorted(eligible, key=lambda i: (-fusion[i], distances[i], records[i]["chunk_id"]))
        hits, seen = [], set()
        for i in ranking:
            record = records[i]
            if record["text"] in seen:
                continue
            seen.add(record["text"])
            hit = {key: value for key, value in record.items() if key != "embedding"}
            hit.update(
                distance=distances[i],
                lexical_score=lexical.get(i, 0),
                hybrid_score=fusion[i],
                match_type=match_type,
                matched_phrase=phrase if match_type == "exact_phrase" else None,
                matched_text=matching_excerpt(record["text"], phrase)
                if match_type == "exact_phrase"
                else None,
            )
            hits.append(hit)
            if len(hits) == top_k:
                break
        message = None
        if not hits:
            message = "No passages meet the configured retrieval distance cutoff."
        elif match_type == "semantic":
            message = "No passage matched all search keywords. These are semantic suggestions; check the source before treating them as an exact answer."
        return response(hits, len(vector), message)
