"""Recover bounded adjacent text when a definition spans chunk boundaries."""

from pathlib import Path


def expand_text_context(seeds, records, max_characters=12000):
    by_document = {}
    for record in records:
        if Path(record["filename"]).suffix.lower() != ".pdf":
            by_document.setdefault(record["document_id"], {})[record["chunk_index"]] = record
    expanded, seen, used = [], set(), 0
    for seed in seeds:
        siblings = by_document.get(seed["document_id"])
        if not siblings:
            expanded.append(seed)
            continue
        index = seed["chunk_index"]
        neighbors = [siblings[i] for i in range(index - 2, index + 3) if i in siblings]
        unseen = [c for c in neighbors if c["chunk_id"] not in seen]
        if not unseen:
            continue
        text = "\n\n".join(c["text"] for c in unseen)
        if used + len(text) > max_characters:
            if seed["chunk_id"] not in seen and used + len(seed["text"]) <= max_characters:
                expanded.append(seed)
                seen.add(seed["chunk_id"])
                used += len(seed["text"])
            continue
        expanded.append(
            {
                **seed,
                "text": text,
                "context_chunk_ids": [c["chunk_id"] for c in unseen],
                "match_type": "adjacent_context",
            }
        )
        seen.update(c["chunk_id"] for c in unseen)
        used += len(text)
    return expanded
