"""Resolve explicit source names against the currently allowed documents."""

import re
from pathlib import Path


def named_source_scope(query, records, allowed):
    records = list(records)
    matches = []
    for exact in (True, False):
        for record in records:
            filename = record["filename"]
            aliases = (
                [filename, Path(filename).stem]
                if exact
                else re.findall(r"\b[a-z]{2,}\d*(?:[.-]\d+)+", filename, re.I)
            )
            for alias in aliases:
                if len(alias) >= 4 and re.search(
                    r"(?<!\w)" + re.escape(alias) + r"(?!\w)", query, re.I
                ):
                    matches.append((record["document_id"], alias))
                    break
        if matches:
            break
    if not matches:
        return allowed, query
    cleaned = query
    for _, alias in matches:
        cleaned = re.sub(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\b(?:according to|based on)\b", " ", cleaned, flags=re.I)
    scoped = list(
        dict.fromkeys(identifier for identifier, _ in matches if identifier in set(allowed))
    )
    return scoped or ["__no_selected_imported_documents__"], cleaned.strip(" ,:;") or query
