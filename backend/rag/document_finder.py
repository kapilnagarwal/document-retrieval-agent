"""Find the source document for a phrase, then read its complete imported text."""

import re
import time

WHOLE_DOCUMENT = re.compile(
    r"\b(?:whole|entire|full|complete)\s+(?:document|note|file)(?:\s+(?:content|text))?\b", re.I
)


def wants_document(query):
    return bool(WHOLE_DOCUMENT.search(query))


def document_phrase(query):
    quoted = re.search(r'"([^"\n]+)"|`([^`\n]+)`', query)
    if quoted:
        return (quoted.group(1) or quoted.group(2)).strip()
    phrase = re.sub(
        r"\b(?:and\s+)?(?:(?:its|the)\s+)?(?:whole|entire|full|complete)\s+(?:document|note|file)(?:\s+(?:content|text))?\b",
        " ",
        query,
        flags=re.I,
    )
    phrase = re.sub(
        r"^\s*(?:please\s+)?(?:find|search(?:\s+for)?|show(?:\s+me)?|give(?:\s+me)?|retrieve)\s+",
        "",
        phrase,
        flags=re.I,
    )
    phrase = re.sub(
        r"^\s*(?:(?:the\s+)?(?:documents?|notes?|files?)\s+)?(?:containing|with|about|for)\s+",
        "",
        phrase,
        flags=re.I,
    )
    phrase = re.sub(
        r"\s+(?:text|content|cmds?|commands?)\s*(?:and\s*(?:its|the)?)?\s*$", "", phrase, flags=re.I
    )
    return " ".join(phrase.strip(" ?.,:;").split())


class DocumentFinder:
    def __init__(self, ingestion):
        self.ingestion = ingestion

    def find(self, query, document_ids=None, limit=5, subject=None, filename=None):
        start = time.perf_counter()
        phrase = subject if subject is not None else document_phrase(query)
        matches, unavailable = [], 0
        if phrase:
            pattern = re.compile(
                r"(?<!\w)" + r"\s+".join(re.escape(word) for word in phrase.split()) + r"(?!\w)",
                re.I,
            )
            allowed = set(document_ids or [])
            for record in self.ingestion.manifest.all():
                if allowed and record["document_id"] not in allowed:
                    continue
                try:
                    document = self.ingestion.content(record["document_id"])
                except (ValueError, OSError):
                    unavailable += 1
                    continue
                if filename:
                    names = [record["filename"], document.get("source_name") or ""]
                    count = int(
                        any(
                            name.replace("\\", "/").rsplit("/", 1)[-1].casefold()
                            == filename.casefold()
                            for name in names
                        )
                    )
                else:
                    count = len(pattern.findall(document["content"]))
                if count:
                    matches.append(
                        dict(
                            document_id=record["document_id"],
                            filename=record["filename"],
                            source_name=document.get("source_name"),
                            character_count=len(document["content"]),
                            match_count=count,
                        )
                    )
        matches.sort(
            key=lambda match: (-match["match_count"], match["filename"], match["document_id"])
        )
        if not phrase:
            message = 'Specify the text to locate, for example: find "what is Supplier" and its whole document content.'
        elif not matches:
            message = f'No imported document contains the phrase "{phrase}". No semantically similar documents were substituted.'
        elif len(matches) > 1:
            message = "More than one source contains the phrase. Choose a document to view its complete text."
        else:
            message = "Showing the complete imported text of the matching source document."
        if unavailable:
            message += f" {unavailable} imported source(s) could not be read."
        return dict(
            result_type="documents",
            query=query,
            search_phrase=phrase,
            documents=matches[:limit],
            total_matches=len(matches),
            retrieval_seconds=time.perf_counter() - start,
            message=message,
        )
