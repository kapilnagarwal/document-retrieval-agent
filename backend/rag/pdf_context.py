"""Recover complete PDF pages and table rows around supported retrieval hits."""

from collections import OrderedDict


class PDFContext:
    def __init__(self, ingestion, max_characters=24000, max_pages=12):
        self.ingestion = ingestion
        self.max_characters = max_characters
        self.max_pages = max_pages
        self.cache = OrderedDict()

    def expand(self, hits, records):
        import pymupdf

        result, seen, characters, pages = [], set(), 0, 0
        for hit in hits:
            record = records.get(hit["document_id"], {})
            page_number = hit.get("page_number")
            if not record.get("filename", "").lower().endswith(".pdf") or not page_number:
                result.append(hit)
                continue
            path = self.ingestion.directory / f"{hit['document_id']}.bin"
            expanded = False
            try:
                with pymupdf.open(path, filetype="pdf") as document:
                    # The preceding page often contains the scenario heading; the next
                    # page often contains the response rows from the same procedure.
                    for number in (page_number, page_number - 1, page_number + 1):
                        key = (hit["document_id"], record["sha256"], number)
                        if key in seen:
                            expanded = True
                            continue
                        if not 1 <= number <= len(document) or pages >= self.max_pages:
                            continue
                        if key not in self.cache:
                            page = document[number - 1]
                            text = page.get_text()
                            # Strict lines ignore filled text backgrounds which can
                            # otherwise be mistaken for individual table rows.
                            try:
                                tables = page.find_tables(strategy="lines_strict").tables
                                if tables:
                                    text += "\n\nStructured tables:\n" + "\n\n".join(
                                        table.to_markdown() for table in tables
                                    )
                            except (ValueError, RuntimeError):
                                pass  # Preserve complete page text if table detection fails.
                            self.cache[key] = text
                            while len(self.cache) > 128:
                                self.cache.popitem(last=False)
                        self.cache.move_to_end(key)
                        text = self.cache[key]
                        if not text.strip() or characters + len(text) > self.max_characters:
                            continue  # Never cut a table row to meet the context budget.
                        result.append(
                            {
                                **hit,
                                "chunk_id": f"{hit['document_id']}:page:{number}",
                                "page_number": number,
                                "text": text,
                                "distance": None,
                                "match_type": "page_context",
                            }
                        )
                        seen.add(key)
                        characters += len(text)
                        pages += 1
                        expanded = True
            except (OSError, ValueError, RuntimeError):
                pass
            if not expanded:
                result.append(hit)
        return result
