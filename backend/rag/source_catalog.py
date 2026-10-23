"""Cache complete imported source snapshots, independently of chunk boundaries."""


class SourceCatalog:
    def __init__(self, ingestion):
        self.ingestion = ingestion
        self.cache = {}
        self.unavailable = []

    def records(self, document_ids=None):
        self.unavailable = []
        allowed = set(document_ids or [])
        active = {}
        names = self.ingestion.source_names()
        for record in self.ingestion.manifest.all():
            identifier = record["document_id"]
            key = (identifier, record["sha256"])
            active[key] = True
            if allowed and identifier not in allowed:
                continue
            if key not in self.cache:
                try:
                    self.cache[key] = self.ingestion.content(identifier)
                except (OSError, ValueError, KeyError):
                    self.unavailable.append(identifier)
                    continue
            yield {
                **self.cache[key],
                "source_name": names.get(identifier) or self.cache[key].get("source_name"),
            }
        self.cache = {key: value for key, value in self.cache.items() if key in active}
