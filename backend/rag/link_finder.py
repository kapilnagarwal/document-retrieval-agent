"""Extract real links and source evidence locally; never invent or fetch a URL."""

import html
import re
import time
from urllib.parse import unquote_plus, urlsplit, urlunsplit

from backend.rag.lexical import contains_literal

URL = re.compile(r'https?://[^\s<>"\x27]+', re.I)
# Local vocabulary alias: CNCConsole is the product spelling used in the saved page URLs.
ALIASES = {"cncc": ["cncc", "cncconsole", "cnc console"]}


def clean_url(raw):
    value = html.unescape(raw).rstrip(".,;")
    while value.endswith(")") and value.count(")") > value.count("("):
        value = value[:-1]
    while value.endswith("]") and value.count("]") > value.count("["):
        value = value[:-1]
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
        ):
            return None
        return value
    except ValueError:
        return None


def provider_matches(url, provider):
    if not provider:
        return True
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    if provider == "confluence":
        return (
            "confluence" in host
            or "/pages/viewpage.action" in parsed.path
            or "/display/" in parsed.path
        )
    if provider == "jira":
        return "jira" in host or "/browse/" in parsed.path
    if provider == "gitlab":
        return "gitlab" in host or "/merge_requests/" in parsed.path
    return False


def terms_match(text, terms):
    terms = ["cncc" if term == "cncconsole" else term for term in terms]
    normalized = re.sub(r"[_+/:=.-]+", " ", unquote_plus(text))
    return all(
        any(contains_literal(normalized, alias) for alias in ALIASES.get(term, [term]))
        for term in terms
    )


def nearby_evidence(lines, index):
    result = [lines[index]]
    for direction, steps in [(-1, 2), (1, 1)]:
        for offset in range(1, steps + 1):
            position = index + direction * offset
            if position < 0 or position >= len(lines):
                break
            line = lines[position]
            if not line.strip() or URL.search(line) or re.match(r"\s*[=-]{4,}", line):
                break
            if direction == -1:
                result.insert(0, line)
            else:
                result.append(line)
    return "\n".join(result)


class LinkFinder:
    def __init__(self, catalog):
        self.catalog = catalog

    def find(self, query, plan, document_ids=None, limit=5):
        start = time.perf_counter()
        terms = plan["subject"].split()
        found = {}
        for document in self.catalog.records(document_ids):
            lines = document["content"].splitlines()
            for index, line in enumerate(lines):
                for match in URL.finditer(line):
                    url = clean_url(match.group())
                    if not url or not provider_matches(url, plan["provider"]):
                        continue
                    context = nearby_evidence(lines, index)
                    direct = terms_match(url, terms)
                    if not direct and not terms_match(context, terms):
                        continue
                    parsed = urlsplit(url)
                    key = urlunsplit(
                        (
                            parsed.scheme.lower(),
                            parsed.netloc.lower(),
                            parsed.path,
                            parsed.query,
                            "",
                        )
                    )
                    evidence = dict(
                        document_id=document["document_id"],
                        filename=document["filename"],
                        source_name=document.get("source_name"),
                        line_number=index + 1,
                        text=context,
                    )
                    if key not in found:
                        path = unquote_plus(parsed.path.rsplit("/", 1)[-1])
                        title = path if path and path != "viewpage.action" else "Referenced page"
                        found[key] = dict(
                            url=url,
                            title=title,
                            match_reason="topic_in_url"
                            if direct
                            else "topic_in_nearby_source_text",
                            sources=[],
                        )
                    if evidence not in found[key]["sources"]:
                        found[key]["sources"].append(evidence)
        matches = sorted(
            found.values(), key=lambda item: (item["match_reason"] != "topic_in_url", item["url"])
        )
        maximum = 200 if plan["all_requested"] else limit
        message = (
            f"Found {len(matches)} matching saved link(s). These are references from imported notes; page contents were not fetched."
            if matches
            else "No saved link matches the requested provider and topic in the selected imported sources. No generic passages or invented URLs were substituted."
        )
        if self.catalog.unavailable:
            message += f" {len(self.catalog.unavailable)} source(s) could not be read; coverage is incomplete."
        if len(matches) > maximum:
            message += f" Showing {maximum}; narrow the topic to see more."
        return dict(
            result_type="links",
            query=query,
            plan=plan,
            links=matches[:maximum],
            total_matches=len(matches),
            retrieval_seconds=time.perf_counter() - start,
            message=message,
        )
