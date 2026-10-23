"""Retrieve complete matching command lines from imported sources, without chunk boundaries."""

import re
import time

from backend.rag.lexical import contains_literal, matching_excerpt, query_tokens

TOOLS = r"podman|docker|kubectl|helm|git|curl|gradle|mvn"
START = re.compile(r"^\s*(?:(?:\d+[.)]|[$>#])\s*)?(?:sudo\s+)?(?:" + TOOLS + r")\b", re.I)


class CommandFinder:
    def __init__(self, catalog):
        self.catalog = catalog

    def find(self, query, plan, document_ids, limit):
        start = time.perf_counter()
        subject = plan["subject"]
        # 'with version tag' describes a constraint; 'podman tag' names a subcommand.
        if plan["required_terms"] and not re.search(r"\b(?:podman|docker)\s+tag\b", query, re.I):
            subject = re.sub(r"\btags?\b", "", subject, flags=re.I)
        terms = query_tokens(subject)
        hits = []
        for document in self.catalog.records(document_ids):
            lines = document["content"].splitlines()
            matches = []
            seen = set()
            for index, line in enumerate(lines):
                if not START.search(line):
                    continue
                end = index
                while end < len(lines) - 1 and lines[end].rstrip().endswith("\\"):
                    end += 1
                command = "\n".join(lines[index : end + 1])
                if (
                    terms
                    and all(contains_literal(command, term) for term in terms)
                    and all(contains_literal(command, term) for term in plan["required_terms"])
                ):
                    if command not in seen:
                        matches.append((index + 1, command))
                        seen.add(command)
            if not matches:
                continue
            text = "\n".join(command for _, command in matches)
            phrase = " ".join(terms)
            exact = bool(phrase and matching_excerpt(text, phrase))
            hits.append(
                dict(
                    chunk_id="commands:" + document["document_id"],
                    document_id=document["document_id"],
                    filename=document["filename"],
                    source_name=document.get("source_name"),
                    chunk_index=0,
                    line_number=matches[0][0],
                    command_count=len(matches),
                    text=text,
                    matched_text=text,
                    matched_phrase=phrase if exact else None,
                    match_type="exact_phrase" if exact else "all_keywords",
                    distance=None,
                )
            )
        hits.sort(
            key=lambda item: (item.get("source_name") or item["filename"], item["document_id"])
        )
        total = len(hits)
        message = (
            f"Found {sum(hit['command_count'] for hit in hits)} matching command(s) in {total} source(s). Only matching command lines are shown."
            if hits
            else "No complete command line matches every requested term in the selected imported sources. Other commands or versions were not substituted."
        )
        if total > limit:
            message += (
                f" Showing {limit} of {total} matching sources; increase the limit or ask for all."
            )
        if self.catalog.unavailable:
            message += f" {len(self.catalog.unavailable)} imported source(s) could not be read."
        return dict(
            result_type="passages",
            query=query,
            plan=plan,
            embedding_dimension=0,
            chunks=hits[:limit],
            total_matches=total,
            retrieval_seconds=time.perf_counter() - start,
            required_terms=plan["required_terms"],
            retrieval_method="complete_source_command_lines",
            message=message,
        )
