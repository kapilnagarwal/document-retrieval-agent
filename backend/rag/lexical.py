"""Local keyword ranking and explicit literal constraints for technical notes."""

import math
import re
from collections import Counter, defaultdict

STOP_WORDS = set(
    "a an the is are was were does do did it contain contains find show me give get all any my notes note document documents what which where how can could would should please cmds cmd commands command with for of in on and or to from about tags version versions".split()
)
STOP_WORDS.update({"i", "its", "using", "use"})
VERSION = re.compile(r"(?<![\w.])v?\d+(?:\.\d+)+(?:[-+][A-Za-z0-9]+(?:[.-]\d+)*)?(?![\w])", re.I)


def tokens(text):
    return [
        word
        for word in re.findall(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", text.lower())
        if word not in STOP_WORDS
    ]


def required_literals(query):
    # Explicit dotted versions and quoted phrases constrain search rather than merely boosting it.
    quoted = [match[0] or match[1] for match in re.findall(r'"([^"\n]+)"|`([^`\n]+)`', query)]
    versions = VERSION.findall(query)
    return list(dict.fromkeys(term.lower() for term in quoted + versions if term.strip()))


def contains_literal(text, literal):
    if VERSION.fullmatch(literal):
        # Do not confuse 1.0.4-3 with 1.0.4-30 or 1.0.4-3.1. Archive suffixes remain allowed.
        pattern = r"(?<![\w.])" + re.escape(literal) + r"(?![A-Za-z0-9]|[.-]\d)"
        return re.search(pattern, text, re.I) is not None
    return phrase_pattern(literal).search(text) is not None


def query_tokens(query):
    query = re.sub(
        r"^\s*(?:please\s+)?(?:find|show|give|get|list|retrieve|search(?:\s+for)?)\s+(?:me\s+)?",
        "",
        query,
        flags=re.I,
    )
    return list(dict.fromkeys(tokens(query)))


def phrase_pattern(phrase):
    return re.compile(
        r"(?<!\w)" + r"\s+".join(re.escape(word) for word in phrase.split()) + r"(?!\w)", re.I
    )


def matching_excerpt(text, phrase):
    lines = text.splitlines()
    selected = set()
    for match in phrase_pattern(phrase).finditer(text):
        start = text[: match.start()].count("\n")
        end = text[: match.end()].count("\n")
        # Preserve shell commands that continue onto subsequent lines using backslashes.
        while start > 0 and lines[start - 1].rstrip().endswith("\\"):
            start -= 1
        while end < len(lines) - 1 and lines[end].rstrip().endswith("\\"):
            end += 1
        selected.update(range(start, end + 1))
    return "\n".join(lines[index] for index in sorted(selected))


class KeywordIndex:
    """Small in-memory BM25 index. Rebuilt only when the local vector store changes."""

    def __init__(self, records):
        self.records = records
        self.lengths = []
        self.postings = defaultdict(dict)
        for index, record in enumerate(records):
            counts = Counter(tokens(record["text"]))
            self.lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                self.postings[term][index] = frequency
        self.average_length = sum(self.lengths) / max(1, len(records))

    def scores(self, query, eligible):
        scores = defaultdict(float)
        count = len(self.records)
        for term in query_tokens(query):
            postings = self.postings.get(term, {})
            idf = math.log(1 + (count - len(postings) + 0.5) / (len(postings) + 0.5))
            for index, frequency in postings.items():
                if index not in eligible:
                    continue
                length_ratio = self.lengths[index] / max(1, self.average_length)
                scores[index] += (
                    idf * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * length_ratio))
                )
        return scores
