"""Conservative checks for exact technical facts against cited evidence.

These checks detect mismatches; passing is not a proof of semantic correctness.
"""

import re


def verify_exact_facts(answer, chunks):
    evidence = {f"S{i}": chunk["text"] for i, chunk in enumerate(chunks, 1)}
    warnings = []
    # Check paragraphs/table rows independently: a correct citation elsewhere in
    # the answer must not legitimize a different status, release, or field name.
    for paragraph in re.split(r"\n\s*\n|\n(?=\s*\|)", answer):
        labels = re.findall(r"\[(S\d+)\]", paragraph)
        if not labels:
            continue
        source = "\n".join(evidence.get(label, "") for label in labels)
        facts = re.findall(
            r"\b(?:[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+|[a-z]+(?:[A-Z][a-zA-Z0-9]*)+|[45]\d\d|R\d{2}\+?)\b",
            paragraph,
        )
        missing = sorted(
            {
                fact
                for fact in facts
                if not re.search(r"(?<!\w)" + re.escape(fact) + r"(?!\w)", source)
            }
        )
        if missing:
            warnings.append("Technical values absent from the cited passage: " + ", ".join(missing))
    return list(dict.fromkeys(warnings))


def parse_audit(text, chunks):
    import json

    try:
        data = json.loads(
            text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        )
        if (
            not isinstance(data.get("supported"), bool)
            or not isinstance(data.get("checks"), list)
            or not data["checks"]
        ):
            raise ValueError("Missing audit checks")
        evidence = {f"S{i}": c["text"] for i, c in enumerate(chunks, 1)}
        for check in data["checks"]:
            quote = check.get("quote", "")
            # PDF extraction inserts line breaks/spaces inside otherwise exact quotations.
            # Normalize whitespace only; codes, spelling and punctuation remain constraints.
            normalized_quote = " ".join(quote.split())
            normalized_source = " ".join(evidence.get(check.get("source"), "").split())
            if len(normalized_quote) < 8 or normalized_quote not in normalized_source:
                return ["Answer audit could not be grounded in an exact source quotation."]
        if not data["supported"]:
            return [
                "Answer audit found unsupported claims or mismatched conditions; inspect the cited evidence."
            ]
        return []
    except (ValueError, TypeError, AttributeError, KeyError):
        return ["Answer audit was inconclusive; verify the answer against its sources."]


def missing_error_codes(question, answer, chunks, focus):
    if focus not in {"http", "both"} or re.search(
        r"\b(?:R\d{2}|[45](?:\d\d|xx))\b", question, re.I
    ):
        return []
    text = "\n".join(c["text"] for c in chunks)
    expected = set(
        re.findall(
            r"\b([45]\d\d)\s*(?:NOK|Bad Request|Forbidden|Not Found|Internal Server|Service Unavailable|Not Implemented)",
            text,
        )
    )
    return sorted(code for code in expected if not re.search(r"\b" + code + r"\b", answer))
