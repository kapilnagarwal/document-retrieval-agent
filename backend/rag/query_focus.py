"""Keep explicit entity questions separate from an earlier conversational topic."""

import re


def technical_entities(question):
    return list(
        dict.fromkeys(
            word
            for word in re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", question)
            if re.search(r"[a-z][A-Z]", word)
            or re.fullmatch(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+", word)
        )
    )


def needs_reference_resolution(question):
    return bool(
        re.search(
            r"\b(?:those|these|that|them|it|above|previous|same|first|second|third)\b",
            question,
            re.I,
        )
    )


def field_question(question):
    entities = technical_entities(question)
    asks_fields = bool(
        re.search(
            r"\b(?:contains?|fields?|attributes?|structure|schema|properties|payload|definition|consists?)\b",
            question,
            re.I,
        )
    )
    asks_definition = bool(
        re.search(r"\b(?:what is|what are|explain|describe|details?|information)\b", question, re.I)
    )
    asks_errors = bool(re.search(r"\b(?:errors?|failures?|rejections?|[45]xx)\b", question, re.I))
    return entities if asks_fields or (asks_definition and not asks_errors) else []


def field_content_query(question):
    """Remove presentation requests, retaining qualifiers and quoted exact phrases."""
    pattern = re.compile(
        r'(?P<quoted>"[^"\n]+"|`[^`\n]+`)|'
        r"\b(?:give|provide|show|tell(?:\s+me)?)\s+(?:me\s+)?(?:more\s+)?(?:details?|information)\b|"
        r"\b(?:in\s+detail|explain|describe|details?|information)\b|"
        r"\b(?:for|about|on|of)\s+(?:this|that|it)\b",
        re.I,
    )
    return (
        re.sub(
            r"\s+",
            " ",
            pattern.sub(lambda match: match[0] if match.group("quoted") else " ", question),
        ).strip(" ,:;?")
        or question
    )


def field_evidence_score(text, entities):
    """Prefer field descriptions following the requested entity over a bare mention."""
    best = 0
    for entity in entities:
        for match in re.finditer(r"(?<!\w)" + re.escape(entity) + r"(?!\w)", text, re.I):
            following = text[match.end() : match.end() + 1000]
            keys = re.findall(
                r'^\s*(?:[•*-]\s*)?["\']?([A-Za-z][A-Za-z0-9_-]*)["\']?\s*:\s*\S',
                following,
                re.M,
            )
            definition = bool(
                re.search(
                    r"\b(?:attributes?|properties|schema|shall contain|data type|field name)\b",
                    following,
                    re.I,
                )
            )
            best = max(best, len(set(keys)) + int(definition))
    return best


def is_correction(question):
    return bool(
        re.search(
            r"\b(?:this|that|it|answer|output|info(?:rmation)?|results?)\b.*\b(?:not\s+(?:the\s+)?(?:correct|right|accurate|proper)|wrong|incorrect|inaccurate|missing)\b|"
            r"\b(?:not\s+(?:the\s+)?(?:correct|right|accurate|proper)|wrong|incorrect|inaccurate)\b.*\b(?:answer|output|info(?:rmation)?|results?)\b|"
            r"^\s*(?:wrong|incorrect|try again|check again)[.!\s]*$",
            question,
            re.I,
        )
    )


def correction_target(question, history):
    if not is_correction(question):
        return None
    return next(
        (turn["question"] for turn in reversed(history) if not is_correction(turn["question"])),
        None,
    )


def error_code_focus(question):
    if not re.search(r"\b(?:codes?|status)\b", question, re.I) or not re.search(
        r"\b(?:handshake|errors?|responses?|HTTP|TLS|alerts?)\b", question, re.I
    ):
        return None
    if re.search(r"\b(?:TLS|alerts?|certificates?)\b", question, re.I):
        return "tls"
    if re.search(r"\b(?:HTTP|capability|SecNegotiate\w*)\b", question, re.I):
        return "http"
    return "both"


def code_content_query(question):
    return re.sub(
        r"\b(?:(?:error|response|status)\s+)?codes?\b|\bstatus\b", " ", question, flags=re.I
    )


def code_evidence_score(text, focus):
    http = len(set(re.findall(r"\b[45]\d{2}\s*(?:NOK|[A-Z][a-z]+|\|)", text)))
    # PDF text may separate the code and NOK by line breaks; a bare error code
    # also counts when response/HTTP/cause terminology is present.
    if re.search(r"(?i)HTTP|NOK|response|ProblemDetail", text):
        http = max(http, len(set(re.findall(r"\b[45]\d{2}\b", text))))
    tls = int(
        bool(re.search(r"(?i)alert|certificate", text) and re.search(r"\b[a-z]+_[a-z_]+\b", text))
    )
    return http if focus == "http" else tls
