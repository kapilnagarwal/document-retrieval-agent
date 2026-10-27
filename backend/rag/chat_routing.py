"""Choose when a chat turn needs documents, conversation context, or the web."""

import re


def asks_web(question):
    if re.search(
        r"\b(?:do not|don't)\s+(?:use|search|browse)\b|\bwithout\s+(?:(?:using|searching|browsing)\s+)?(?:the\s+)?(?:web|internet|online)\b",
        question,
        re.I,
    ):
        return False
    return bool(
        re.search(
            r"\b(?:online|internet|public web|search the web|browse|web search|latest|current news|public.*specs?)\b",
            question,
            re.I,
        )
    )


def requires_documents(question):
    return bool(
        re.search(
            r"\b(?:uploaded|indexed|my documents?|my files?|these documents?|according to|based on|from the (?:document|PDF|file))\b|\b[a-z]{2,}\.[0-9]+\b|\.(?:pdf|docx|txt|md)\b",
            question,
            re.I,
        )
    )


def route_turn(request):
    if request.answer_mode != "auto":
        return request.answer_mode
    question = request.question
    if asks_web(question) and (
        not requires_documents(question)
        or re.search(r"\b(?:online|internet|web|browse)\b", question, re.I)
    ):
        return "web"
    if requires_documents(question):
        return "documents"
    if request.history and re.search(
        r"\b(?:summari[sz]e|rephrase|rewrite|reformat|shorter|simpler|simplify|translate|elaborate|explain further|more detail|bullet points|put .* (?:table|list)|explain (?:it|that|this|your answer)|give (?:me )?an example)\b",
        question,
        re.I,
    ):
        return "conversation"
    if re.search(
        r"^\s*(?:hi|hello|hey|thanks|thank you)\b|^\s*(?:write|draft|brainstorm|calculate|translate)\b|\b(?:general knowledge|without (?:using )?(?:RAG|documents)|not from (?:the )?documents)\b",
        question,
        re.I,
    ):
        return "general"
    return "documents"


def conversational_prompt(question, history):
    import json

    # Old document identifiers are scoped to an earlier answer, not this response.
    cleaned = [{**turn, "answer": re.sub(r"\[S\d+\]", "", turn["answer"])} for turn in history]
    return json.dumps({"conversation": cleaned, "current_question": question}, ensure_ascii=False)
