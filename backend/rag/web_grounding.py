"""Use provider grounding metadata, never model-invented URLs, for web citations."""

from urllib.parse import urlsplit


def public_url(url):
    try:
        if any(character.isspace() or character in "<>" for character in url):
            return False
        parts = urlsplit(url)
        return (
            parts.scheme == "https"
            and bool(parts.hostname)
            and not parts.username
            and not parts.password
        )
    except (ValueError, TypeError):
        return False


def collect_grounding(response, output):
    candidates = getattr(response, "candidates", None)
    if not isinstance(candidates, list) or not candidates:
        return
    metadata = getattr(candidates[0], "grounding_metadata", None)
    if metadata is None:
        return
    dump = metadata.model_dump(mode="json", exclude_none=True)
    # Metadata often arrives only in a final streaming chunk. Preserve earlier parts.
    for key, value in dump.items():
        if value:
            output[key] = value


def ground_answer(text, metadata):
    chunks = metadata.get("grounding_chunks", [])
    sources, index_map = [], {}
    for index, chunk in enumerate(chunks):
        web = chunk.get("web", {})
        url = web.get("uri", "")
        if public_url(url):
            index_map[index] = len(sources) + 1
            sources.append({"title": web.get("title") or urlsplit(url).hostname, "url": url})
    insertions = {}
    for support in metadata.get("grounding_supports", []):
        segment = support.get("segment", {})
        end = segment.get("end_index")
        if not isinstance(end, int) or not 0 < end <= len(text):
            continue
        # Validate the reported segment before inserting into Unicode text.
        start = segment.get("start_index", 0)
        if segment.get("text") and text[start:end] != segment["text"]:
            continue
        links = []
        for index in support.get("grounding_chunk_indices", []):
            number = index_map.get(index)
            if number:
                links.append(f"[W{number}]({sources[number - 1]['url']})")
        if links:
            insertions.setdefault(end, set()).update(links)
    for end in sorted(insertions, reverse=True):
        text = text[:end] + " " + " ".join(sorted(insertions[end])) + text[end:]
    return text, sources, bool(insertions)
