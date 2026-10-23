import re

from fastapi import APIRouter, Request

from backend.models.schemas import (
    DocumentSearchResponse,
    LinkSearchResponse,
    SearchRequest,
    SearchResponse,
)
from backend.rag.query_plan import plan_query, scoped_document_ids

router = APIRouter()


@router.post("/search", response_model=SearchResponse | DocumentSearchResponse | LinkSearchResponse)
def search(body: SearchRequest, request: Request):
    services = request.app.state.services
    plan = plan_query(body.query, body.mode)
    with services.ingestion.lock:
        identifiers = scoped_document_ids(
            services.ingestion.manifest.all(), body.document_ids, body.scope
        )
        limit = 200 if plan["all_requested"] else (body.top_k or services.settings.retrieval_top_k)
        if plan["intent"] == "links":
            return services.link_finder.find(body.query, plan, identifiers, limit)
        if plan["intent"] == "documents":
            result = services.document_finder.find(
                body.query, identifiers, limit, subject=plan["subject"], filename=plan["filename"]
            )
            result["plan"] = plan
            return result
        if (
            plan["intent"] == "commands"
            and body.mode != "semantic"
            and re.search(
                r"\b(?:podman|docker|kubectl|helm|git|curl|gradle|mvn)\b", body.query, re.I
            )
        ):
            return services.command_finder.find(body.query, plan, identifiers, limit)
        result = services.retriever.search(
            body.query,
            limit,
            identifiers,
            allow_semantic=body.mode == "semantic"
            or (body.mode == "auto" and plan["natural_question"]),
            keyword_query=plan["subject"],
        )
        result["plan"] = plan
        if result["total_matches"] > len(result["chunks"]):
            suffix = f" Showing {len(result['chunks'])} of {result['total_matches']} eligible matches; increase the limit or narrow the query."
            result["message"] = (result.get("message") or "") + suffix
        names = services.ingestion.source_names()
        for hit in result["chunks"]:
            hit["source_name"] = names.get(hit["document_id"])
        return result
