import time
from contextlib import nullcontext
from copy import deepcopy

from backend.models.schemas import ChatRequest
from backend.rag.evidence import assess_evidence
from backend.rag.prompt import INSUFFICIENT, SYSTEM, build_prompt, validate_citations


class RAGPipeline:
    def __init__(self, retriever, generator, default_top_k=5, ingestion=None, minimum_coverage=0.5):
        self.retriever, self.generator, self.default_top_k = retriever, generator, default_top_k
        self.ingestion = ingestion
        self.minimum_coverage = minimum_coverage

    def answer(self, request: ChatRequest):
        start = time.perf_counter()
        # Freeze evidence and version metadata while storage is locked. Never hold
        # this lock across a provider request, retries, or citation validation.
        with self.ingestion.lock if self.ingestion else nullcontext():
            records = (
                {record["document_id"]: record for record in self.ingestion.manifest.all()}
                if self.ingestion
                else {}
            )
            identifiers = request.document_ids
            if self.ingestion:
                identifiers = [
                    identifier
                    for identifier in records
                    if not identifiers or identifier in identifiers
                ]
                identifiers = identifiers or ["__no_active_documents__"]
            result = deepcopy(
                self.retriever.search(
                    request.question, request.top_k or self.default_top_k, identifiers
                )
            )
            for hit in result["chunks"]:
                record = records.get(hit["document_id"], {})
                hit.update(
                    source_id=record.get("source_id", hit["document_id"]),
                    version=record.get("version", 1),
                    sha256=record.get("sha256"),
                )
        assessment = assess_evidence(request.question, result["chunks"], self.minimum_coverage)
        supported = [
            hit for hit in result["chunks"] if hit["chunk_id"] in assessment.supporting_chunk_ids
        ]
        context, prompt, sources = build_prompt(request.question, supported)
        generation_seconds, usage, warnings = 0, None, []
        answer_status = "insufficient_evidence"
        if assessment.sufficient:
            generation_start = time.perf_counter()
            answer, usage = self.generator.generate(prompt)
            generation_seconds = time.perf_counter() - generation_start
            answer, sources, warnings = validate_citations(answer, sources)
            answer_status = (
                "insufficient_evidence"
                if answer.strip() == INSUFFICIENT
                else ("unverified" if warnings else "answered")
            )
        else:
            answer = INSUFFICIENT
        return dict(
            question=request.question,
            answer=answer,
            answer_status=answer_status,
            evidence=assessment.model_dump(),
            sources=sources,
            retrieved_chunks=result["chunks"],
            retrieval_seconds=result["retrieval_seconds"],
            generation_seconds=generation_seconds,
            total_seconds=time.perf_counter() - start,
            token_usage=usage,
            warnings=warnings,
            debug={
                "embedding_dimension": result["embedding_dimension"],
                "context": context,
                "system_prompt": SYSTEM,
                "user_prompt": prompt,
            }
            if request.debug
            else None,
        )
