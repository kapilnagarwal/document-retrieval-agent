"""Evaluate evidence coverage offline, or generated answers with --live.

Only the committed synthetic corpus is used. Live mode sends that synthetic
context to Gemini and incurs provider usage; it never opens the user's index.
"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from dotenv import load_dotenv

from backend.config import Settings
from backend.models.schemas import ChatRequest
from backend.rag.generator import GeminiGenerator, GenerationError
from backend.rag.prompt import INSUFFICIENT
from backend.services import Services

ROOT = Path(__file__).resolve().parents[1]


class EvidenceOnly:
    def generate(self, prompt):
        return INSUFFICIENT, None


def answer_checks(case, result):
    answer = result["answer"]
    if case.get("abstain"):
        # Explicit uncertainty can coexist with a useful partial answer.
        correct = result["answer_status"] == "insufficient_evidence" or any(
            phrase in answer.lower()
            for phrase in (
                "not specified",
                "not provided",
                "not documented",
                "does not specify",
                "cannot determine",
                "not available",
                "cannot comply",
            )
        )
    else:
        correct = (
            all(value.lower() in answer.lower() for value in case["required"])
            and result["answer_status"] == "answered"
        )
    correct = correct and not any(
        value.lower() in answer.lower() for value in case.get("forbidden", [])
    )
    return correct


def evaluate(corpus, services, live=False):
    for filename, content in corpus["documents"].items():
        services.ingestion.ingest(filename, content.encode())
    details = []
    for case in corpus["queries"]:
        # Offline follow-ups use a labeled standalone query to measure retrieval
        # separately. Live runs exercise the actual model's history rewrite.
        query = case["query"] if live else case.get("standalone") or case["query"]
        result = services.pipeline.answer(
            ChatRequest(question=query, history=case.get("history", []) if live else [], top_k=5)
        )
        ranking = list(dict.fromkeys(c["filename"] for c in result["retrieved_chunks"]))
        context = "\n".join(c["text"] for c in result["retrieved_chunks"])
        expected = set(case["expected"])
        source_recall = (
            len(expected.intersection(ranking)) / len(expected)
            if expected
            else float(not result["evidence"]["sufficient"])
        )
        values = case.get("required", [])
        fact_coverage = (
            sum(v.lower() in context.lower() for v in values) / len(values) if values else None
        )
        details.append(
            {
                **case,
                "ranking": ranking,
                "source_recall": source_recall,
                "fact_coverage": fact_coverage,
                "evidence_sufficient": result["evidence"]["sufficient"],
                "answer_pass": answer_checks(case, result) if live else None,
                "answer": result["answer"] if live else None,
                "warnings": result["warnings"],
            }
        )
        print(case["id"], f"recall={source_recall:.2f}", f"facts={fact_coverage}", flush=True)
    categories = {}
    for category in sorted({c["category"] for c in details}):
        rows = [c for c in details if c["category"] == category]
        facts = [c["fact_coverage"] for c in rows if c["fact_coverage"] is not None]
        categories[category] = {
            "cases": len(rows),
            "source_recall": sum(c["source_recall"] for c in rows) / len(rows),
            "fact_coverage": sum(facts) / len(facts) if facts else None,
            "answer_pass_rate": sum(c["answer_pass"] for c in rows) / len(rows) if live else None,
        }
    passed = all(
        m["source_recall"] >= 0.9
        and (m["fact_coverage"] is None or m["fact_coverage"] >= 0.9)
        and (not live or m["answer_pass_rate"] >= 0.8)
        for m in categories.values()
    )
    return {
        "passed": passed,
        "mode": "live_generation" if live else "offline_evidence_only",
        "categories": categories,
        "limits": "Synthetic cases, lexical answer assertions, and model-based audit do not prove general answer correctness. Offline follow-ups use labeled standalone questions.",
        "queries": details,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--cases", type=Path, default=ROOT / "evaluation/representative_cases.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/evaluation/quality-report.json")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    corpus = json.loads(args.cases.read_text())
    with TemporaryDirectory(prefix="rag-quality-") as directory:
        settings = Settings(
            chroma_persist_directory=Path(directory) / "chroma",
            upload_directory=Path(directory) / "uploads",
            allow_gemini_document_context=args.live,
        )
        generator = GeminiGenerator(settings) if args.live else EvidenceOnly()
        try:
            report = evaluate(corpus, Services(settings, generator=generator), args.live)
        except GenerationError as exc:
            report = {
                "passed": False,
                "mode": "live_generation",
                "categories": {},
                "blocked": str(exc),
                "limits": "Live evaluation incomplete; no answer-quality result is claimed.",
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "mode": report["mode"],
                "categories": report["categories"],
            },
            indent=2,
        )
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
