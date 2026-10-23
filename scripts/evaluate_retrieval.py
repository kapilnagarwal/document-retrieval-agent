"""Measure real local retrieval and abstention on a synthetic labeled corpus."""

import argparse
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from backend.config import Settings
from backend.rag.evidence import assess_evidence
from backend.services import Services

ROOT = Path(__file__).resolve().parents[1]


def score_results(
    cases: list[dict], rankings: dict[str, list[str]], supported: dict[str, bool]
) -> dict:
    """Compute macro recall, reciprocal rank, and negative-query false support."""
    recalls, reciprocal_ranks, false_support = [], [], []
    for case in cases:
        expected = set(case["expected"])
        ranking = rankings[case["id"]]
        if expected:
            recalls.append(len(expected.intersection(ranking)) / len(expected))
            reciprocal_ranks.append(
                next((1 / rank for rank, name in enumerate(ranking, 1) if name in expected), 0.0)
            )
        else:
            false_support.append(float(supported[case["id"]]))
    if not recalls or not false_support:
        raise ValueError("Evaluation needs both relevant-source and negative queries")
    return {
        "recall_at_k": sum(recalls) / len(recalls),
        "mrr_at_k": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "false_support_rate": sum(false_support) / len(false_support),
        "positive_queries": len(recalls),
        "negative_queries": len(false_support),
    }


def evaluate(corpus: dict, services: Services, top_k: int = 5) -> dict:
    for name, text in corpus["documents"].items():
        services.ingestion.ingest(name, text.encode())
    timings, details, rankings, supported = [], [], {}, {}
    for case in corpus["queries"]:
        started = time.perf_counter()
        result = services.retriever.search(case["query"], top_k)
        assessment = assess_evidence(
            case["query"], result["chunks"], services.settings.chat_min_keyword_coverage
        )
        elapsed = time.perf_counter() - started
        rankings[case["id"]] = list(dict.fromkeys(hit["filename"] for hit in result["chunks"]))
        supported[case["id"]] = assessment.sufficient
        timings.append(elapsed)
        details.append(
            {
                **case,
                "ranking": rankings[case["id"]],
                "evidence": assessment.model_dump(),
                "seconds": elapsed,
            }
        )
    positive = [case for case in corpus["queries"] if case["expected"]]
    return {
        "metrics": {
            **score_results(corpus["queries"], rankings, supported),
            "positive_support_rate": sum(supported[c["id"]] for c in positive) / len(positive),
            "mean_seconds": sum(timings) / len(timings),
            "p95_seconds": sorted(timings)[max(0, int(0.95 * len(timings) + 0.999) - 1)],
        },
        "top_k": top_k,
        "embedding_model": services.settings.embedding_model,
        "minimum_keyword_coverage": services.settings.chat_min_keyword_coverage,
        "queries": details,
        "scope": "Synthetic local corpus only; no Gemini calls or private documents.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "evaluation/retrieval_cases.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/evaluation/report.json")
    parser.add_argument("--top-k", type=int, choices=range(1, 31), default=5)
    parser.add_argument("--minimum-coverage", type=float, default=0.5)
    parser.add_argument("--min-recall", type=float, default=0.9)
    parser.add_argument("--min-mrr", type=float, default=0.75)
    parser.add_argument("--max-false-support", type=float, default=0.1)
    parser.add_argument("--min-positive-support", type=float, default=0.8)
    args = parser.parse_args()
    corpus = json.loads(args.cases.read_text())
    with TemporaryDirectory(prefix="retrieval-evaluation-") as directory:
        settings = Settings(
            _env_file=None,
            chroma_persist_directory=Path(directory) / "chroma",
            upload_directory=Path(directory) / "uploads",
            chunk_size=800,
            chunk_overlap=150,
            chat_min_keyword_coverage=args.minimum_coverage,
            allow_gemini_document_context=False,
        )
        report = evaluate(corpus, Services(settings), args.top_k)
    metrics = report["metrics"]
    report["passed"] = (
        metrics["recall_at_k"] >= args.min_recall
        and metrics["mrr_at_k"] >= args.min_mrr
        and metrics["false_support_rate"] <= args.max_false_support
        and metrics["positive_support_rate"] >= args.min_positive_support
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], **metrics}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
