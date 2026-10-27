import json
from pathlib import Path

from scripts.evaluate_quality import answer_checks, evaluate


def test_representative_evaluation_categories_and_fact_coverage(services):
    corpus = json.loads(
        (Path(__file__).resolve().parents[1] / "evaluation/representative_cases.json").read_text()
    )
    # Test structure and retrieval with deterministic vectors; the separately run
    # real-model benchmark is the quality measurement, not this isolated test.
    assert {c["category"] for c in corpus["queries"]} == {
        "normal",
        "errors",
        "fields",
        "releases",
        "followups",
        "unsupported",
    }
    assert len(corpus["queries"]) >= 30
    assert all(c.get("standalone") for c in corpus["queries"] if c["category"] == "followups")
    positive = {
        "query": "EntityResponse fields",
        "required": ["fieldName"],
        "forbidden": ["WRONG_CAUSE"],
        "abstain": False,
    }
    assert answer_checks(positive, {"answer": "fieldName [S1]", "answer_status": "answered"})
    assert not answer_checks(
        positive, {"answer": "fieldName WRONG_CAUSE [S1]", "answer_status": "answered"}
    )
    assert not answer_checks(positive, {"answer": "fieldName [S1]", "answer_status": "unverified"})
    report = evaluate(
        {
            "documents": {"example.txt": "EntityResponse fields:\nfieldName: documented"},
            "queries": [
                {
                    "id": "one",
                    "category": "fields",
                    "query": "EntityResponse fields",
                    "expected": ["example.txt"],
                    "required": ["fieldName"],
                }
            ],
        },
        services,
    )
    assert report["passed"]
    assert report["mode"] == "offline_evidence_only"
    assert report["categories"]["fields"]["answer_pass_rate"] is None
