import pytest

from backend.rag.query_plan import plan_query, scoped_document_ids


@pytest.mark.parametrize(
    "query,intent,subject,provider",
    [
        ("find confluence pages related to cncc", "links", "cncc", "confluence"),
        ("show me CNCC Confluence URLs", "links", "cncc", "confluence"),
        ("list all confluence links about CNCC", "links", "cncc", "confluence"),
        ("show CNCC pages on Confluence", "links", "cncc", "confluence"),
        ("get references for CNCConsole from confluence", "links", "cncc", "confluence"),
        ("find confluence pages about CNC Console", "links", "cncc", "confluence"),
        ("find jira links about CNCC", "links", "cncc", "jira"),
        ("find podman build cmd", "commands", "podman build", None),
        ("show me commands related to podman build", "commands", "podman build", None),
        ("list helm list commands", "commands", "helm list", None),
        (
            "find what is Supplier text and its whole document content",
            "documents",
            "what is Supplier",
            None,
        ),
        ("show whole document containing Supplier", "documents", "Supplier", None),
        ("find file new 30.txt", "documents", "new 30.txt", None),
        ("new 30.txt", "documents", "new 30.txt", None),
        ("find git add README.md cmd", "commands", "git add readme.md", None),
        ("git add README.md", "commands", "git add readme.md", None),
        ("find notes related to encryption", "passages", "encryption", None),
    ],
)
def test_query_plans(query, intent, subject, provider):
    plan = plan_query(query)
    assert (plan["intent"], plan["subject"], plan["provider"]) == (intent, subject, provider)


def test_explicit_modes_override_auto_intent():
    assert plan_query("find confluence pages related to cncc", "passages")["intent"] == "passages"
    assert plan_query("cncc", "links")["intent"] == "links"


@pytest.mark.parametrize(
    "query",
    [
        "check what is the handshake status code in failure scenarios",
        "Please check what happens when the handshake fails",
        "tell me how handshake errors are handled",
        "show me which status codes indicate failure",
        "find out why the handshake failed",
        "what happens when the handshake fails",
    ],
)
def test_prefaced_questions_allow_semantic_search(query):
    assert plan_query(query)["natural_question"]


@pytest.mark.parametrize("query", ["check handshake", "podman build", "find helm 1.2.3"])
def test_keyword_lookups_remain_strict(query):
    assert not plan_query(query)["natural_question"]


def test_scope_is_intersection_and_empty_never_means_all():
    records = [
        {"document_id": "note", "filename": "note.txt"},
        {"document_id": "guide", "filename": "guide.pdf"},
    ]
    assert scoped_document_ids(records, [], "notes") == ["note"]
    assert scoped_document_ids(records, ["guide"], "notes") == [
        "__no_selected_imported_documents__"
    ]
    assert scoped_document_ids(records, ["unknown"], "all") == [
        "__no_selected_imported_documents__"
    ]


def test_default_scope_handles_more_than_200_documents():
    records = [{"document_id": str(i), "filename": f"note-{i}.txt"} for i in range(350)]
    assert len(scoped_document_ids(records, [], "notes")) == 350
