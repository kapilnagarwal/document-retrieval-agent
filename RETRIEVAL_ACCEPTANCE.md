# Retrieval acceptance cases

| Behavior | Expected outcome |
| --- | --- |
| Exact command request | Complete matching command lines, including continuations |
| Exact version request | Other versions are excluded |
| Whole-document request | One complete imported source, preserving original text |
| Saved-reference request | Actual saved URLs with source-line evidence; no page fetching |
| Unknown strict lookup | Empty results rather than silent semantic substitution |
| Explicit semantic mode | Approximate results labeled as suggestions |
| Replacement indexing fails | Previously active document remains searchable |
| Replacement succeeds | New active version; superseded history remains inspectable |
| Chat has unrelated nearest passages | Insufficient evidence, no Gemini call |
| Source changes during generation | Response retains captured source/version/hash metadata |

The isolated tests use synthetic sources. The separate labeled retrieval benchmark is described in [evaluation/README.md](evaluation/README.md). Machine-specific historical corpus reports stay in ignored local storage and are not public acceptance evidence.
