"""Tiny retrieval evaluation: checks whether expected keywords occur in top-k chunks."""

from app.rag import RAGEngine

CASES = [
    ("What does RAG connect language models with?", ["organisational", "knowledge"]),
    ("What validates API requests?", ["Pydantic"]),
    ("What framework exposes the service as HTTP endpoints?", ["FastAPI"]),
]

engine = RAGEngine()

hits = 0
for question, expected in CASES:
    results = engine.retrieve(question, top_k=3)
    context = " ".join(r["text"] for r in results).lower()
    ok = all(term.lower() in context for term in expected)
    hits += int(ok)
    print(f"{question} -> {'PASS' if ok else 'FAIL'}")

print(f"Retrieval hit rate: {hits / len(CASES):.2%}")
