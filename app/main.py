from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.rag import RAGEngine

app = FastAPI(
    title="Enterprise RAG Assistant",
    version="1.0.0",
    description="A compact retrieval-augmented generation API."
)

engine = RAGEngine()


class QueryRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


class Source(BaseModel):
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok", "documents": len(engine.chunks)}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    return engine.generate(request.query, request.top_k)
