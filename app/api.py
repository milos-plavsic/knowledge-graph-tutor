from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.main import explain_with_graph

app = FastAPI(title="Knowledge Graph Tutor", version="0.1.0")


class ExplainRequest(BaseModel):
    topic: str = Field(..., min_length=1)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/explain")
def explain(body: ExplainRequest) -> dict[str, str]:
    text = explain_with_graph(body.topic)
    return {"topic": body.topic, "explanation": text}
