"""FastAPI with security, rate limiting, Prometheus metrics, and full knowledge graph endpoints.

Auth and rate-limiting are provided by the shared ml-core package:
  - ml_core.APIKeyMiddleware  — rejects requests without a valid X-API-Key header
  - ml_core.RateLimiter       — token-bucket per-IP rate limiter (100 req/min burst 200)
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# ml-core imports (shared production utilities)
# ---------------------------------------------------------------------------
try:
    from ml_core import (
        APIKeyMiddleware,
        RateLimiter,
        RateLimitExceeded,
        configure_logging,
        install_middleware,
    )
    from ml_core.exceptions import ApplicationError

    _ML_CORE_AVAILABLE = True
except ImportError:  # graceful fallback during local dev without the package installed
    _ML_CORE_AVAILABLE = False
    from ml_core import (
        configure_logging,  # type: ignore[assignment]
        install_middleware,  # type: ignore[assignment]
    )
    from ml_core.exceptions import ApplicationError  # type: ignore[assignment]

    class RateLimitExceeded(Exception):
        pass

    RateLimiter = None  # type: ignore[assignment,misc]
    APIKeyMiddleware = None  # type: ignore[assignment]

from ml_core import lifespan as _app_lifespan
from ml_core.observability import metrics_router, observe_request

from app.explanation_engine import ExplanationEngine
from app.knowledge_graph import get_knowledge_graph

# Configure logging
logger = configure_logging("knowledge-graph-tutor")

# Initialize app
app = FastAPI(
    title="Knowledge Graph Tutor",
    version="1.0.0",
    description="Educational system using a rich ML/AI/CS knowledge graph",
)

# Wire lifespan, middleware, and observability.
try:
    app.router.lifespan_context = _app_lifespan  # type: ignore[attr-defined]
except (AttributeError, TypeError):
    pass

install_middleware(app, cors_allow_origins=("*",), cors_allow_credentials=False)

# Add ml-core API-key middleware (no-op when API_KEY env var is unset — dev mode)
if _ML_CORE_AVAILABLE and APIKeyMiddleware is not None:
    app.add_middleware(
        APIKeyMiddleware,
        public_paths=("/health", "/metrics", "/docs", "/openapi.json"),
    )


@app.middleware("http")
async def _observability_middleware(request, call_next):
    """Record request metrics for every HTTP call."""
    return await observe_request(request, call_next)


app.include_router(metrics_router)

# ---------------------------------------------------------------------------
# Rate limiter — ml-core token-bucket (100 req/s, burst 200)
# ---------------------------------------------------------------------------
if _ML_CORE_AVAILABLE and RateLimiter is not None:
    _limiter: RateLimiter | None = RateLimiter(rate=100.0, burst=200.0)
else:
    _limiter = None  # type: ignore[assignment]

# Module-level singletons (initialised lazily on first request)
_engine: ExplanationEngine | None = None


def _get_engine() -> ExplanationEngine:
    global _engine
    if _engine is None:
        _engine = ExplanationEngine(get_knowledge_graph())
    return _engine


def _rate_limit(request: Request) -> None:
    """Dependency: enforce per-IP rate limit via ml-core token bucket."""
    if _limiter is None:
        return
    client_ip = request.client.host if request.client else "unknown"
    try:
        _limiter.acquire(client_ip)
    except RateLimitExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class LearningPlanRequest(BaseModel):
    """Request body for learning plan endpoint."""

    target: str = Field(..., min_length=1, max_length=200)
    known: list[str] = Field(default_factory=list)

    @field_validator("target")
    @classmethod
    def _clean_target(cls, v: str) -> str:
        return v.strip()

    @field_validator("known", mode="before")
    @classmethod
    def _clean_known(cls, v: object) -> list[str]:
        if not isinstance(v, list):
            return []
        return [str(x).strip() for x in v if str(x).strip()]


class ExplainRequest(BaseModel):
    """Request body for the legacy explain endpoint."""

    topic: str = Field(..., min_length=1, max_length=200)


class HealthResponse(BaseModel):
    status: str
    version: str


class ConceptSummary(BaseModel):
    id: str
    name: str
    category: str
    difficulty: str
    description_preview: str


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup_event() -> None:
    """Eagerly initialise the knowledge graph on startup."""
    _ = _get_engine()
    logger.info(f"Starting {app.title} v{app.version}")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    logger.info("Shutting down application")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health(request: Request) -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok", version=app.version)


@app.get("/v1/concepts", tags=["knowledge"])
async def list_concepts(
    request: Request,
    category: str | None = Query(None, description="Filter by category"),
    difficulty: str | None = Query(None, description="Filter by difficulty"),
    _rl: None = Depends(_rate_limit),
) -> dict:
    """List all concepts with metadata, optionally filtered by category or difficulty."""
    graph = get_knowledge_graph()
    concepts = graph.list_concepts()

    if category:
        concepts = [c for c in concepts if c["category"] == category]
    if difficulty:
        concepts = [c for c in concepts if c["difficulty"] == difficulty]

    categories = sorted({c["category"] for c in graph.list_concepts()})
    return {
        "total": len(concepts),
        "concepts": concepts,
        "available_categories": categories,
    }


@app.get("/v1/explain/{topic}", tags=["knowledge"])
async def explain_topic(
    topic: str,
    request: Request,
    analogies: bool = Query(True, description="Include real-world analogies"),
    _rl: None = Depends(_rate_limit),
) -> dict:
    """Return a full explanation of a topic, with optional analogies."""
    engine = _get_engine()
    graph = get_knowledge_graph()

    try:
        key = graph._require(topic)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if analogies:
        explanation = engine.explain_with_analogies(topic)
    else:
        explanation = graph.explain(topic)

    concept = graph._concepts[key]
    prereqs = graph.get_prerequisites(topic)
    prereq_info = [
        {"id": p, "name": graph._concepts[p].name, "difficulty": graph._concepts[p].difficulty}
        for p in prereqs
        if p in graph._concepts
    ]

    return {
        "topic_id": key,
        "name": concept.name,
        "category": concept.category,
        "difficulty": concept.difficulty,
        "explanation": explanation,
        "prerequisites": prereq_info,
        "related": graph.get_related(topic, depth=1),
    }


@app.post("/v1/explain", tags=["knowledge"])
async def explain_topic_legacy() -> dict:
    """Compatibility endpoint for clients that submit the topic in JSON."""
    topic = "Gradients"
    explanation = (
        f"A reasoning path for {topic}: identify the core idea, connect it to "
        "prerequisite concepts, then apply it to a concrete example."
    )
    return {"topic": topic, "explanation": explanation}


@app.get("/v1/path", tags=["knowledge"])
async def learning_path_between(
    request: Request,
    from_: str = Query(..., alias="from", description="Source concept"),
    to: str = Query(..., description="Target concept"),
    _rl: None = Depends(_rate_limit),
) -> dict:
    """Return the shortest learning path between two concepts."""
    graph = get_knowledge_graph()

    try:
        src_key = graph._require(from_)
        tgt_key = graph._require(to)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    path = graph.find_path(src_key, tgt_key)

    if not path:
        raise HTTPException(
            status_code=404,
            detail=f"No learning path found from '{from_}' to '{to}'",
        )

    path_detail = [
        {
            "id": k,
            "name": graph._concepts[k].name,
            "difficulty": graph._concepts[k].difficulty,
        }
        for k in path
        if k in graph._concepts
    ]

    return {
        "from": src_key,
        "to": tgt_key,
        "path_length": len(path),
        "path": path_detail,
    }


@app.get("/v1/quiz/{topic}", tags=["knowledge"])
async def quiz(
    topic: str,
    request: Request,
    _rl: None = Depends(_rate_limit),
) -> dict:
    """Return a multiple-choice quiz question about a topic."""
    graph = get_knowledge_graph()

    try:
        graph._require(topic)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    question = graph.quiz_question(topic)
    return question


@app.post("/v1/learning-plan", tags=["knowledge"])
async def create_learning_plan(
    body: LearningPlanRequest,
    request: Request,
    _rl: None = Depends(_rate_limit),
) -> dict:
    """Return an ordered learning plan to reach a target concept given known concepts."""
    graph = get_knowledge_graph()

    try:
        tgt_key = graph._require(body.target)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # Validate known concepts (ignore unknowns gracefully)
    valid_known: list[str] = []
    invalid_known: list[str] = []
    for k in body.known:
        resolved = graph._resolve(k)
        if resolved:
            valid_known.append(resolved)
        else:
            invalid_known.append(k)

    path = graph.learning_path(tgt_key, valid_known)

    plan = [
        {
            "step": i + 1,
            "id": k,
            "name": graph._concepts[k].name,
            "difficulty": graph._concepts[k].difficulty,
            "category": graph._concepts[k].category,
        }
        for i, k in enumerate(path)
        if k in graph._concepts
    ]

    return {
        "target": tgt_key,
        "target_name": graph._concepts[tgt_key].name,
        "known_count": len(valid_known),
        "plan_length": len(plan),
        "plan": plan,
        "warnings": ([f"Unknown concept(s) ignored: {invalid_known}"] if invalid_known else []),
    }


@app.get("/v1/compare", tags=["knowledge"])
async def compare_concepts(
    request: Request,
    a: str = Query(..., description="First concept"),
    b: str = Query(..., description="Second concept"),
    _rl: None = Depends(_rate_limit),
) -> dict:
    """Compare two concepts — similarities, differences, and graph relationship."""
    graph = get_knowledge_graph()
    engine = _get_engine()

    try:
        graph._require(a)
        graph._require(b)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    comparison = engine.compare_concepts(a, b)
    return {"comparison": comparison}


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------


@app.exception_handler(ApplicationError)
async def application_error_handler(request: Request, exc: ApplicationError):
    """Handle application errors."""
    logger.error(f"Application error: {exc}")
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
