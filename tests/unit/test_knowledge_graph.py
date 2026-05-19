"""Unit tests for the KnowledgeGraph domain logic.

Covers: concept lookup, BFS path-finding, DFS prerequisite traversal,
topological sort, learning path generation, quiz generation, and the
explanation engine.  All tests run without a network and without starting
the FastAPI server.
"""

from __future__ import annotations

import pytest

from app.knowledge_graph import KnowledgeGraph, get_knowledge_graph

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def graph() -> KnowledgeGraph:
    """Module-scoped graph so the 60+ node object is created once."""
    return KnowledgeGraph()


# ---------------------------------------------------------------------------
# 1. Graph size - at least 50 nodes
# ---------------------------------------------------------------------------


def test_graph_has_at_least_50_concepts(graph: KnowledgeGraph) -> None:
    concepts = graph.list_concepts()
    assert len(concepts) >= 50, f"Expected ≥50 concepts, got {len(concepts)}"


# ---------------------------------------------------------------------------
# 2. Canonical key resolution
# ---------------------------------------------------------------------------


def test_resolve_canonical_key(graph: KnowledgeGraph) -> None:
    assert graph._resolve("neural_network") == "neural_network"


def test_resolve_alias(graph: KnowledgeGraph) -> None:
    # "MLP" is an alias for neural_network
    assert graph._resolve("MLP") == "neural_network"


def test_resolve_display_name(graph: KnowledgeGraph) -> None:
    # "Gradient Descent" (display name) should resolve to the canonical key
    assert graph._resolve("Gradient Descent") == "gradient_descent"


def test_resolve_unknown_returns_none(graph: KnowledgeGraph) -> None:
    assert graph._resolve("nonexistent_xyz_concept_99") is None


def test_require_raises_on_unknown(graph: KnowledgeGraph) -> None:
    with pytest.raises(KeyError):
        graph._require("nonexistent_xyz_concept_99")


# ---------------------------------------------------------------------------
# 3. BFS path-finding
# ---------------------------------------------------------------------------


def test_path_same_node(graph: KnowledgeGraph) -> None:
    path = graph.find_path("calculus", "calculus")
    assert path == ["calculus"]


def test_path_direct_edge(graph: KnowledgeGraph) -> None:
    # calculus → gradient_descent is a direct edge
    path = graph.find_path("calculus", "gradient_descent")
    assert "calculus" in path
    assert "gradient_descent" in path
    assert path[0] == "calculus"
    assert path[-1] == "gradient_descent"


def test_path_multi_hop(graph: KnowledgeGraph) -> None:
    # linear_algebra → neural_network → attention → self_attention → …
    path = graph.find_path("linear_algebra", "multi_head_attention")
    assert len(path) >= 3
    assert path[0] == "linear_algebra"
    assert path[-1] == "multi_head_attention"


def test_path_returns_empty_when_unreachable(graph: KnowledgeGraph) -> None:
    # xgboost has no outgoing edges leading to linear_algebra
    path = graph.find_path("xgboost", "linear_algebra")
    assert path == []


def test_path_via_alias(graph: KnowledgeGraph) -> None:
    # "SGD" is an alias for gradient_descent
    path = graph.find_path("SGD", "adam_optimizer")
    assert path[0] == "gradient_descent"
    assert path[-1] == "adam_optimizer"


# ---------------------------------------------------------------------------
# 4. Prerequisite traversal
# ---------------------------------------------------------------------------


def test_prerequisites_root_node(graph: KnowledgeGraph) -> None:
    # calculus has no prerequisites → empty list
    prereqs = graph.get_prerequisites("calculus")
    assert prereqs == []


def test_prerequisites_intermediate_node(graph: KnowledgeGraph) -> None:
    prereqs = graph.get_prerequisites("gradient_descent")
    # Should include calculus and loss_function (and their transitive prereqs)
    assert "calculus" in prereqs


def test_prerequisites_deep_node(graph: KnowledgeGraph) -> None:
    prereqs = graph.get_prerequisites("transformer")
    # Transformer requires neural_network, attention, positional_encoding, and
    # their transitive dependencies.
    assert "linear_algebra" in prereqs or "machine_learning" in prereqs


def test_prerequisites_not_include_self(graph: KnowledgeGraph) -> None:
    prereqs = graph.get_prerequisites("backpropagation")
    assert "backpropagation" not in prereqs


# ---------------------------------------------------------------------------
# 5. Related concepts
# ---------------------------------------------------------------------------


def test_related_depth_1(graph: KnowledgeGraph) -> None:
    related = graph.get_related("neural_network", depth=1)
    assert isinstance(related, list)
    # cnn, rnn, attention, embedding etc. should appear
    assert len(related) >= 2


def test_related_depth_2_larger_than_depth_1(graph: KnowledgeGraph) -> None:
    rel1 = set(graph.get_related("neural_network", depth=1))
    rel2 = set(graph.get_related("neural_network", depth=2))
    assert rel2 >= rel1


def test_related_does_not_include_self(graph: KnowledgeGraph) -> None:
    related = graph.get_related("cnn", depth=1)
    assert "cnn" not in related


# ---------------------------------------------------------------------------
# 6. Learning path
# ---------------------------------------------------------------------------


def test_learning_path_empty_known(graph: KnowledgeGraph) -> None:
    path = graph.learning_path("gradient_descent", [])
    assert "gradient_descent" in path
    assert "calculus" in path


def test_learning_path_skips_known(graph: KnowledgeGraph) -> None:
    path = graph.learning_path("gradient_descent", ["calculus", "loss_function"])
    # Known concepts should not appear in the returned plan
    assert "calculus" not in path
    assert "loss_function" not in path
    assert "gradient_descent" in path


def test_learning_path_respects_topological_order(graph: KnowledgeGraph) -> None:
    path = graph.learning_path("backpropagation", [])
    # calculus must appear before gradient_descent, which must appear before backpropagation
    idx = {k: i for i, k in enumerate(path)}
    if "calculus" in idx and "gradient_descent" in idx:
        assert idx["calculus"] < idx["gradient_descent"]
    if "gradient_descent" in idx and "backpropagation" in idx:
        assert idx["gradient_descent"] < idx["backpropagation"]


# ---------------------------------------------------------------------------
# 7. Topological sort - internal helper
# ---------------------------------------------------------------------------


def test_topological_sort_simple(graph: KnowledgeGraph) -> None:
    keys = ["gradient_descent", "calculus", "loss_function"]
    result = graph._topological_sort(keys)
    idx = {k: i for i, k in enumerate(result)}
    # calculus must come before gradient_descent; loss_function before gradient_descent
    assert idx.get("calculus", 99) < idx.get("gradient_descent", 100)


# ---------------------------------------------------------------------------
# 8. Explain output
# ---------------------------------------------------------------------------


def test_explain_returns_string(graph: KnowledgeGraph) -> None:
    text = graph.explain("neural_network")
    assert isinstance(text, str)
    assert len(text) > 100


def test_explain_contains_concept_name(graph: KnowledgeGraph) -> None:
    text = graph.explain("transformer")
    assert "Transformer" in text


def test_explain_contains_prerequisites_section(graph: KnowledgeGraph) -> None:
    text = graph.explain("backpropagation")
    assert "Prerequisite" in text


def test_explain_contains_enables_section(graph: KnowledgeGraph) -> None:
    text = graph.explain("calculus")
    assert "Enables" in text


# ---------------------------------------------------------------------------
# 9. Quiz generation
# ---------------------------------------------------------------------------


def test_quiz_returns_required_keys(graph: KnowledgeGraph) -> None:
    q = graph.quiz_question("neural_network")
    for key in ("question", "options", "correct_index", "correct_answer", "explanation"):
        assert key in q, f"Missing key: {key}"


def test_quiz_correct_index_valid(graph: KnowledgeGraph) -> None:
    q = graph.quiz_question("gradient_descent")
    assert 0 <= q["correct_index"] < len(q["options"])


def test_quiz_options_contains_correct_answer(graph: KnowledgeGraph) -> None:
    q = graph.quiz_question("attention")
    assert q["correct_answer"] in q["options"]


def test_quiz_has_at_least_two_options(graph: KnowledgeGraph) -> None:
    q = graph.quiz_question("relu")
    assert len(q["options"]) >= 2


# ---------------------------------------------------------------------------
# 10. list_concepts structure
# ---------------------------------------------------------------------------


def test_list_concepts_contains_required_fields(graph: KnowledgeGraph) -> None:
    concepts = graph.list_concepts()
    for c in concepts[:5]:  # spot-check first 5
        for field in ("id", "name", "category", "difficulty", "description_preview"):
            assert field in c, f"Field '{field}' missing in concept {c.get('id')}"


def test_list_concepts_difficulty_values(graph: KnowledgeGraph) -> None:
    valid = {"beginner", "intermediate", "advanced"}
    for c in graph.list_concepts():
        assert c["difficulty"] in valid, f"Bad difficulty: {c['difficulty']}"


# ---------------------------------------------------------------------------
# 11. Singleton helper
# ---------------------------------------------------------------------------


def test_get_knowledge_graph_singleton() -> None:
    g1 = get_knowledge_graph()
    g2 = get_knowledge_graph()
    assert g1 is g2


# ---------------------------------------------------------------------------
# 12. Concept category coverage
# ---------------------------------------------------------------------------


def test_multiple_categories_present(graph: KnowledgeGraph) -> None:
    cats = {c["category"] for c in graph.list_concepts()}
    expected = {"mathematics", "machine_learning", "deep_learning", "nlp", "optimisation"}
    assert cats >= expected, f"Missing categories: {expected - cats}"


# ---------------------------------------------------------------------------
# 13. Explanation engine (integrated with graph)
# ---------------------------------------------------------------------------


def test_explanation_engine_with_analogies(graph: KnowledgeGraph) -> None:
    from app.explanation_engine import ExplanationEngine

    engine = ExplanationEngine(graph)
    text = engine.explain_with_analogies("backpropagation")
    assert "analogy" in text.lower() or "like" in text.lower()
    assert len(text) > 200


def test_explanation_engine_compare_same_category(graph: KnowledgeGraph) -> None:
    from app.explanation_engine import ExplanationEngine

    engine = ExplanationEngine(graph)
    result = engine.compare_concepts("relu", "sigmoid")
    assert "ReLU" in result or "Sigmoid" in result
    assert "Similarities" in result
    assert "Differences" in result


def test_explanation_engine_compare_different_category(graph: KnowledgeGraph) -> None:
    from app.explanation_engine import ExplanationEngine

    engine = ExplanationEngine(graph)
    result = engine.compare_concepts("gradient_descent", "clustering")
    assert isinstance(result, str)
    assert len(result) > 100
