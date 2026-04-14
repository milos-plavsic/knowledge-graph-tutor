from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

GRAPH_NOTE = (
    "Curriculum-style prerequisite graph bundled as JSON (education sector). "
    "See `data/stem_prereqs.json` and README for references."
)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_prereq_graph() -> nx.DiGraph:
    path = project_root() / "data" / "stem_prereqs.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    g = nx.DiGraph()
    for n in payload["nodes"]:
        g.add_node(n["id"], label=n["label"])
    for e in payload["edges"]:
        g.add_edge(e["from"], e["to"])
    return g


def topic_to_target(topic: str) -> str:
    t = topic.lower()
    if any(k in t for k in ("momentum", "collision")):
        return "momentum"
    if any(k in t for k in ("energy", "work")):
        return "energy"
    if any(k in t for k in ("kinematic", "velocity", "acceleration")):
        return "kinematics"
    if any(k in t for k in ("vector", "component")):
        return "vectors_intro"
    if any(k in t for k in ("gradient", "derivative", "backprop", "neural", "chain")):
        return "functions"
    if "triangle" in t or "geometry" in t:
        return "geometry_right_tri"
    return "algebra_linear"


def explain_with_graph(topic: str) -> str:
    g = load_prereq_graph()
    target = topic_to_target(topic)
    if target not in g:
        target = "algebra_linear"
    start = "fractions"
    if not nx.has_path(g, start, target):
        return f"Topic: {topic} | no prerequisite path from {start} to {target}"
    path = nx.shortest_path(g, start, target)
    labels = [g.nodes[n]["label"] for n in path]
    return f"Topic: {topic} | reasoning path: {' -> '.join(labels)} | {GRAPH_NOTE}"
