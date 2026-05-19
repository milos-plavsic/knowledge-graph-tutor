"""Explanation engine wrapping KnowledgeGraph with analogies and concept comparisons."""

from __future__ import annotations

from app.knowledge_graph import KnowledgeGraph, get_knowledge_graph

# ---------------------------------------------------------------------------
# Real-world analogies keyed by concept id
# ---------------------------------------------------------------------------

_ANALOGIES: dict[str, str] = {
    "backpropagation": (
        "Think of backpropagation like tracing blame in a relay race. When the team loses, "
        "you watch the replay from the finish line backwards to see exactly where each runner "
        "slowed down and by how much — then each runner adjusts their technique proportionally."
    ),
    "gradient_descent": (
        "Gradient descent is like descending a foggy mountain blindfolded. You can only feel "
        "the slope directly under your feet. Each step you take in the steepest downhill "
        "direction. The learning rate is how big a step you dare to take in the fog."
    ),
    "attention": (
        "Attention is like a spotlight on a stage. Instead of paying equal attention to every "
        "actor, you dynamically direct the spotlight to whoever is most relevant for the scene "
        "happening right now. The model learns which 'actors' (tokens) to illuminate."
    ),
    "neural_network": (
        "A neural network is like an assembly line where each station (layer) processes the "
        "product (data) and passes it on. Each station specialises: early stations do rough "
        "shaping, late stations add fine detail. The assembly line is configured by training."
    ),
    "transformer": (
        "A Transformer is like a town meeting where everyone can speak to everyone else "
        "simultaneously, rather than passing notes sequentially. Self-attention is the "
        "mechanism by which every participant decides how much to 'listen' to every other."
    ),
    "random_forest": (
        "A random forest is like asking a thousand slightly different experts and taking the "
        "majority vote. Each expert (tree) was trained on a different sample of the data "
        "and only sees a subset of the evidence, ensuring diverse perspectives."
    ),
    "overfitting": (
        "Overfitting is like a student who memorises exam answers instead of understanding "
        "the subject. They score 100% on past exams but fail new questions, because they "
        "learned the specific answers, not the underlying concepts."
    ),
    "regularisation": (
        "Regularisation is like Occam's Razor applied to model parameters: prefer the "
        "simpler explanation. It's as if you tell the model: 'You can use any hypothesis, "
        "but I'll penalise you for each extra complexity you add.'"
    ),
    "relu": (
        "ReLU is like a gate valve: it lets positive values through unchanged but completely "
        "blocks negative values. Its simplicity makes it computationally cheap and its "
        "non-saturation for positives keeps gradients healthy during backpropagation."
    ),
    "loss_function": (
        "The loss function is like a score on a golf course: it tells you how far from the "
        "hole your shot landed. The model's job is to keep playing to minimise the score, "
        "and the loss landscape is the golf course's terrain."
    ),
    "cross_entropy": (
        "Cross-entropy is like measuring how surprised you are by an outcome. If your model "
        "confidently predicts 'cat' and the answer is indeed 'cat', surprise is low. If it "
        "confidently predicts 'dog' but it's a cat, surprise — and hence loss — is very high."
    ),
    "cnn": (
        "A CNN is like the human visual system: the retina detects local edges, the visual "
        "cortex assembles those into shapes, then the prefrontal cortex recognises objects. "
        "Each convolutional layer is a stage in that visual hierarchy."
    ),
    "clustering": (
        "Clustering is like sorting a pile of mixed fruit without any labels. You group "
        "similar-looking items together by feel and colour. The algorithm defines 'similar' "
        "mathematically, but the goal is the same: discover natural groupings."
    ),
    "reinforcement_learning": (
        "Reinforcement learning is like training a dog: you don't explain the rules in "
        "words; you give treats (rewards) for good behaviour and withhold them for bad "
        "behaviour. The dog (agent) figures out the optimal policy through trial and error."
    ),
    "embedding": (
        "An embedding is like a map coordinate for words. Just as GPS places Paris close "
        "to Lyon and far from Tokyo, embeddings place 'king' close to 'queen' and far "
        "from 'bicycle', capturing semantic relationships as geometric proximity."
    ),
    "transfer_learning": (
        "Transfer learning is like a surgeon who learned anatomy for one operation applying "
        "that knowledge to a different procedure. The foundational understanding (weights) "
        "carries over; only the specialised steps (head layers) need relearning."
    ),
    "retrieval_augmented_generation": (
        "RAG is like an open-book exam: instead of relying only on memorised knowledge, "
        "the student (model) can look up relevant reference material before answering. "
        "The retriever is the student finding the right book pages; the generator is the "
        "student synthesising an answer from those pages."
    ),
    "language_model": (
        "A language model is like an autocomplete system trained on a vast library. "
        "Given any text fragment, it predicts the most likely continuation. Scaling up "
        "the library and model size produces the emergent capabilities seen in GPT-4."
    ),
    "decision_tree": (
        "A decision tree is like a game of '20 Questions'. At each node, you ask a yes/no "
        "question about the data. Based on the answer, you branch left or right, progressively "
        "narrowing down the possible outcomes until you reach a leaf with a prediction."
    ),
    "q_learning": (
        "Q-learning is like a rat in a maze that marks each intersection with a 'value' "
        "indicating how good that spot is for reaching cheese. Over many runs, the values "
        "propagate backwards from cheese to start, eventually guiding optimal navigation."
    ),
}

_GENERIC_ANALOGY = (
    "Think of {name} as a modular component in a larger pipeline: it takes structured input, "
    "applies a principled transformation, and produces output that the next stage can build on. "
    "The power comes from composing many such components together."
)


class ExplanationEngine:
    """Generates rich explanations and comparisons using the knowledge graph."""

    def __init__(self, graph: KnowledgeGraph | None = None) -> None:
        self._graph = graph or get_knowledge_graph()

    def explain_with_analogies(self, topic: str) -> str:
        """Return a full explanation enriched with a real-world analogy."""
        key = self._graph._require(topic)
        base_explanation = self._graph.explain(topic)
        concept = self._graph._concepts[key]

        analogy = _ANALOGIES.get(key, _GENERIC_ANALOGY.format(name=concept.name))

        analogy_section = f"### Real-World Analogy\n\n{analogy}"

        # Insert analogy after the first paragraph
        lines = base_explanation.split("\n\n")
        if len(lines) > 1:
            return "\n\n".join(lines[:1] + [analogy_section] + lines[1:])
        return base_explanation + "\n\n" + analogy_section

    def compare_concepts(self, a: str, b: str) -> str:
        """Return a structured comparison of two concepts."""
        key_a = self._graph._require(a)
        key_b = self._graph._require(b)
        ca = self._graph._concepts[key_a]
        cb = self._graph._concepts[key_b]

        lines: list[str] = [f"# Comparing {ca.name} vs {cb.name}\n"]

        # Category comparison
        if ca.category == cb.category:
            lines.append(f"**Both belong to:** {ca.category.replace('_', ' ').title()}\n")
        else:
            lines.append(
                f"**{ca.name}** is in: {ca.category.replace('_', ' ').title()}  \n"
                f"**{cb.name}** is in: {cb.category.replace('_', ' ').title()}\n"
            )

        # Difficulty comparison
        diff_map = {"beginner": 1, "intermediate": 2, "advanced": 3}
        da, db = diff_map.get(ca.difficulty, 2), diff_map.get(cb.difficulty, 2)
        if da == db:
            lines.append(f"**Difficulty:** Both are **{ca.difficulty}** level.\n")
        elif da < db:
            lines.append(
                f"**Difficulty:** {ca.name} ({ca.difficulty}) is simpler than "
                f"{cb.name} ({cb.difficulty}).\n"
            )
        else:
            lines.append(
                f"**Difficulty:** {cb.name} ({cb.difficulty}) is simpler than "
                f"{ca.name} ({ca.difficulty}).\n"
            )

        # Relationship in the graph
        path_ab = self._graph.find_path(key_a, key_b)
        path_ba = self._graph.find_path(key_b, key_a)

        if len(path_ab) > 1:
            intermediate = [self._graph._concepts[k].name for k in path_ab[1:-1]]
            if intermediate:
                lines.append(
                    f"**Dependency:** {ca.name} leads to {cb.name} via "
                    + " → ".join(intermediate)
                    + f" → {cb.name}.\n"
                )
            else:
                lines.append(
                    f"**Dependency:** {ca.name} is a **direct prerequisite** of {cb.name}.\n"
                )
        elif len(path_ba) > 1:
            lines.append(
                f"**Dependency:** {cb.name} is a prerequisite of {ca.name} "
                f"(reverse direction).\n"
            )
        else:
            # Check shared predecessors
            preds_a = set(self._graph.get_prerequisites(key_a))
            preds_b = set(self._graph.get_prerequisites(key_b))
            shared = preds_a & preds_b
            if shared:
                shared_names = [self._graph._concepts[k].name for k in shared]
                lines.append(
                    "**Common foundations:** Both build on "
                    + ", ".join(f"**{n}**" for n in shared_names[:4])
                    + ".\n"
                )
            else:
                lines.append(
                    f"**Relationship:** {ca.name} and {cb.name} are in separate parts of the "
                    "knowledge graph with no direct dependency path.\n"
                )

        # Similarities
        lines.append("## Similarities\n")
        sims: list[str] = []
        if ca.category == cb.category:
            sims.append(f"- Both are in the same field: **{ca.category}**.")
        shared_prereqs = set(self._graph.get_prerequisites(key_a)) & set(
            self._graph.get_prerequisites(key_b)
        )
        if shared_prereqs:
            sims.append(
                "- Both depend on: "
                + ", ".join(self._graph._concepts[k].name for k in list(shared_prereqs)[:3])
                + "."
            )
        if not sims:
            sims.append("- No direct structural similarities detected in the graph.")
        lines.extend(sims)

        # Differences
        lines.append("\n## Differences\n")
        diffs: list[str] = []
        diffs.append(f"- **{ca.name}**: {ca.description.split('.')[0]}.")
        diffs.append(f"- **{cb.name}**: {cb.description.split('.')[0]}.")
        if ca.difficulty != cb.difficulty:
            diffs.append(
                f"- Complexity: {ca.name} is {ca.difficulty}; " f"{cb.name} is {cb.difficulty}."
            )
        lines.extend(diffs)

        return "\n".join(lines)
