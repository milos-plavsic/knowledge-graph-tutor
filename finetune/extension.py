"""Fine-tuning ideas for graph+tutor systems (graph weights, retrieval, or LLM head)."""

from ml_core import configure_logging

logger = configure_logging(__name__)


def describe_graph_finetune_playbook() -> dict:
    """Execute the describe graph finetune playbook routine."""
    return {
        "graph_side": [
            "Refine edge weights from student outcome logs (personalized prerequisite strength).",
            "Active learning: add/remove edges where tutor explanations fail downstream quizzes.",
        ],
        "llm_side": [
            "LoRA on the explanation head using (query, gold path, teacher rewrite) triples.",
        ],
        "data": "Use `data/stem_prereqs.json` as seed; expand with district curriculum CSVs.",
    }


def main() -> None:
    """Execute the main routine."""
    import json

    logger.info(json.dumps(describe_graph_finetune_playbook(), indent=2))


if __name__ == "__main__":
    main()
