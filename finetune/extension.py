"""Fine-tuning ideas for graph+tutor systems (graph weights, retrieval, or LLM head)."""


def describe_graph_finetune_playbook() -> dict:
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
    import json

    print(json.dumps(describe_graph_finetune_playbook(), indent=2))


if __name__ == "__main__":
    main()
