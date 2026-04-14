import os


def explain_with_graph(topic: str) -> str:
    path = ["Neural Networks", "Chain Rule", "Backpropagation"]
    return f"Topic: {topic} | reasoning path: {' -> '.join(path)}"


def main() -> None:
    topic = os.getenv("DEMO_TOPIC", "Backpropagation")
    print("Knowledge Graph Tutor")
    print(explain_with_graph(topic))


if __name__ == "__main__":
    main()
