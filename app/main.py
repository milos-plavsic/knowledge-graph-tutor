import os

from app.graph_path import explain_with_graph


def main() -> None:
    topic = os.getenv("DEMO_TOPIC", "Backpropagation")
    print("Knowledge Graph Tutor")
    print(explain_with_graph(topic))


if __name__ == "__main__":
    main()
