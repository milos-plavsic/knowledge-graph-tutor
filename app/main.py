import os

from ml_core import configure_logging

from app.graph_path import explain_with_graph

logger = configure_logging(__name__)


def main() -> None:
    """Main."""
    topic = os.getenv("DEMO_TOPIC", "Backpropagation")
    logger.info("Knowledge Graph Tutor")
    logger.info(explain_with_graph(topic))


if __name__ == "__main__":
    main()
