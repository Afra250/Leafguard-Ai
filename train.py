"""Train the LeafGuard MobileNetV2 classifier."""

import argparse
import logging

from src.training import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Root containing train/validation/test")
    parser.add_argument("--model-dir", default="models", help="Directory for model and metadata")
    parser.add_argument("--reports-dir", default="reports", help="Directory for test metrics")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    metrics = train_model(args.data_dir, args.model_dir, args.reports_dir, args.epochs)
    print(
        "Held-out test metrics "
        f"(n={metrics['test_sample_count']}): accuracy={metrics['accuracy']:.4f}, "
        f"macro precision={metrics['precision_macro']:.4f}, "
        f"macro recall={metrics['recall_macro']:.4f}, "
        f"macro F1={metrics['f1_macro']:.4f}"
    )


if __name__ == "__main__":
    main()
