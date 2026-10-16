"""Evaluate the saved model on the separate data/test directory."""

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit(
            "TensorFlow is not installed. Install dependencies from requirements.txt."
        ) from exc

    from src.training import IMAGE_SIZE, _load_split, _prepare_dataset
    from src.metrics import (
        calculate_metrics,
        format_classification_report,
        save_confusion_matrix,
    )

    model_dir = Path(args.model_dir)
    model_file = model_dir / "leafguard_model.keras"
    metadata_file = model_dir / "model_metadata.json"
    if not model_file.is_file() or not metadata_file.is_file():
        raise SystemExit("Trained model or metadata is missing. Run train.py first.")
    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read model metadata: {exc}") from exc
    if not isinstance(metadata, dict):
        raise SystemExit("Model metadata must be a JSON object.")
    class_names = metadata.get("class_names")
    if not isinstance(class_names, list) or not class_names:
        raise SystemExit("Model metadata does not contain a valid class_names list.")
    if metadata.get("preprocessing") != "mobilenet_v2":
        raise SystemExit("Model metadata declares unsupported image preprocessing.")

    test_ds, _ = _load_split(tf, Path(args.data_dir) / "test", class_names)
    test_ds = _prepare_dataset(tf, test_ds, training=False)
    model = tf.keras.models.load_model(model_file, compile=False)
    if tuple(model.input_shape[1:]) != (*IMAGE_SIZE, 3) or model.output_shape[-1] != len(
        class_names
    ):
        raise SystemExit("Model input/output shapes do not match the metadata.")

    true_labels, predicted_labels = [], []
    for images, labels in test_ds:
        true_labels.append(labels.numpy())
        predicted_labels.append(np.argmax(model.predict(images, verbose=0), axis=1))
    if not true_labels:
        raise SystemExit("The test set contains no images.")
    metrics = calculate_metrics(
        np.concatenate(true_labels), np.concatenate(predicted_labels), class_names
    )
    metrics["test_sample_count"] = int(sum(len(batch) for batch in true_labels))
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (reports_dir / "classification_report.txt").write_text(
        format_classification_report(metrics["classification_report"]), encoding="utf-8"
    )
    save_confusion_matrix(
        metrics["confusion_matrix"], class_names, reports_dir / "confusion_matrix.png"
    )
    print(json.dumps({key: metrics[key] for key in (
        "accuracy", "precision_macro", "recall_macro", "f1_macro", "test_sample_count"
    )}, indent=2))


if __name__ == "__main__":
    main()
