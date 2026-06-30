"""PCA visualization for TankChallenge RAG embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from tank_rag import (
    DEFAULT_FAISS_META,
    DEFAULT_MODEL_NAME,
    build_faiss_index,
    read_index,
)


HERE = os.path.dirname(__file__)
DEFAULT_PNG = os.path.join(HERE, "static", "pca_embedding_map.png")
DEFAULT_CSV = os.path.join(HERE, "pca_embedding_points.csv")


def _load_cases(meta_path: str) -> list[dict]:
    if not os.path.exists(meta_path):
        build_faiss_index()
    with open(meta_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    return metadata["cases"]


def _embed_cases(cases: list[dict], model_name: str):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    documents = [case["document"] for case in cases]
    return model.encode(documents, batch_size=32, show_progress_bar=True)


def build_pca_visualization(
    model_name: str = DEFAULT_MODEL_NAME,
    meta_path: str = DEFAULT_FAISS_META,
    output_png: str = DEFAULT_PNG,
    output_csv: str = DEFAULT_CSV,
) -> dict:
    cases = _load_cases(meta_path)
    if not cases:
        read_index()
        build_faiss_index(model_name=model_name)
        cases = _load_cases(meta_path)

    embeddings = np.asarray(_embed_cases(cases, model_name), dtype="float32")
    pca = PCA(n_components=2, random_state=42)
    points = pca.fit_transform(embeddings)

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case_id",
                "source",
                "hit_label",
                "target_type",
                "distance",
                "impact_error",
                "pca_x",
                "pca_y",
            ]
        )
        for case, point in zip(cases, points):
            writer.writerow(
                [
                    case["case_id"],
                    case["source"],
                    case["hit_label"],
                    case["target_type"],
                    case["distance"],
                    case["impact_error"],
                    float(point[0]),
                    float(point[1]),
                ]
            )

    success_x = [point[0] for case, point in zip(cases, points) if case["hit_label"] == "success"]
    success_y = [point[1] for case, point in zip(cases, points) if case["hit_label"] == "success"]
    failure_x = [point[0] for case, point in zip(cases, points) if case["hit_label"] != "success"]
    failure_y = [point[1] for case, point in zip(cases, points) if case["hit_label"] != "success"]

    plt.figure(figsize=(11, 7), dpi=140)
    plt.scatter(success_x, success_y, c="#0b7a53", label="success", s=58, alpha=0.82, edgecolors="white", linewidths=0.6)
    plt.scatter(failure_x, failure_y, c="#b3261e", label="failure", s=58, alpha=0.82, edgecolors="white", linewidths=0.6)
    plt.title("TankChallenge Battle Case Embeddings PCA", fontsize=15, weight="bold")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    plt.grid(True, color="#d9e0ea", linewidth=0.8, alpha=0.8)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()

    return {
        "case_count": len(cases),
        "model_name": model_name,
        "explained_variance_ratio": [
            float(pca.explained_variance_ratio_[0]),
            float(pca.explained_variance_ratio_[1]),
        ],
        "output_png": output_png,
        "output_csv": output_csv,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PCA visualization for RAG embeddings")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-png", default=DEFAULT_PNG)
    parser.add_argument("--output-csv", default=DEFAULT_CSV)
    args = parser.parse_args()
    result = build_pca_visualization(
        model_name=args.model,
        output_png=args.output_png,
        output_csv=args.output_csv,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
