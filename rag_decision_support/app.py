"""Flask dashboard for TankChallenge RAG decision support."""

from __future__ import annotations

import csv
import json
import os
import urllib.request

from flask import Flask, jsonify, render_template, request

from tank_rag import (
    build_faiss_index,
    build_chroma_index,
    compare_backends,
    evaluate,
    read_index,
    recommend,
    search,
    search_chroma,
    search_faiss,
    summarize_matches,
    write_index,
)


app = Flask(__name__)
PCA_POINTS_CSV = os.path.join(os.path.dirname(__file__), "pca_embedding_points.csv")


def _float_arg(name: str, default: float | None = None) -> float | None:
    value = request.values.get(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _query_from_request() -> dict:
    return {
        "target_type": request.values.get("target_type", "moving_enemy"),
        "distance": _float_arg("distance", 85.0),
        "body_error": _float_arg("body_error", 3.0),
        "turret_error": _float_arg("turret_error", 0.8),
        "pitch_error": _float_arg("pitch_error", -0.1),
        "impact_error": _float_arg("impact_error"),
        "enemy_speed": _float_arg("enemy_speed", 0.4),
        "lead_distance": _float_arg("lead_distance", 0.0),
    }


def _run_query(query: dict, top_k: int = 5, backend: str = "faiss") -> dict:
    cases = read_index()
    if backend == "faiss":
        matches = search_faiss(query, top_k=top_k)
    elif backend == "chroma":
        matches = search_chroma(query, top_k=top_k)
    else:
        matches = search(cases, query, top_k=top_k)
    result = recommend(matches, query)
    return {
        "query": query,
        "backend": backend,
        "recommendation": result,
        "quality": summarize_matches(matches),
        "comparison": compare_backends(query, top_k=top_k),
        "matches": [
            {
                "score": round(score, 3),
                "case_id": case.case_id,
                "source": case.source,
                "target_type": case.target_type,
                "hit_label": case.hit_label,
                "distance": case.distance,
                "body_error": case.body_error,
                "turret_error": case.turret_error,
                "pitch_error": case.pitch_error,
                "impact_error": case.impact_error,
                "range_error": case.range_error,
                "enemy_speed": case.enemy_speed,
                "lead_distance": case.lead_distance,
                "document": case.document,
            }
            for score, case in matches
        ],
    }


@app.get("/")
def index():
    cases = read_index()
    metrics = evaluate(cases)
    query = {
        "target_type": "moving_enemy",
        "distance": 85.0,
        "body_error": 3.0,
        "turret_error": 0.8,
        "pitch_error": -0.1,
        "enemy_speed": 0.4,
        "lead_distance": 0.0,
    }
    result = _run_query(query, backend="faiss")
    return render_template("index.html", metrics=metrics, result=result)


@app.post("/query")
def query():
    top_k = int(_float_arg("top_k", 5) or 5)
    backend = request.values.get("backend", "faiss")
    result = _run_query(_query_from_request(), top_k=max(1, min(top_k, 12)), backend=backend)
    cases = read_index()
    metrics = evaluate(cases)
    return render_template("index.html", metrics=metrics, result=result)


@app.route("/api/query", methods=["GET", "POST"])
def api_query():
    top_k = int(_float_arg("top_k", 5) or 5)
    backend = request.values.get("backend", "faiss")
    return jsonify(_run_query(_query_from_request(), top_k=max(1, min(top_k, 12)), backend=backend))


@app.post("/api/rebuild")
def api_rebuild():
    cases = write_index()
    vector_result = build_faiss_index()
    chroma_result = build_chroma_index()
    return jsonify({"indexed_cases": len(cases), "faiss_index": vector_result, "chroma_index": chroma_result})


@app.get("/api/metrics")
def api_metrics():
    return jsonify(evaluate(read_index()))


@app.get("/api/pca-points")
def api_pca_points():
    points = []
    if os.path.exists(PCA_POINTS_CSV):
        with open(PCA_POINTS_CSV, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                points.append(
                    {
                        "case_id": row["case_id"],
                        "source": row["source"],
                        "hit_label": row["hit_label"],
                        "target_type": row["target_type"],
                        "distance": float(row["distance"] or 0),
                        "impact_error": float(row["impact_error"] or 0),
                        "pca_x": float(row["pca_x"]),
                        "pca_y": float(row["pca_y"]),
                    }
                )
    return jsonify({"points": points})


@app.get("/api/live-query")
def api_live_query():
    source = request.values.get("source", "http://127.0.0.1:5000/fire_status")
    try:
        with urllib.request.urlopen(source, timeout=2.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return jsonify({"error": repr(exc), "source": source}), 502

    query = {
        "target_type": request.values.get("target_type", "moving_enemy"),
        "distance": payload.get("distance") or payload.get("distance_fire") or payload.get("target_distance") or 85.0,
        "body_error": payload.get("body_error") or payload.get("body_error_fire") or 0.0,
        "turret_error": payload.get("turret_error") or payload.get("turret_error_fire") or 0.0,
        "pitch_error": payload.get("pitch_error") or payload.get("pitch_error_fire") or 0.0,
        "enemy_speed": payload.get("enemy_speed") or payload.get("enemy_speed_fire") or 0.0,
        "lead_distance": payload.get("lead_distance") or payload.get("lead_distance_fire") or 0.0,
    }
    top_k = int(_float_arg("top_k", 5) or 5)
    backend = request.values.get("backend", "faiss")
    return jsonify({"source": source, "raw": payload, "result": _run_query(query, top_k=top_k, backend=backend)})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5056, debug=False)
