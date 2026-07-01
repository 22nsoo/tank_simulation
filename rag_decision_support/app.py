# -*- coding: utf-8 -*-
"""Flask dashboard for TankChallenge RAG decision support."""

from __future__ import annotations

import csv
import json
import os
import urllib.request

from flask import Flask, jsonify, render_template, request

from drive_planner import plan_drive
from tactical_context import (
    DEFAULT_SOURCE_BASE,
    fetch_live_payloads,
    ingest_live_tactical_context,
    recommend_mobility_action,
    search_tactical_context,
    summarize_tactical_payloads,
)
from tank_rag_llm import (
    DEFAULT_LLM_PROVIDER,
    build_faiss_index,
    build_chroma_index,
    chat_with_llm,
    compare_backends,
    evaluate,
    paid_llm_enabled,
    read_index,
    recommend as rule_recommend,
    recommend_with_llm,
    search,
    search_chroma,
    search_faiss,
    summarize_matches,
    write_index,
)
from world_map import DEFAULT_MAP_FILE, list_map_files, parse_world_map


app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
app.json.ensure_ascii = False
PCA_POINTS_CSV = os.path.join(os.path.dirname(__file__), "pca_embedding_points.csv")
PAGE_TITLES = {
    "decision": "의사결정",
    "fire": "사격",
    "drive": "주행",
}


@app.after_request
def add_utf8_headers(response):
    if response.mimetype in {"text/html", "application/json"}:
        response.headers["Content-Type"] = f"{response.mimetype}; charset=utf-8"
    return response


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


def _normalize_recommendation(result: dict) -> dict:
    if "summary" not in result:
        result["summary"] = result.get("reason", "LLM decision completed.")
    return result


def _tactical_query_text(query: dict, message: str | None = None) -> str:
    parts = [
        f"target_type {query.get('target_type')}",
        f"distance {query.get('distance')}",
        f"body_error {query.get('body_error')}",
        f"turret_error {query.get('turret_error')}",
        f"pitch_error {query.get('pitch_error')}",
        f"enemy_speed {query.get('enemy_speed')}",
        f"lead_distance {query.get('lead_distance')}",
    ]
    if message:
        parts.append(message)
    return ", ".join(str(part) for part in parts)


def _live_tactical_contexts(query: dict, message: str | None = None, source_base: str | None = None) -> tuple[list[dict], dict]:
    ingest = ingest_live_tactical_context(source_base or DEFAULT_SOURCE_BASE)
    if ingest.get("status") != "success":
        return [], ingest
    contexts = search_tactical_context(_tactical_query_text(query, message), top_k=3)
    return contexts, ingest


def _load_world_map(map_file: str | None = None) -> dict:
    filename = map_file or DEFAULT_MAP_FILE
    try:
        return parse_world_map(filename)
    except Exception as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "filename": filename,
            "bounds": {"x_min": 0.0, "x_max": 300.0, "z_min": 0.0, "z_max": 300.0},
            "obstacles": [],
            "available_maps": list_map_files(),
        }


def _run_query(
    query: dict,
    top_k: int = 5,
    backend: str = "faiss",
    llm_provider: str | None = None,
    llm_model: str | None = None,
    source_base: str | None = None,
    use_llm: bool = True,
) -> dict:
    cases = read_index()
    if backend == "faiss":
        matches = search_faiss(query, top_k=top_k)
    elif backend == "chroma":
        matches = search_chroma(query, top_k=top_k)
    else:
        matches = search(cases, query, top_k=top_k)
    tactical_contexts, tactical_ingest = _live_tactical_contexts(query, source_base=source_base)
    if use_llm and paid_llm_enabled():
        result = _normalize_recommendation(
            recommend_with_llm(
                matches,
                query,
                provider=llm_provider or DEFAULT_LLM_PROVIDER,
                model_name=llm_model,
                tactical_contexts=tactical_contexts,
            )
        )
    else:
        result = _normalize_recommendation(rule_recommend(matches, query))
        result["decision_mode"] = "rule_free_mode" if use_llm else "rule_fast_page"
    action = recommend_mobility_action(
        tactical_ingest.get("summary") or {},
        shot_recommendation=result,
        query=query,
    )
    drive_plan = plan_drive(tactical_ingest.get("summary") or {}, action)
    world_map = _load_world_map(request.values.get("map_file"))
    return {
        "query": query,
        "backend": backend,
        "decision_mode": result.get("decision_mode", "llm"),
        "action": action,
        "drive_plan": drive_plan,
        "world_map": world_map,
        "tactical_context": {
            "ingest_status": tactical_ingest.get("status"),
            "summary": tactical_ingest.get("summary") or {},
            "contexts": tactical_contexts,
            "error": tactical_ingest.get("error"),
        },
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


def _retrieve_matches(query: dict, top_k: int, backend: str):
    cases = read_index()
    if backend == "faiss":
        return search_faiss(query, top_k=top_k)
    if backend == "chroma":
        return search_chroma(query, top_k=top_k)
    return search(cases, query, top_k=top_k)


def _default_query() -> dict:
    return {
        "target_type": "moving_enemy",
        "distance": 85.0,
        "body_error": 3.0,
        "turret_error": 0.8,
        "pitch_error": -0.1,
        "enemy_speed": 0.4,
        "lead_distance": 0.0,
    }


def _page_name(raw: str | None = None) -> str:
    value = str(raw or "decision").strip().lower()
    return value if value in PAGE_TITLES else "decision"


def _render_dashboard(page: str = "decision", result: dict | None = None):
    cases = read_index()
    metrics = evaluate(cases)
    if result is None:
        result = _run_query(_default_query(), backend="faiss", use_llm=False)
    page = _page_name(page)
    return render_template(
        "index.html",
        metrics=metrics,
        result=result,
        page=page,
        page_title=PAGE_TITLES[page],
        pages=PAGE_TITLES,
        paid_llm_enabled=paid_llm_enabled(),
    )


@app.get("/")
def index():
    return _render_dashboard("decision")


@app.get("/decision")
def decision_page():
    return _render_dashboard("decision")


@app.get("/fire")
def fire_page():
    return _render_dashboard("fire")


@app.get("/drive")
def drive_page():
    return _render_dashboard("drive")


@app.post("/query")
def query():
    top_k = int(_float_arg("top_k", 5) or 5)
    backend = request.values.get("backend", "faiss")
    page = _page_name(request.values.get("page"))
    result = _run_query(
        _query_from_request(),
        top_k=max(1, min(top_k, 12)),
        backend=backend,
        llm_provider=request.values.get("llm_provider") or DEFAULT_LLM_PROVIDER,
        llm_model=request.values.get("llm_model"),
        source_base=request.values.get("source_base"),
    )
    return _render_dashboard(page, result=result)


@app.route("/api/query", methods=["GET", "POST"])
def api_query():
    top_k = int(_float_arg("top_k", 5) or 5)
    backend = request.values.get("backend", "faiss")
    return jsonify(
        _run_query(
            _query_from_request(),
            top_k=max(1, min(top_k, 12)),
            backend=backend,
            llm_provider=request.values.get("llm_provider") or DEFAULT_LLM_PROVIDER,
            llm_model=request.values.get("llm_model"),
            source_base=request.values.get("source_base"),
        )
    )


@app.route("/api/chat", methods=["POST"])
def api_chat():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message") or request.values.get("message") or "").strip()
    if not message:
        return jsonify({"error": "message is required"}), 400

    top_k = int(payload.get("top_k") or _float_arg("top_k", 5) or 5)
    backend = str(payload.get("backend") or request.values.get("backend", "faiss"))
    query = {
        "target_type": payload.get("target_type", request.values.get("target_type", "moving_enemy")),
        "distance": payload.get("distance", _float_arg("distance", 85.0)),
        "body_error": payload.get("body_error", _float_arg("body_error", 3.0)),
        "turret_error": payload.get("turret_error", _float_arg("turret_error", 0.8)),
        "pitch_error": payload.get("pitch_error", _float_arg("pitch_error", -0.1)),
        "impact_error": payload.get("impact_error", _float_arg("impact_error")),
        "enemy_speed": payload.get("enemy_speed", _float_arg("enemy_speed", 0.4)),
        "lead_distance": payload.get("lead_distance", _float_arg("lead_distance", 0.0)),
    }
    matches = _retrieve_matches(query, top_k=max(1, min(top_k, 12)), backend=backend)
    source_base = payload.get("source_base") or request.values.get("source_base")
    tactical_contexts, tactical_ingest = _live_tactical_contexts(query, message=message, source_base=source_base)
    if paid_llm_enabled():
        answer = chat_with_llm(
            matches,
            query,
            message,
            provider=payload.get("llm_provider") or request.values.get("llm_provider") or DEFAULT_LLM_PROVIDER,
            model_name=payload.get("llm_model") or request.values.get("llm_model"),
            tactical_contexts=tactical_contexts,
        )
    else:
        base = rule_recommend(matches, query)
        summary = tactical_ingest.get("summary") or {}
        answer = {
            "answer": (
                "무료 모드라 외부 LLM API를 호출하지 않았습니다. "
                f"로컬 RAG 규칙 판단은 {'FIRE' if base.get('fire') else 'HOLD'}입니다. "
                f"{base.get('summary', '')} "
                f"실시간 context: 적 {summary.get('enemy_count', 'N/A')}개, "
                f"장애물 {summary.get('obstacle_count', 'N/A')}개, "
                f"전방 차단 {summary.get('front_blocked', 'N/A')}."
            ),
            "decision_mode": "rule_free_mode_chat",
            "evidence_case_ids": [case.case_id for _, case in matches[:3]],
            "tactical_context_ids": [item.get("id") for item in tactical_contexts],
        }
    return jsonify(
        {
            "query": query,
            "backend": backend,
            "message": message,
            "chat": answer,
            "tactical_context": {
                "ingest_status": tactical_ingest.get("status"),
                "summary": tactical_ingest.get("summary") or {},
                "contexts": tactical_contexts,
                "error": tactical_ingest.get("error"),
            },
            "matches": [
                {
                    "score": round(score, 3),
                    "case_id": case.case_id,
                    "hit_label": case.hit_label,
                    "distance": case.distance,
                    "impact_error": case.impact_error,
                }
                for score, case in matches
            ],
        }
    )


@app.route("/api/action", methods=["GET", "POST"])
def api_action():
    payload = request.get_json(silent=True) or {}
    top_k = int(payload.get("top_k") or _float_arg("top_k", 5) or 5)
    backend = str(payload.get("backend") or request.values.get("backend", "faiss"))
    query = {
        "target_type": payload.get("target_type", request.values.get("target_type", "moving_enemy")),
        "distance": payload.get("distance", _float_arg("distance", 85.0)),
        "body_error": payload.get("body_error", _float_arg("body_error", 3.0)),
        "turret_error": payload.get("turret_error", _float_arg("turret_error", 0.8)),
        "pitch_error": payload.get("pitch_error", _float_arg("pitch_error", -0.1)),
        "impact_error": payload.get("impact_error", _float_arg("impact_error")),
        "enemy_speed": payload.get("enemy_speed", _float_arg("enemy_speed", 0.4)),
        "lead_distance": payload.get("lead_distance", _float_arg("lead_distance", 0.0)),
    }
    source_base = payload.get("source_base") or request.values.get("source_base")
    matches = _retrieve_matches(query, top_k=max(1, min(top_k, 12)), backend=backend)
    tactical_contexts, tactical_ingest = _live_tactical_contexts(query, message="drive or fire action", source_base=source_base)
    if paid_llm_enabled():
        recommendation = _normalize_recommendation(
            recommend_with_llm(
                matches,
                query,
                provider=payload.get("llm_provider") or request.values.get("llm_provider") or DEFAULT_LLM_PROVIDER,
                model_name=payload.get("llm_model") or request.values.get("llm_model"),
                tactical_contexts=tactical_contexts,
            )
        )
    else:
        recommendation = _normalize_recommendation(rule_recommend(matches, query))
        recommendation["decision_mode"] = "rule_free_mode"
    action = recommend_mobility_action(
        tactical_ingest.get("summary") or {},
        shot_recommendation=recommendation,
        query=query,
    )
    drive_plan = plan_drive(tactical_ingest.get("summary") or {}, action)
    return jsonify(
        {
            "query": query,
            "backend": backend,
            "recommendation": recommendation,
            "action": action,
            "drive_plan": drive_plan,
            "tactical_context": {
                "ingest_status": tactical_ingest.get("status"),
                "summary": tactical_ingest.get("summary") or {},
                "contexts": tactical_contexts,
                "error": tactical_ingest.get("error"),
            },
            "matches": [
                {
                    "score": round(score, 3),
                    "case_id": case.case_id,
                    "hit_label": case.hit_label,
                    "distance": case.distance,
                    "impact_error": case.impact_error,
                }
                for score, case in matches
            ],
        }
    )


@app.route("/api/control/set-destination", methods=["POST"])
def api_control_set_destination():
    payload = request.get_json(silent=True) or {}
    destination = payload.get("destination") if isinstance(payload.get("destination"), dict) else payload
    source_base = str(payload.get("source_base") or request.values.get("source_base") or DEFAULT_SOURCE_BASE).rstrip("/")
    if not isinstance(destination, dict) or destination.get("x") is None or destination.get("z") is None:
        return jsonify({"status": "error", "message": "destination requires x and z"}), 400
    body = json.dumps(destination).encode("utf-8")
    req = urllib.request.Request(
        f"{source_base}/set_destination",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=3.0) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return jsonify({"status": "error", "message": repr(exc), "source_base": source_base}), 502
    return jsonify({"status": "success", "source_base": source_base, "control": data})


@app.route("/api/tactical/ingest", methods=["GET", "POST"])
def api_tactical_ingest():
    source_base = request.values.get("source_base", DEFAULT_SOURCE_BASE)
    return jsonify(ingest_live_tactical_context(source_base=source_base))


@app.route("/api/tactical/live", methods=["GET", "POST"])
def api_tactical_live():
    source_base = request.values.get("source_base", DEFAULT_SOURCE_BASE)
    live = fetch_live_payloads(source_base=source_base)
    summary = summarize_tactical_payloads(live)
    status = "success" if summary.get("available_payloads") else "unavailable"
    error = None if summary.get("available_payloads") else "no live LiDAR/YOLO payloads were reachable"
    action = recommend_mobility_action(summary)
    drive_plan = plan_drive(summary, action)
    return jsonify(
        {
            "ingest_status": status,
            "error": error,
            "summary": summary,
            "action": action,
            "drive_plan": drive_plan,
        }
    )


@app.route("/api/tactical/search", methods=["GET", "POST"])
def api_tactical_search():
    text = request.values.get("q") or request.values.get("message") or "current live tactical context"
    top_k = int(_float_arg("top_k", 3) or 3)
    return jsonify({"query": text, "contexts": search_tactical_context(text, top_k=max(1, min(top_k, 10)))})


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


@app.get("/api/world-map")
def api_world_map():
    map_file = request.values.get("map_file") or DEFAULT_MAP_FILE
    try:
        return jsonify(parse_world_map(map_file))
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {exc}", "filename": map_file}), 400


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
    return jsonify(
        {
            "source": source,
            "raw": payload,
            "result": _run_query(
                query,
                top_k=top_k,
                backend=backend,
                llm_provider=request.values.get("llm_provider") or DEFAULT_LLM_PROVIDER,
                llm_model=request.values.get("llm_model"),
                source_base=request.values.get("source_base"),
            ),
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5056, debug=False)
