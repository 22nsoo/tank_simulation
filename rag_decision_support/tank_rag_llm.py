# -*- coding: utf-8 -*-
"""Local RAG-style decision support for TankChallenge shot logs.

This module intentionally uses only the Python standard library so the
portfolio demo can run on the current project without extra installs.
It converts shot CSV rows into readable case documents, retrieves similar
past cases with a hybrid numeric/text score, and produces aim/fire advice.

LLM mode can optionally replace the rule-based final decision step: retrieved
Top-k cases are passed to an LLM, which returns a JSON FIRE/HOLD decision.
If the LLM is unavailable or returns invalid JSON, the rule-based recommender
is used as a safety fallback.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from statistics import mean
from typing import Iterable


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_INDEX = os.path.join(ROOT, "rag_decision_support", "case_index.jsonl")
DEFAULT_REPORT = os.path.join(ROOT, "rag_decision_support", "rag_report.md")
DEFAULT_FAISS_INDEX = os.path.join(ROOT, "rag_decision_support", "faiss_cases.index")
DEFAULT_FAISS_META = os.path.join(ROOT, "rag_decision_support", "faiss_cases_meta.json")
DEFAULT_CHROMA_DIR = os.path.join(ROOT, "rag_decision_support", "chroma_store")
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_PROVIDER = os.getenv("TANK_RAG_LLM_PROVIDER", "llama").strip().lower()
DEFAULT_GEMINI_MODEL = os.getenv("TANK_RAG_GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_LLAMA_MODEL = os.getenv("TANK_RAG_LLAMA_MODEL", "llama3.1-70b")
DEFAULT_LLAMA_BASE_URL = os.getenv("LLAMA_API_BASE_URL", "https://api.llama-api.com").rstrip("/")
_MODEL_CACHE = {}
_FAISS_CACHE = {}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def paid_llm_enabled() -> bool:
    """Return whether external LLM calls are allowed."""
    return _env_bool("TANK_RAG_ALLOW_PAID_LLM", True)


NUMERIC_FEATURES = (
    "distance",
    "body_error",
    "turret_error",
    "pitch_error",
    "impact_error",
    "enemy_speed",
    "lead_distance",
)


@dataclass
class ShotCase:
    case_id: str
    source: str
    target_type: str
    distance: float | None
    body_error: float | None
    turret_error: float | None
    pitch_error: float | None
    impact_error: float | None
    range_error: float | None
    enemy_speed: float | None
    lead_distance: float | None
    hit_label: str
    action: str
    document: str

    def to_json(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_json(cls, data: dict) -> "ShotCase":
        return cls(**data)


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _first_float(row: dict, names: Iterable[str]) -> float | None:
    for name in names:
        value = _to_float(row.get(name))
        if value is not None:
            return value
    return None


def _is_success(hit: str, impact_error: float | None) -> bool:
    text = str(hit or "").strip().lower()
    if text in {"enemy", "tank", "hit", "true", "1", "success"}:
        return True
    if impact_error is not None and impact_error <= 2.0:
        return True
    return False


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_가-힣]+", text.lower()))


def _case_from_row(row: dict, source: str, row_index: int) -> ShotCase | None:
    distance = _first_float(row, ("distance_fire", "target_distance", "ballistic_R_fire"))
    body_error = _first_float(row, ("body_error_fire",))
    turret_error = _first_float(row, ("turret_error_fire",))
    pitch_error = _first_float(row, ("pitch_error_fire",))
    impact_error = _first_float(row, ("impact_error_to_enemy_fire",))
    range_error = _first_float(row, ("range_error_fire",))
    enemy_speed = _first_float(row, ("enemy_speed_fire",))
    lead_distance = _first_float(row, ("lead_distance_fire",))
    hit = str(row.get("hit", "")).strip() or "unknown"
    target_type = str(row.get("target_type", "")).strip() or "stationary"

    if distance is None or impact_error is None:
        return None
    if distance < 5.0:
        return None
    if pitch_error is not None and abs(pitch_error) > 45.0:
        return None
    if turret_error is not None and abs(turret_error) > 45.0:
        return None
    if impact_error > 100.0:
        return None

    success = _is_success(hit, impact_error)
    hit_label = "success" if success else "failure"
    action = "fire_hit" if success else "fire_miss_or_terrain"
    case_id = f"{os.path.basename(source)}:{row.get('shot_id') or row_index}"

    moving = enemy_speed is not None and enemy_speed > 0.05
    los_text = "moving target" if moving else "stationary target"
    document = (
        f"[Situation] {los_text}, distance {distance:.1f}m, "
        f"body yaw error {body_error if body_error is not None else 0:.2f}deg, "
        f"turret yaw error {turret_error if turret_error is not None else 0:.2f}deg, "
        f"pitch error {pitch_error if pitch_error is not None else 0:.2f}deg. "
        f"Enemy speed {enemy_speed if enemy_speed is not None else 0:.2f}m/s, "
        f"lead distance {lead_distance if lead_distance is not None else 0:.2f}m. "
        f"[Action] Fired at target. [Result] {hit_label}; hit field '{hit}', "
        f"impact error {impact_error:.2f}m, range error "
        f"{range_error if range_error is not None else 0:.2f}m."
    )

    return ShotCase(
        case_id=case_id,
        source=os.path.relpath(source, ROOT),
        target_type=target_type,
        distance=distance,
        body_error=body_error,
        turret_error=turret_error,
        pitch_error=pitch_error,
        impact_error=impact_error,
        range_error=range_error,
        enemy_speed=enemy_speed,
        lead_distance=lead_distance,
        hit_label=hit_label,
        action=action,
        document=document,
    )


def load_cases() -> list[ShotCase]:
    patterns = [
        os.path.join(ROOT, "shot_analysis", "shot_log_*.csv"),
        os.path.join(ROOT, "shot_analysis", "moving_target_logs", "moving_shot_log_*.csv"),
    ]
    cases: list[ShotCase] = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path, newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                for index, row in enumerate(reader, start=1):
                    case = _case_from_row(row, path, index)
                    if case:
                        cases.append(case)
    return cases


def write_index(path: str = DEFAULT_INDEX) -> list[ShotCase]:
    cases = load_cases()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            handle.write(json.dumps(case.to_json(), ensure_ascii=False) + "\n")
    return cases


def read_index(path: str = DEFAULT_INDEX) -> list[ShotCase]:
    if not os.path.exists(path):
        return write_index(path)
    cases: list[ShotCase] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                cases.append(ShotCase.from_json(json.loads(line)))
    return cases


def _query_document(query: dict) -> str:
    parts = ["[Situation]"]
    if query.get("target_type"):
        parts.append(str(query["target_type"]).replace("_", " "))
    if query.get("distance") is not None:
        parts.append(f"distance {float(query['distance']):.1f}m")
    if query.get("body_error") is not None:
        parts.append(f"body yaw error {float(query['body_error']):.2f}deg")
    if query.get("turret_error") is not None:
        parts.append(f"turret yaw error {float(query['turret_error']):.2f}deg")
    if query.get("pitch_error") is not None:
        parts.append(f"pitch error {float(query['pitch_error']):.2f}deg")
    if query.get("enemy_speed") is not None:
        parts.append(f"enemy speed {float(query['enemy_speed']):.2f}m/s")
    if query.get("lead_distance") is not None:
        parts.append(f"lead distance {float(query['lead_distance']):.2f}m")
    return ", ".join(parts)


def _numeric_similarity(query: dict, case: ShotCase) -> float:
    scales = {
        "distance": 120.0,
        "body_error": 45.0,
        "turret_error": 20.0,
        "pitch_error": 5.0,
        "impact_error": 8.0,
        "enemy_speed": 4.0,
        "lead_distance": 8.0,
    }
    scores: list[float] = []
    for key in NUMERIC_FEATURES:
        q_value = query.get(key)
        c_value = getattr(case, key)
        if q_value is None or c_value is None:
            continue
        diff = abs(float(q_value) - float(c_value))
        scores.append(max(0.0, 1.0 - diff / scales[key]))
    return mean(scores) if scores else 0.0


def search(cases: list[ShotCase], query: dict, top_k: int = 5) -> list[tuple[float, ShotCase]]:
    query_tokens = _tokenize(_query_document(query))
    query_target_type = str(query.get("target_type") or "").strip().lower()
    scored: list[tuple[float, ShotCase]] = []
    for case in cases:
        case_tokens = _tokenize(case.document)
        text_score = 0.0
        if query_tokens and case_tokens:
            text_score = len(query_tokens & case_tokens) / len(query_tokens | case_tokens)
        numeric_score = _numeric_similarity(query, case)
        target_score = 0.0
        if query_target_type:
            case_target_type = str(case.target_type or "").strip().lower()
            if query_target_type == case_target_type:
                target_score = 1.0
            elif "moving" in query_target_type and "moving" in case_target_type:
                target_score = 1.0
            elif "stationary" in query_target_type and "stationary" in case_target_type:
                target_score = 1.0
        score = 0.68 * numeric_score + 0.17 * text_score + 0.15 * target_score
        scored.append((score, case))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k]


def _distance_bucket(distance: float | None) -> str:
    if distance is None:
        return "unknown"
    if distance < 70:
        return "close"
    if distance < 105:
        return "mid"
    return "far"


def _target_mode(target_type: str | None, enemy_speed: float | None = None) -> str:
    text = str(target_type or "").lower()
    if "moving" in text:
        return "moving"
    if enemy_speed is not None and enemy_speed > 0.05:
        return "moving"
    return "stationary"


def _weighted_average(values: list[tuple[float, float]]) -> float | None:
    total_weight = sum(max(weight, 0.0) for weight, _ in values)
    if total_weight <= 0:
        return None
    return sum(max(weight, 0.0) * value for weight, value in values) / total_weight


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _rerank_vector_matches(
    matches: list[tuple[float, ShotCase]],
    query: dict,
    top_k: int,
) -> list[tuple[float, ShotCase]]:
    query_mode = _target_mode(query.get("target_type"), _to_float(query.get("enemy_speed")))
    query_bucket = _distance_bucket(_to_float(query.get("distance")))
    reranked: list[tuple[float, ShotCase]] = []
    for score, case in matches:
        case_mode = _target_mode(case.target_type, case.enemy_speed)
        case_bucket = _distance_bucket(case.distance)
        adjusted = score
        if case_mode == query_mode:
            adjusted += 0.08
        else:
            adjusted -= 0.10
        if case_bucket == query_bucket:
            adjusted += 0.06
        elif query_bucket != "unknown" and case_bucket != "unknown":
            adjusted -= 0.03
        reranked.append((adjusted, case))
    reranked.sort(key=lambda item: item[0], reverse=True)
    return reranked[:top_k]


def _load_embedding_dependencies():
    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Embedding search requires sentence-transformers and faiss-cpu. "
            "Install with: python -m pip install sentence-transformers faiss-cpu"
        ) from exc
    return faiss, np, SentenceTransformer


def _normalize_embeddings(np_module, embeddings):
    vectors = np_module.asarray(embeddings, dtype="float32")
    norms = np_module.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def build_faiss_index(
    model_name: str = DEFAULT_MODEL_NAME,
    index_path: str = DEFAULT_FAISS_INDEX,
    meta_path: str = DEFAULT_FAISS_META,
) -> dict:
    faiss, np_module, SentenceTransformer = _load_embedding_dependencies()
    cases = write_index()
    model = SentenceTransformer(model_name)
    documents = [case.document for case in cases]
    embeddings = model.encode(documents, batch_size=32, show_progress_bar=True)
    vectors = _normalize_embeddings(np_module, embeddings)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, index_path)
    metadata = {
        "model_name": model_name,
        "case_count": len(cases),
        "dimension": int(vectors.shape[1]),
        "cases": [case.to_json() for case in cases],
    }
    with open(meta_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    _FAISS_CACHE.clear()
    return {
        "model_name": model_name,
        "case_count": len(cases),
        "dimension": int(vectors.shape[1]),
        "index_path": index_path,
        "meta_path": meta_path,
    }


def build_chroma_index(
    model_name: str = DEFAULT_MODEL_NAME,
    persist_dir: str = DEFAULT_CHROMA_DIR,
    collection_name: str = "tank_battle_cases",
) -> dict:
    try:
        import chromadb
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Chroma search requires chromadb and sentence-transformers. "
            "Install with: python -m pip install chromadb sentence-transformers"
        ) from exc

    cases = write_index()
    model = SentenceTransformer(model_name)
    embeddings = model.encode([case.document for case in cases], batch_size=32, show_progress_bar=True)
    vectors = np.asarray(embeddings, dtype="float32").tolist()
    client = chromadb.PersistentClient(path=persist_dir)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.get_or_create_collection(collection_name)
    collection.add(
        ids=[case.case_id for case in cases],
        documents=[case.document for case in cases],
        embeddings=vectors,
        metadatas=[
            {
                "source": case.source,
                "target_type": case.target_type,
                "distance": case.distance or 0.0,
                "impact_error": case.impact_error or 0.0,
                "hit_label": case.hit_label,
            }
            for case in cases
        ],
    )
    with open(os.path.join(persist_dir, "case_payloads.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            {
                "model_name": model_name,
                "collection_name": collection_name,
                "cases": [case.to_json() for case in cases],
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    return {
        "model_name": model_name,
        "case_count": len(cases),
        "persist_dir": persist_dir,
        "collection_name": collection_name,
    }


def search_faiss(
    query: dict,
    top_k: int = 5,
    model_name: str | None = None,
    index_path: str = DEFAULT_FAISS_INDEX,
    meta_path: str = DEFAULT_FAISS_META,
) -> list[tuple[float, ShotCase]]:
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        build_faiss_index(model_name=model_name or DEFAULT_MODEL_NAME, index_path=index_path, meta_path=meta_path)
    faiss, np_module, SentenceTransformer = _load_embedding_dependencies()
    cache_key = (os.path.abspath(index_path), os.path.abspath(meta_path))
    if cache_key in _FAISS_CACHE:
        index, metadata, cases = _FAISS_CACHE[cache_key]
    else:
        index = faiss.read_index(index_path)
        with open(meta_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
        cases = [ShotCase.from_json(case_data) for case_data in metadata["cases"]]
        _FAISS_CACHE[cache_key] = (index, metadata, cases)
    resolved_model = model_name or metadata.get("model_name") or DEFAULT_MODEL_NAME
    if resolved_model in _MODEL_CACHE:
        model = _MODEL_CACHE[resolved_model]
    else:
        model = SentenceTransformer(resolved_model)
        _MODEL_CACHE[resolved_model] = model
    query_embedding = model.encode([_query_document(query)], show_progress_bar=False)
    query_vector = _normalize_embeddings(np_module, query_embedding)
    search_k = max(top_k * 6, top_k)
    scores, indices = index.search(query_vector, search_k)
    matches: list[tuple[float, ShotCase]] = []
    for score, case_index in zip(scores[0], indices[0]):
        if 0 <= int(case_index) < len(cases):
            matches.append((float(score), cases[int(case_index)]))
    return _rerank_vector_matches(matches, query, top_k)


def search_chroma(
    query: dict,
    top_k: int = 5,
    model_name: str | None = None,
    persist_dir: str = DEFAULT_CHROMA_DIR,
    collection_name: str = "tank_battle_cases",
) -> list[tuple[float, ShotCase]]:
    payload_path = os.path.join(persist_dir, "case_payloads.json")
    if not os.path.exists(payload_path):
        build_chroma_index(model_name=model_name or DEFAULT_MODEL_NAME, persist_dir=persist_dir, collection_name=collection_name)
    try:
        import chromadb
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Chroma search requires chromadb and sentence-transformers. "
            "Install with: python -m pip install chromadb sentence-transformers"
        ) from exc

    with open(payload_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    resolved_model = model_name or payload.get("model_name") or DEFAULT_MODEL_NAME
    if resolved_model in _MODEL_CACHE:
        model = _MODEL_CACHE[resolved_model]
    else:
        model = SentenceTransformer(resolved_model)
        _MODEL_CACHE[resolved_model] = model
    query_embedding = np.asarray(model.encode([_query_document(query)], show_progress_bar=False), dtype="float32")[0].tolist()
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection(collection_name)
    search_k = max(top_k * 6, top_k)
    result = collection.query(query_embeddings=[query_embedding], n_results=search_k)
    cases_by_id = {case_data["case_id"]: ShotCase.from_json(case_data) for case_data in payload["cases"]}
    matches: list[tuple[float, ShotCase]] = []
    ids = result.get("ids", [[]])[0]
    distances = result.get("distances", [[]])[0]
    for case_id, distance in zip(ids, distances):
        case = cases_by_id.get(case_id)
        if case:
            score = 1.0 / (1.0 + float(distance))
            matches.append((score, case))
    return _rerank_vector_matches(matches, query, top_k)


def recommend(matches: list[tuple[float, ShotCase]], query: dict) -> dict:
    if not matches:
        return {
            "fire": False,
            "confidence": 0.0,
            "summary": "No comparable cases were found.",
            "yaw_correction_deg": 0.0,
            "pitch_correction_deg": 0.0,
            "decision_threshold": 0.0,
            "distance_bucket": _distance_bucket(query.get("distance")),
            "target_mode": _target_mode(query.get("target_type"), query.get("enemy_speed")),
            "reason_codes": ["NO_MATCHES"],
        }

    weighted_total = sum(score for score, _ in matches) or 1.0
    weighted_success = sum(score for score, case in matches if case.hit_label == "success")
    success_rate = weighted_success / weighted_total
    miss_matches = [(score, case) for score, case in matches if case.hit_label != "success"]
    success_matches = [(score, case) for score, case in matches if case.hit_label == "success"]

    query_turret_signed = float(query.get("turret_error") or 0.0)
    query_turret_error = abs(query_turret_signed)
    query_pitch_error = float(query.get("pitch_error") or 0.0)
    query_body_error = abs(float(query.get("body_error") or 0.0))
    query_distance = _to_float(query.get("distance"))
    distance_bucket = _distance_bucket(query_distance)
    target_mode = _target_mode(query.get("target_type"), _to_float(query.get("enemy_speed")))

    same_bucket_success = [
        (score, case)
        for score, case in success_matches
        if _distance_bucket(case.distance) == distance_bucket and _target_mode(case.target_type, case.enemy_speed) == target_mode
    ]
    same_mode_success = [
        (score, case)
        for score, case in success_matches
        if _target_mode(case.target_type, case.enemy_speed) == target_mode
    ]
    preferred_success = same_bucket_success or same_mode_success or success_matches

    yaw_correction = 0.0
    pitch_correction = 0.0
    if preferred_success:
        success_yaw_target = _weighted_average(
            [(score, case.turret_error or 0.0) for score, case in preferred_success]
        )
        success_pitch_target = _weighted_average(
            [(score, case.pitch_error or 0.0) for score, case in preferred_success]
        )
        if success_yaw_target is not None:
            yaw_correction = success_yaw_target - query_turret_signed
        if success_pitch_target is not None:
            pitch_correction = success_pitch_target - query_pitch_error

    if miss_matches:
        miss_yaw = _weighted_average([(score, case.turret_error or 0.0) for score, case in miss_matches])
        miss_pitch = _weighted_average([(score, case.pitch_error or 0.0) for score, case in miss_matches])
        miss_weight = min(0.45, sum(score for score, _ in miss_matches) / weighted_total)
        if miss_yaw is not None:
            yaw_correction += (query_turret_signed - miss_yaw) * miss_weight
        if miss_pitch is not None:
            pitch_correction += (query_pitch_error - miss_pitch) * miss_weight

    if not preferred_success and miss_matches:
        yaw_correction = -_clip(query_turret_signed, -1.5, 1.5)
        pitch_correction = -query_pitch_error * 0.5

    bucket_pitch_bias = {
        "close": 0.02,
        "mid": 0.05,
        "far": 0.10,
        "unknown": 0.0,
    }[distance_bucket]
    if target_mode == "moving":
        bucket_pitch_bias += 0.03
    pitch_correction += bucket_pitch_bias
    yaw_correction = _clip(yaw_correction, -2.5, 2.5)
    pitch_correction = _clip(pitch_correction, -1.0, 1.0)

    threshold_table = {
        ("stationary", "close"): 0.52,
        ("stationary", "mid"): 0.57,
        ("stationary", "far"): 0.62,
        ("moving", "close"): 0.58,
        ("moving", "mid"): 0.64,
        ("moving", "far"): 0.70,
        ("stationary", "unknown"): 0.60,
        ("moving", "unknown"): 0.66,
    }
    threshold = threshold_table.get((target_mode, distance_bucket), 0.62)
    if query_turret_error > 1.0:
        threshold += 0.05
    if abs(query_pitch_error) > 0.25:
        threshold += 0.05
    if query_body_error > 20.0:
        threshold += 0.04
    threshold = _clip(threshold, 0.50, 0.85)

    reason_codes = []
    risk_notes = []
    if query_turret_error > 1.0:
        risk_notes.append("turret yaw error is above the stable-fire band")
        reason_codes.append("YAW_RISK")
    if abs(query_pitch_error) > 0.25:
        risk_notes.append("pitch error is still meaningful")
        reason_codes.append("PITCH_RISK")
    if query_body_error > 20.0:
        risk_notes.append("body yaw is far from target direction")
        reason_codes.append("BODY_ALIGNMENT_RISK")
    if same_bucket_success:
        reason_codes.append("SAME_BUCKET_SUCCESS_WEIGHTED")
    elif same_mode_success:
        reason_codes.append("SAME_MODE_SUCCESS_WEIGHTED")
    if miss_matches:
        reason_codes.append("FAILURE_DIRECTION_PENALTY")

    fire = success_rate >= threshold and query_turret_error <= 1.2 and abs(query_pitch_error) <= 0.35
    if risk_notes and success_rate < max(0.75, threshold + 0.08):
        fire = False

    summary = (
        f"Similar-case success rate is {success_rate:.0%} "
        f"(threshold {threshold:.0%}, {target_mode}/{distance_bucket}). "
        f"Recommendation: {'fire' if fire else 'hold aim / correct first'}."
    )
    if risk_notes:
        summary += " Main risks: " + "; ".join(risk_notes) + "."

    return {
        "fire": fire,
        "confidence": round(success_rate, 3),
        "summary": summary,
        "yaw_correction_deg": round(yaw_correction, 3),
        "pitch_correction_deg": round(pitch_correction, 3),
        "decision_threshold": round(threshold, 3),
        "distance_bucket": distance_bucket,
        "target_mode": target_mode,
        "reason_codes": reason_codes,
    }


def _format_tactical_contexts(tactical_contexts: list[dict] | None) -> str:
    if not tactical_contexts:
        return "No live tactical context was retrieved."
    blocks = []
    for index, item in enumerate(tactical_contexts, start=1):
        blocks.append(
            "\n".join(
                [
                    f"Tactical Context {index}",
                    f"id: {item.get('id')}",
                    f"score: {float(item.get('score', 0.0)):.3f}",
                    str(item.get("document") or ""),
                ]
            )
        )
    return "\n\n".join(blocks)


def build_llm_prompt(
    matches: list[tuple[float, ShotCase]],
    query: dict,
    tactical_contexts: list[dict] | None = None,
) -> str:
    """Build the RAG prompt used by LLM decision mode.

    The LLM does not see the full CSV. It receives only the current query and
    the retrieved Top-k historical cases, so the answer remains grounded in
    available evidence.
    """
    case_blocks: list[str] = []
    for rank, (score, case) in enumerate(matches, start=1):
        case_blocks.append(
            "\n".join(
                [
                    f"Case {rank}",
                    f"score: {score:.3f}",
                    f"case_id: {case.case_id}",
                    f"source: {case.source}",
                    f"target_type: {case.target_type}",
                    f"distance: {case.distance}",
                    f"body_error: {case.body_error}",
                    f"turret_error: {case.turret_error}",
                    f"pitch_error: {case.pitch_error}",
                    f"enemy_speed: {case.enemy_speed}",
                    f"lead_distance: {case.lead_distance}",
                    f"impact_error: {case.impact_error}",
                    f"range_error: {case.range_error}",
                    f"result: {case.hit_label}",
                    f"document: {case.document}",
                ]
            )
        )

    return f"""
You are a strict fire-control decision support module for a tank simulator.
Use only the retrieved historical cases as evidence. Do not invent case IDs.

Current situation:
- target_type: {query.get('target_type')}
- distance: {query.get('distance')} m
- body_yaw_error: {query.get('body_error')} deg
- turret_yaw_error: {query.get('turret_error')} deg
- pitch_error: {query.get('pitch_error')} deg
- impact_error: {query.get('impact_error')} m
- enemy_speed: {query.get('enemy_speed')} m/s
- lead_distance: {query.get('lead_distance')} m

Retrieved similar historical shot cases:
{chr(10).join(case_blocks)}

Retrieved live LiDAR/YOLO tactical context:
{_format_tactical_contexts(tactical_contexts)}

Decision policy:
- Recommend FIRE only when similar success cases dominate and current yaw/pitch errors are inside a stable band.
- Prefer HOLD when similar failure cases dominate, turret yaw error is large, pitch error is large, or body alignment risk is high.
- Prefer HOLD or reposition when live tactical context says front_blocked is true or an obstacle is close to the firing lane.
- Use live tactical context to answer enemy count, nearest enemy, obstacle, and line-of-fire questions.
- Suggest small yaw/pitch corrections using the successful cases and avoid repeated miss patterns.
- The output must be conservative. Safety and repeatability are more important than aggressive firing.

Return only valid JSON with exactly this schema:
{{
  "fire": true,
  "confidence": 0.0,
  "decision": "FIRE",
  "yaw_correction_deg": 0.0,
  "pitch_correction_deg": 0.0,
  "reason": "short evidence-based explanation",
  "evidence_case_ids": ["case_id"],
  "risk_notes": ["risk note"]
}}
""".strip()


def _extract_json_object(text: str) -> dict:
    """Parse a JSON object from an LLM response."""
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(stripped[start : end + 1])


def _normalize_llm_recommendation(data: dict, matches: list[tuple[float, ShotCase]]) -> dict:
    """Normalize and constrain the LLM JSON for downstream UI/API use."""
    valid_case_ids = {case.case_id for _, case in matches}
    evidence = data.get("evidence_case_ids") or []
    if not isinstance(evidence, list):
        evidence = []
    evidence = [str(case_id) for case_id in evidence if str(case_id) in valid_case_ids]

    fire = bool(data.get("fire", False))
    decision = str(data.get("decision") or ("FIRE" if fire else "HOLD")).upper()
    if decision not in {"FIRE", "HOLD"}:
        decision = "FIRE" if fire else "HOLD"
    fire = decision == "FIRE"

    def as_float(value: object, default: float = 0.0) -> float:
        parsed = _to_float(value)
        return default if parsed is None else parsed

    confidence = _clip(as_float(data.get("confidence"), 0.0), 0.0, 1.0)
    yaw_correction = _clip(as_float(data.get("yaw_correction_deg"), 0.0), -2.5, 2.5)
    pitch_correction = _clip(as_float(data.get("pitch_correction_deg"), 0.0), -1.0, 1.0)

    risk_notes = data.get("risk_notes") or []
    if not isinstance(risk_notes, list):
        risk_notes = [str(risk_notes)]

    return {
        "fire": fire,
        "confidence": round(confidence, 3),
        "decision": decision,
        "yaw_correction_deg": round(yaw_correction, 3),
        "pitch_correction_deg": round(pitch_correction, 3),
        "reason": str(data.get("reason") or "LLM decision completed."),
        "evidence_case_ids": evidence,
        "risk_notes": [str(note) for note in risk_notes],
        "decision_mode": "llm",
    }


def _llm_fallback(matches: list[tuple[float, ShotCase]], query: dict, reason: str) -> dict:
    """Use the deterministic recommender when LLM mode cannot complete."""
    base = recommend(matches, query)
    safe_reason = _safe_llm_error(reason)
    return {
        "fire": base["fire"],
        "confidence": base["confidence"],
        "decision": "FIRE" if base["fire"] else "HOLD",
        "yaw_correction_deg": base["yaw_correction_deg"],
        "pitch_correction_deg": base["pitch_correction_deg"],
        "reason": f"LLM fallback: {safe_reason}. {base['summary']}",
        "evidence_case_ids": [case.case_id for _, case in matches[:3]],
        "risk_notes": base.get("reason_codes", []),
        "decision_threshold": base.get("decision_threshold"),
        "distance_bucket": base.get("distance_bucket"),
        "target_mode": base.get("target_mode"),
        "decision_mode": "rule_fallback",
    }


def _safe_llm_error(reason: str) -> str:
    text = str(reason or "").strip()
    if "api key" in text.lower() or "invalid_api_key" in text.lower() or "401" in text:
        return "LLM provider authentication failed"
    if len(text) > 180:
        return text[:177] + "..."
    return text or "LLM provider call failed"


def _call_gemini_llm(prompt: str, model_name: str | None = None) -> str:
    """Call Google Gemini and return raw text."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not set")

    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise RuntimeError(
            "google-generativeai package is not installed. "
            "Install with: python -m pip install google-generativeai"
        ) from exc

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name or DEFAULT_GEMINI_MODEL,
        generation_config={
            "temperature": 0.1,
            "response_mime_type": "application/json",
        },
    )
    response = model.generate_content(prompt)
    return response.text or ""


def _call_gemini_text_llm(prompt: str, model_name: str | None = None) -> str:
    """Call Google Gemini for natural-language chatbot answers."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is not set")

    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise RuntimeError(
            "google-generativeai package is not installed. "
            "Install with: python -m pip install google-generativeai"
        ) from exc

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name or DEFAULT_GEMINI_MODEL,
        generation_config={"temperature": 0.2},
    )
    response = model.generate_content(prompt)
    return response.text or ""


def _call_llama_llm(prompt: str, model_name: str | None = None, *, json_mode: bool = False) -> str:
    """Call an OpenAI-compatible Llama API chat-completions endpoint."""
    api_key = os.getenv("LLAMA_API_KEY")
    if not api_key:
        raise RuntimeError("LLAMA_API_KEY is not set")

    base_url = os.getenv("LLAMA_API_BASE_URL", DEFAULT_LLAMA_BASE_URL).rstrip("/")
    url = f"{base_url}/chat/completions"
    payload: dict = {
        "model": model_name or DEFAULT_LLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1 if json_mode else 0.2,
    }
    if json_mode and os.getenv("TANK_RAG_LLAMA_RESPONSE_FORMAT", "").lower() in {"1", "true", "yes"}:
        payload["response_format"] = {"type": "json_object"}

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "TankChallenge-RAG/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(os.getenv("LLAMA_API_TIMEOUT_SEC", "30"))) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Llama API HTTP {exc.code}: {detail[:300]}") from exc

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("Llama API returned no choices")
    message = choices[0].get("message") or {}
    text = message.get("content")
    if isinstance(text, list):
        text = "".join(str(item.get("text", item)) for item in text)
    return str(text or "").strip()


def _select_llm_provider(provider: str | None) -> str:
    selected = str(provider or DEFAULT_LLM_PROVIDER or "llama").strip().lower()
    if selected in {"google", "gemini"}:
        return "gemini"
    if selected in {"llama", "llama_api", "llama-api"}:
        return "llama"
    raise ValueError(f"Unsupported LLM provider: {selected}")


def _provider_model_name(provider: str, model_name: str | None = None) -> str:
    if model_name:
        return model_name
    return DEFAULT_GEMINI_MODEL if provider == "gemini" else DEFAULT_LLAMA_MODEL


def _call_provider_llm(provider: str, prompt: str, model_name: str | None = None, *, json_mode: bool = False) -> str:
    if not paid_llm_enabled():
        raise RuntimeError("Paid external LLM calls are disabled. Set TANK_RAG_ALLOW_PAID_LLM=true to enable them.")
    if provider == "gemini":
        return _call_gemini_llm(prompt, model_name=model_name) if json_mode else _call_gemini_text_llm(prompt, model_name=model_name)
    if provider == "llama":
        return _call_llama_llm(prompt, model_name=model_name, json_mode=json_mode)
    raise ValueError(f"Unsupported LLM provider: {provider}")


def recommend_with_llm(
    matches: list[tuple[float, ShotCase]],
    query: dict,
    model_name: str | None = None,
    provider: str | None = None,
    tactical_contexts: list[dict] | None = None,
) -> dict:
    """Return FIRE/HOLD advice using retrieved cases plus an LLM.

    Supported provider:
    - gemini: requires GEMINI_API_KEY or GOOGLE_API_KEY and python -m pip install google-generativeai
    - llama: requires LLAMA_API_KEY and an OpenAI-compatible /chat/completions endpoint

    The deterministic recommend() function remains the fallback path.
    """
    if not matches:
        return _llm_fallback(matches, query, "no comparable cases were found")

    selected_provider = _select_llm_provider(provider)
    prompt = build_llm_prompt(matches, query, tactical_contexts=tactical_contexts)

    try:
        text = _call_provider_llm(selected_provider, prompt, model_name=model_name, json_mode=True)

        data = _extract_json_object(text)
        result = _normalize_llm_recommendation(data, matches)
        result["llm_provider"] = selected_provider
        result["llm_model"] = _provider_model_name(selected_provider, model_name)
        return result
    except Exception as exc:
        return _llm_fallback(matches, query, str(exc))


def build_chat_prompt(matches: list[tuple[float, ShotCase]], query: dict, message: str) -> str:
    """Build a Korean chatbot prompt grounded in the retrieved RAG cases."""
    case_blocks: list[str] = []
    for rank, (score, case) in enumerate(matches, start=1):
        case_blocks.append(
            "\n".join(
                [
                    f"Case {rank}",
                    f"score: {score:.3f}",
                    f"case_id: {case.case_id}",
                    f"source: {case.source}",
                    f"target_type: {case.target_type}",
                    f"distance: {case.distance}",
                    f"body_error: {case.body_error}",
                    f"turret_error: {case.turret_error}",
                    f"pitch_error: {case.pitch_error}",
                    f"enemy_speed: {case.enemy_speed}",
                    f"lead_distance: {case.lead_distance}",
                    f"impact_error: {case.impact_error}",
                    f"range_error: {case.range_error}",
                    f"result: {case.hit_label}",
                    f"document: {case.document}",
                ]
            )
        )

    return f"""
너는 TankChallenge RAG 의사결정 보조 챗봇이다.
반드시 검색된 과거 사례와 현재 상황만 근거로 한국어로 답한다.
근거 없는 추측을 하지 말고, 모르면 모른다고 말한다.
답변은 짧고 실전적으로 작성한다.

현재 상황:
- target_type: {query.get('target_type')}
- distance: {query.get('distance')} m
- body_yaw_error: {query.get('body_error')} deg
- turret_yaw_error: {query.get('turret_error')} deg
- pitch_error: {query.get('pitch_error')} deg
- impact_error: {query.get('impact_error')} m
- enemy_speed: {query.get('enemy_speed')} m/s
- lead_distance: {query.get('lead_distance')} m

검색된 유사 과거 사례:
{chr(10).join(case_blocks)}

사용자 질문:
{message}

답변 형식:
- 첫 문장에 결론을 말한다.
- 필요한 경우 yaw/pitch 보정 방향을 말한다.
- 근거 case_id를 1~3개 언급한다.
""".strip()


def chat_with_llm(
    matches: list[tuple[float, ShotCase]],
    query: dict,
    message: str,
    model_name: str | None = None,
    provider: str | None = None,
) -> dict:
    """Answer a user chat message using retrieved RAG cases plus an LLM."""
    selected_provider = _select_llm_provider(provider)
    prompt = build_chat_prompt(matches, query, message)
    try:
        answer = _call_provider_llm(selected_provider, prompt, model_name=model_name, json_mode=False)
        return {
            "answer": answer.strip(),
            "decision_mode": "llm_chat",
            "llm_provider": selected_provider,
            "llm_model": _provider_model_name(selected_provider, model_name),
            "evidence_case_ids": [case.case_id for _, case in matches[:3]],
        }
    except Exception as exc:
        base = recommend(matches, query)
        return {
            "answer": (
                f"LLM 채팅 호출에 실패해서 규칙 기반으로 답합니다: {exc}. "
                f"{base['summary']} yaw 보정 {base['yaw_correction_deg']:+.2f}도, "
                f"pitch 보정 {base['pitch_correction_deg']:+.2f}도입니다."
            ),
            "decision_mode": "rule_fallback_chat",
            "evidence_case_ids": [case.case_id for _, case in matches[:3]],
        }


def build_chat_prompt(
    matches: list[tuple[float, ShotCase]],
    query: dict,
    message: str,
    tactical_contexts: list[dict] | None = None,
) -> str:
    """Build a Korean chatbot prompt grounded in shot cases and live context."""
    case_blocks: list[str] = []
    for rank, (score, case) in enumerate(matches, start=1):
        case_blocks.append(
            "\n".join(
                [
                    f"Case {rank}",
                    f"score: {score:.3f}",
                    f"case_id: {case.case_id}",
                    f"source: {case.source}",
                    f"target_type: {case.target_type}",
                    f"distance: {case.distance}",
                    f"body_error: {case.body_error}",
                    f"turret_error: {case.turret_error}",
                    f"pitch_error: {case.pitch_error}",
                    f"enemy_speed: {case.enemy_speed}",
                    f"lead_distance: {case.lead_distance}",
                    f"impact_error: {case.impact_error}",
                    f"range_error: {case.range_error}",
                    f"result: {case.hit_label}",
                    f"document: {case.document}",
                ]
            )
        )

    return f"""
You are a TankChallenge RAG tactical chatbot.
Answer in Korean.
Use only the current situation, retrieved historical shot cases, and retrieved live LiDAR/YOLO tactical context.
Do not invent case IDs, enemy counts, obstacle positions, or sensor readings.
If live tactical context is unavailable, say that the live LiDAR/YOLO context is unavailable.

Current situation:
- target_type: {query.get('target_type')}
- distance: {query.get('distance')} m
- body_yaw_error: {query.get('body_error')} deg
- turret_yaw_error: {query.get('turret_error')} deg
- pitch_error: {query.get('pitch_error')} deg
- impact_error: {query.get('impact_error')} m
- enemy_speed: {query.get('enemy_speed')} m/s
- lead_distance: {query.get('lead_distance')} m

Retrieved historical shot cases:
{chr(10).join(case_blocks)}

Retrieved live LiDAR/YOLO tactical context:
{_format_tactical_contexts(tactical_contexts)}

User question:
{message}

Answer rules:
- Start with the direct conclusion.
- If the question asks about nearby enemies or obstacles, answer from the live tactical context first.
- If needed, mention yaw/pitch correction direction.
- Mention 1 to 3 evidence case IDs or tactical context IDs when useful.
""".strip()


def chat_with_llm(
    matches: list[tuple[float, ShotCase]],
    query: dict,
    message: str,
    model_name: str | None = None,
    provider: str | None = None,
    tactical_contexts: list[dict] | None = None,
) -> dict:
    """Answer a user chat message using retrieved RAG cases plus live context."""
    selected_provider = _select_llm_provider(provider)
    prompt = build_chat_prompt(matches, query, message, tactical_contexts=tactical_contexts)
    try:
        answer = _call_provider_llm(selected_provider, prompt, model_name=model_name, json_mode=False)
        return {
            "answer": answer.strip(),
            "decision_mode": "llm_chat",
            "llm_provider": selected_provider,
            "llm_model": _provider_model_name(selected_provider, model_name),
            "evidence_case_ids": [case.case_id for _, case in matches[:3]],
            "tactical_context_ids": [item.get("id") for item in (tactical_contexts or [])],
        }
    except Exception as exc:
        base = recommend(matches, query)
        tactical_note = " live LiDAR/YOLO tactical context was included." if tactical_contexts else " live LiDAR/YOLO tactical context is unavailable."
        return {
            "answer": (
                f"LLM 채팅 호출에 실패해서 규칙 기반으로 답합니다: {exc}."
                f"{tactical_note} {base['summary']} yaw 보정 {base['yaw_correction_deg']:+.2f}도, "
                f"pitch 보정 {base['pitch_correction_deg']:+.2f}도입니다."
            ),
            "decision_mode": "rule_fallback_chat",
            "evidence_case_ids": [case.case_id for _, case in matches[:3]],
            "tactical_context_ids": [item.get("id") for item in (tactical_contexts or [])],
        }


def summarize_matches(matches: list[tuple[float, ShotCase]]) -> dict:
    total = len(matches)
    success = [case for _, case in matches if case.hit_label == "success"]
    failure = [case for _, case in matches if case.hit_label != "success"]
    impact_errors = [case.impact_error for _, case in matches if case.impact_error is not None]
    success_distances = [case.distance for case in success if case.distance is not None]
    failure_distances = [case.distance for case in failure if case.distance is not None]
    return {
        "top_k": total,
        "success_count": len(success),
        "failure_count": len(failure),
        "success_rate": len(success) / total if total else 0.0,
        "avg_impact_error": mean(impact_errors) if impact_errors else 0.0,
        "avg_success_distance": mean(success_distances) if success_distances else None,
        "avg_failure_distance": mean(failure_distances) if failure_distances else None,
    }


def compare_backends(query: dict, top_k: int = 5) -> dict:
    cases = read_index()
    hybrid_matches = search(cases, query, top_k=top_k)
    faiss_matches = search_faiss(query, top_k=top_k)
    chroma_matches = search_chroma(query, top_k=top_k)
    hybrid_ids = {case.case_id for _, case in hybrid_matches}
    faiss_ids = {case.case_id for _, case in faiss_matches}
    chroma_ids = {case.case_id for _, case in chroma_matches}
    overlap = sorted(hybrid_ids & faiss_ids)
    return {
        "hybrid": summarize_matches(hybrid_matches),
        "faiss": summarize_matches(faiss_matches),
        "chroma": summarize_matches(chroma_matches),
        "overlap_count": len(overlap),
        "overlap_case_ids": overlap,
        "faiss_chroma_overlap_count": len(faiss_ids & chroma_ids),
    }


def evaluate(cases: list[ShotCase], top_k: int = 5, backend: str = "hybrid") -> dict:
    if len(cases) < 2:
        return {"case_count": len(cases), "accuracy": 0.0}
    correct = 0
    total = 0
    errors_before: list[float] = []
    errors_after: list[float] = []
    for index, case in enumerate(cases):
        train = cases[:index] + cases[index + 1 :]
        query = {key: getattr(case, key) for key in NUMERIC_FEATURES}
        query["target_type"] = case.target_type
        if backend == "faiss":
            matches = search_faiss(query, top_k=top_k)
            matches = [(score, match) for score, match in matches if match.case_id != case.case_id]
            matches = matches[:top_k]
        elif backend == "chroma":
            matches = search_chroma(query, top_k=top_k)
            matches = [(score, match) for score, match in matches if match.case_id != case.case_id]
            matches = matches[:top_k]
        else:
            matches = search(train, query, top_k=top_k)
        rec = recommend(matches, query)
        predicted_success = rec["confidence"] >= 0.5
        actual_success = case.hit_label == "success"
        correct += int(predicted_success == actual_success)
        total += 1
        if case.impact_error is not None:
            errors_before.append(case.impact_error)
            nearest_success = next((m for _, m in matches if m.hit_label == "success"), None)
            if nearest_success and nearest_success.impact_error is not None:
                errors_after.append(min(case.impact_error, nearest_success.impact_error))
            else:
                errors_after.append(case.impact_error)
    return {
        "case_count": len(cases),
        "success_cases": sum(1 for case in cases if case.hit_label == "success"),
        "failure_cases": sum(1 for case in cases if case.hit_label != "success"),
        "retrieval_accuracy": correct / total if total else 0.0,
        "avg_impact_error": mean(errors_before) if errors_before else 0.0,
        "similar_success_error_floor": mean(errors_after) if errors_after else 0.0,
        "backend": backend,
    }


def _search_backend(backend: str, query: dict, top_k: int = 5) -> list[tuple[float, ShotCase]]:
    cases = read_index()
    if backend == "faiss":
        return search_faiss(query, top_k=top_k)
    if backend == "chroma":
        return search_chroma(query, top_k=top_k)
    return search(cases, query, top_k=top_k)


def write_report(path: str = DEFAULT_REPORT, top_k: int = 5, backend: str = "hybrid") -> dict:
    cases = write_index()
    metrics = evaluate(cases, top_k=top_k, backend=backend)
    examples = cases[:3]
    sample_queries = [
        {
            "name": "Moving target mid-range aim",
            "query": {
                "target_type": "moving_enemy",
                "distance": 85.0,
                "body_error": 3.0,
                "turret_error": 0.8,
                "pitch_error": -0.1,
                "enemy_speed": 0.4,
                "lead_distance": 0.0,
            },
        },
        {
            "name": "Stationary far target",
            "query": {
                "target_type": "stationary",
                "distance": 120.0,
                "body_error": 2.0,
                "turret_error": 0.2,
                "pitch_error": 0.05,
                "enemy_speed": 0.0,
                "lead_distance": 0.0,
            },
        },
    ]
    lines = [
        "# TankChallenge RAG Decision Support Report",
        "",
        "## Purpose",
        "",
        "TankChallenge shot logs are converted into natural-language battle case documents.",
        "The system embeds those cases, retrieves similar historical success/failure shots,",
        "and recommends whether to fire or correct aim first.",
        "",
        "## Dataset",
        "",
        f"- Indexed cases: {metrics['case_count']}",
        f"- Success cases: {metrics['success_cases']}",
        f"- Failure cases: {metrics['failure_cases']}",
        f"- Search backend: {backend}",
        f"- Top-k: {top_k}",
        "",
        "## Offline Evaluation",
        "",
        f"- Conservative fire/hold decision agreement: {metrics['retrieval_accuracy']:.1%}",
        f"- Average impact error: {metrics['avg_impact_error']:.2f}m",
        f"- Similar-success error floor: {metrics['similar_success_error_floor']:.2f}m",
        "",
        "> Note: decision agreement is a strict fire/hold threshold metric, not raw retrieval precision.",
        "> Retrieval quality is inspected in the backend comparison table below.",
        "",
        "## Retrieval Quality Comparison",
        "",
        "| Query | Backend | Success/Failure | Success Rate | Avg Impact Error | Avg Success Distance |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for sample in sample_queries:
        comparison = compare_backends(sample["query"], top_k=top_k)
        for name in ("faiss", "chroma", "hybrid"):
            summary = comparison[name]
            avg_success_distance = summary["avg_success_distance"]
            distance_text = f"{avg_success_distance:.1f}m" if avg_success_distance is not None else "N/A"
            lines.append(
                f"| {sample['name']} | {name} | "
                f"{summary['success_count']}/{summary['failure_count']} | "
                f"{summary['success_rate']:.0%} | "
                f"{summary['avg_impact_error']:.2f}m | "
                f"{distance_text} |"
            )
        lines.append(
            f"| {sample['name']} | overlap | "
            f"FAISS-Hybrid: {comparison['overlap_count']}, FAISS-Chroma: {comparison['faiss_chroma_overlap_count']} | "
            "| | | |"
        )
    lines.extend(
        [
            "",
            "## Recommendation Examples",
            "",
        ]
    )
    for sample in sample_queries:
        matches = _search_backend(backend, sample["query"], top_k=top_k)
        rec = recommend(matches, sample["query"])
        quality = summarize_matches(matches)
        lines.extend(
            [
                f"### {sample['name']}",
                "",
                "```json",
                json.dumps(
                    {
                        "query": sample["query"],
                        "recommendation": rec,
                        "quality": quality,
                        "top_matches": [
                            {
                                "case_id": case.case_id,
                                "score": round(score, 3),
                                "hit_label": case.hit_label,
                                "distance": case.distance,
                                "impact_error": case.impact_error,
                            }
                            for score, case in matches[:3]
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## PCA Visualization",
            "",
            "The embedding space is reduced from 384 dimensions to 2 dimensions using PCA.",
            "",
            "![PCA embedding map](static/pca_embedding_map.png)",
            "",
            "PCA data file: `pca_embedding_points.csv`",
            "",
            "## Enhanced Recommendation Logic",
            "",
            "- Weighted average is computed from similar success cases.",
            "- Failure cases adjust the correction away from repeated miss patterns.",
            "- Distance buckets are separated into close, mid, and far.",
            "- Moving and stationary targets use separate confidence thresholds.",
            "- Fire/hold confidence threshold is tuned by distance, target mode, and current aim risk.",
            "",
            "## Example Case Documents",
            "",
        ]
    )
    for case in examples:
        lines.append(f"- `{case.case_id}`: {case.document}")
    lines.extend(
        [
            "",
            "## How To Run",
            "",
            "```powershell",
            "python rag_decision_support\\tank_rag.py build",
            "python rag_decision_support\\tank_rag.py build-embeddings",
            "python rag_decision_support\\tank_rag.py build-chroma",
            "python rag_decision_support\\pca_visualize.py",
            "python rag_decision_support\\tank_rag.py query --backend faiss --distance 85 --body-error 3 --turret-error 0.8 --pitch-error -0.1 --enemy-speed 0.4",
            "python rag_decision_support\\tank_rag.py report --backend faiss",
            "```",
        ]
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    return metrics


def _make_query(args: argparse.Namespace) -> dict:
    return {
        "target_type": args.target_type,
        "distance": args.distance,
        "body_error": args.body_error,
        "turret_error": args.turret_error,
        "pitch_error": args.pitch_error,
        "impact_error": args.impact_error,
        "enemy_speed": args.enemy_speed,
        "lead_distance": args.lead_distance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TankChallenge local RAG decision support")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("build", help="Build JSONL case index from shot logs")
    embedding_parser = subparsers.add_parser("build-embeddings", help="Build Hugging Face + FAISS vector index")
    embedding_parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    chroma_parser = subparsers.add_parser("build-chroma", help="Build Hugging Face + ChromaDB vector store")
    chroma_parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    report_parser = subparsers.add_parser("report", help="Build index and markdown report")
    report_parser.add_argument("--top-k", type=int, default=5)
    report_parser.add_argument("--backend", choices=("hybrid", "faiss", "chroma"), default="hybrid")

    query_parser = subparsers.add_parser("query", help="Search similar cases and recommend action")
    query_parser.add_argument("--distance", type=float, required=True)
    query_parser.add_argument("--body-error", type=float, default=0.0)
    query_parser.add_argument("--turret-error", type=float, default=0.0)
    query_parser.add_argument("--pitch-error", type=float, default=0.0)
    query_parser.add_argument("--impact-error", type=float)
    query_parser.add_argument("--enemy-speed", type=float, default=0.0)
    query_parser.add_argument("--lead-distance", type=float, default=0.0)
    query_parser.add_argument("--target-type", default="moving_enemy")
    query_parser.add_argument("--top-k", type=int, default=5)
    query_parser.add_argument("--backend", choices=("hybrid", "faiss", "chroma"), default="faiss")
    query_parser.add_argument("--model", default=None)
    query_parser.add_argument("--decision-mode", choices=("rule", "llm"), default="rule")
    query_parser.add_argument("--llm-model", default=None)
    query_parser.add_argument("--llm-provider", choices=("llama", "gemini"), default=DEFAULT_LLM_PROVIDER)

    if len(sys.argv) == 1:
        parser.print_help()
        print()
        print("examples:")
        print("  python tank_rag.py build")
        print("  python tank_rag.py build-embeddings")
        print("  python tank_rag.py build-chroma")
        print("  python tank_rag.py report")
        print(
            "  python tank_rag.py query --backend faiss --decision-mode llm --distance 85 --body-error 3 "
            "--turret-error 0.8 --pitch-error -0.1 --enemy-speed 0.4"
        )
        return

    args = parser.parse_args()
    if args.command == "build":
        cases = write_index()
        print(f"indexed_cases={len(cases)}")
        print(f"index_path={DEFAULT_INDEX}")
    elif args.command == "build-embeddings":
        result = build_faiss_index(model_name=args.model)
        print(json.dumps(result, indent=2))
    elif args.command == "build-chroma":
        result = build_chroma_index(model_name=args.model)
        print(json.dumps(result, indent=2))
    elif args.command == "report":
        if args.backend == "faiss":
            build_faiss_index()
        elif args.backend == "chroma":
            build_chroma_index()
        metrics = write_report(top_k=args.top_k, backend=args.backend)
        print(json.dumps(metrics, indent=2))
        print(f"report_path={DEFAULT_REPORT}")
    elif args.command == "query":
        cases = read_index()
        query = _make_query(args)
        if args.backend == "faiss":
            matches = search_faiss(query, top_k=args.top_k, model_name=args.model)
        elif args.backend == "chroma":
            matches = search_chroma(query, top_k=args.top_k, model_name=args.model)
        else:
            matches = search(cases, query, top_k=args.top_k)
        if args.decision_mode == "llm":
            rec = recommend_with_llm(matches, query, model_name=args.llm_model, provider=args.llm_provider)
        else:
            rec = recommend(matches, query)
        output = {
            "query": query,
            "backend": args.backend,
            "decision_mode": args.decision_mode,
            "recommendation": rec,
            "matches": [
                {
                    "score": round(score, 3),
                    "case_id": case.case_id,
                    "source": case.source,
                    "hit_label": case.hit_label,
                    "distance": case.distance,
                    "turret_error": case.turret_error,
                    "pitch_error": case.pitch_error,
                    "impact_error": case.impact_error,
                    "document": case.document,
                }
                for score, case in matches
            ],
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
