import numpy as np
import torch

from ..config import (
    KEY_SIGNAL_AXES,
    KEY_SIGNAL_UNKNOWN_KEY,
    PROJECTION_BATCH_SIZE,
    PROJECTION_CACHE_FOLDER,
    PROJECTION_CONFLICT_MARGIN,
    PROJECTION_DEVICE,
    PROJECTION_DICTIONARY_DIR,
    PROJECTION_DROP_SCORE,
    PROJECTION_ENCODER_MODEL,
    PROJECTION_KEEP_MARGIN,
    PROJECTION_KEEP_SCORE,
    PROJECTION_MODEL_CONFIG,
    PROJECTION_MODEL_PATH,
    PROJECTION_PRECISION,
    PROJECTION_REVIEW_SCORE,
)
from .train.network import KeyEmbeddingProjectionModel
from ..utils.embedding_encoder import encode_texts, ensure_cuda_device, load_encoder
from ..utils.signal_dictionary import load_dictionary_entries, normalize_vector


_PROJECTION_STATE = None


def load_projection_model(path, device=None):
    checkpoint = torch.load(path, map_location=device or "cpu", weights_only=False)
    projection_model = KeyEmbeddingProjectionModel(checkpoint["axes"], checkpoint.get("model_config"))
    projection_model.load_state_dict(checkpoint["state_dict"])
    projection_model.to(device or "cpu")
    projection_model.eval()
    return projection_model, checkpoint


def run_projection_inference(request_payload):
    projection_request = normalize_projection_request(request_payload)
    projection_state = load_projection_state()
    axes_report = {}

    for axis, axis_payload in projection_request.items():
        axes_report[axis] = score_axis(axis, axis_payload, projection_state)

    return {
        "model": "key_embedding_projection_inference",
        "projection_model_path": str(PROJECTION_MODEL_PATH),
        "encoder_model": PROJECTION_ENCODER_MODEL,
        "summary": summarize_axes(axes_report),
        "axes": axes_report,
    }


def normalize_projection_request(request_payload):
    if not isinstance(request_payload, dict) or not request_payload:
        raise ValueError("projection inference payload가 필요합니다.")

    if all(field in request_payload for field in ("axis", "key")):
        axis = clean_text(request_payload.get("axis"))
        return {
            axis: {
                "key": clean_text(request_payload.get("key")),
                "signals": clean_signal_list(request_payload.get("signals")),
            }
        }

    candidate_payload = request_payload.get("result") if isinstance(request_payload.get("result"), dict) else request_payload
    normalized_request = {}
    for axis in KEY_SIGNAL_AXES:
        axis_payload = candidate_payload.get(axis) if isinstance(candidate_payload, dict) else None
        if not isinstance(axis_payload, dict):
            continue

        key = clean_text(axis_payload.get("key"))
        signals = clean_signal_list(axis_payload.get("signals"))
        if key or signals:
            normalized_request[axis] = {
                "key": key or KEY_SIGNAL_UNKNOWN_KEY,
                "signals": signals,
            }

    if not normalized_request:
        raise ValueError("axis/key/signals 또는 Qwen result payload가 필요합니다.")
    return normalized_request


def load_projection_state():
    global _PROJECTION_STATE

    if _PROJECTION_STATE is not None:
        return _PROJECTION_STATE

    ensure_cuda_device(PROJECTION_DEVICE)
    projection_model, checkpoint = load_projection_model(PROJECTION_MODEL_PATH, PROJECTION_DEVICE)
    encoder = load_encoder(PROJECTION_ENCODER_MODEL, PROJECTION_DEVICE, PROJECTION_CACHE_FOLDER, PROJECTION_PRECISION)
    dictionary_entries = load_dictionary_entries(PROJECTION_DICTIONARY_DIR)
    axis_spaces = build_axis_spaces(dictionary_entries, projection_model, encoder)

    _PROJECTION_STATE = {
        "projection_model": projection_model,
        "checkpoint_metadata": checkpoint.get("metadata") or {},
        "encoder": encoder,
        "axis_spaces": axis_spaces,
    }
    return _PROJECTION_STATE


def clear_projection_state():
    global _PROJECTION_STATE

    _PROJECTION_STATE = None


def build_axis_spaces(dictionary_entries, projection_model, encoder):
    axis_spaces = {}

    for axis, axis_entries in dictionary_entries.items():
        texts = []
        text_refs = []
        for axis_entry in axis_entries:
            texts.append(axis_entry["key"])
            text_refs.append((axis_entry["key"], "key"))
            for signal in axis_entry["signals"]:
                texts.append(signal)
                text_refs.append((axis_entry["key"], "signal"))

        projected_vectors = project_texts(axis, texts, projection_model, encoder)
        vectors_by_key = {}
        key_vectors = {}

        for text_ref, vector in zip(text_refs, projected_vectors):
            key, text_type = text_ref
            vectors_by_key.setdefault(key, []).append(vector)
            if text_type == "key":
                key_vectors[key] = vector

        prototypes = {}
        for key, vectors in vectors_by_key.items():
            prototypes[key] = normalize_vector(np.vstack(vectors).mean(axis=0))

        axis_spaces[axis] = {
            "key_vectors": key_vectors,
            "prototypes": prototypes,
        }

    return axis_spaces


def score_axis(axis, axis_payload, projection_state):
    if axis not in KEY_SIGNAL_AXES:
        raise ValueError(f"지원하지 않는 axis입니다: {axis}")

    axis_space = projection_state["axis_spaces"].get(axis) or {}
    key = clean_text(axis_payload.get("key")) or KEY_SIGNAL_UNKNOWN_KEY
    signals = clean_signal_list(axis_payload.get("signals"))

    if not signals:
        return {
            "key": key,
            "summary": empty_signal_summary(),
            "signals": [],
        }

    signal_vectors = project_texts(axis, signals, projection_state["projection_model"], projection_state["encoder"])
    signal_reports = [
        score_signal(key, signal, signal_vector, axis_space)
        for signal, signal_vector in zip(signals, signal_vectors)
    ]

    return {
        "key": key,
        "summary": summarize_signal_reports(signal_reports),
        "signals": signal_reports,
    }


def project_texts(axis, texts, projection_model, encoder):
    embeddings = encode_texts(encoder, texts, PROJECTION_BATCH_SIZE, PROJECTION_MODEL_CONFIG["input_dim"])
    with torch.no_grad():
        tensor = torch.as_tensor(embeddings, dtype=torch.float32, device=PROJECTION_DEVICE)
        projected = projection_model(axis, tensor).detach().cpu().numpy().astype(np.float32)
    return [normalize_vector(vector) for vector in projected]


def score_signal(key, signal, signal_vector, axis_space):
    prototypes = axis_space.get("prototypes") or {}
    key_vectors = axis_space.get("key_vectors") or {}
    own_prototype = prototypes.get(key)
    key_vector = key_vectors.get(key)

    own_score = calculate_similarity(signal_vector, own_prototype) if own_prototype is not None else -1.0
    key_score = calculate_similarity(signal_vector, key_vector) if key_vector is not None else -1.0
    best_other_key, best_other_score = find_best_other_key(key, signal_vector, prototypes)
    margin = own_score - best_other_score

    return {
        "signal": signal,
        "status": decide_status(own_score, best_other_score, margin, key in prototypes),
        "own_score": round(float(own_score), 6),
        "key_score": round(float(key_score), 6),
        "best_other_key": best_other_key,
        "best_other_score": round(float(best_other_score), 6),
        "margin": round(float(margin), 6),
    }


def decide_status(own_score, best_other_score, margin, known_key):
    if not known_key:
        return "unknown_key"
    if best_other_score > own_score + PROJECTION_CONFLICT_MARGIN:
        return "conflict"
    if own_score < PROJECTION_DROP_SCORE and best_other_score < PROJECTION_DROP_SCORE:
        return "drop_candidate"
    if own_score >= PROJECTION_KEEP_SCORE and margin >= PROJECTION_KEEP_MARGIN:
        return "keep"
    if own_score >= PROJECTION_REVIEW_SCORE:
        return "review"
    return "drop_candidate"


def find_best_other_key(key, signal_vector, prototypes):
    best_other_key = ""
    best_other_score = -1.0

    for other_key, prototype in prototypes.items():
        if other_key == key:
            continue

        score = calculate_similarity(signal_vector, prototype)
        if score > best_other_score:
            best_other_key = other_key
            best_other_score = score

    return best_other_key, best_other_score


def calculate_similarity(left_vector, right_vector):
    return float(np.asarray(left_vector, dtype=np.float32) @ np.asarray(right_vector, dtype=np.float32))


def summarize_axes(axes_report):
    summary = empty_signal_summary()
    summary["axis_count"] = len(axes_report)

    for axis_report in axes_report.values():
        axis_summary = axis_report.get("summary") or {}
        for field in signal_summary_fields():
            summary[field] += int(axis_summary.get(field) or 0)

    return summary


def summarize_signal_reports(signal_reports):
    summary = empty_signal_summary()
    summary["signal_count"] = len(signal_reports)

    for signal_report in signal_reports:
        status = signal_report["status"]
        if status == "keep":
            summary["keep_count"] += 1
        elif status == "review":
            summary["review_count"] += 1
        elif status == "conflict":
            summary["conflict_count"] += 1
        elif status == "drop_candidate":
            summary["drop_candidate_count"] += 1
        elif status == "unknown_key":
            summary["unknown_key_count"] += 1

    return summary


def signal_summary_fields():
    return ("signal_count", "keep_count", "review_count", "conflict_count", "drop_candidate_count", "unknown_key_count")


def empty_signal_summary():
    return {
        "axis_count": 0,
        "signal_count": 0,
        "keep_count": 0,
        "review_count": 0,
        "conflict_count": 0,
        "drop_candidate_count": 0,
        "unknown_key_count": 0,
    }


def clean_signal_list(signals):
    if isinstance(signals, str):
        signals = [signals]
    if not isinstance(signals, list):
        return []

    clean_signals = []
    seen_signals = set()
    for signal in signals:
        clean_signal = clean_text(signal)
        if not clean_signal or clean_signal in seen_signals:
            continue
        clean_signals.append(clean_signal)
        seen_signals.add(clean_signal)
    return clean_signals


def clean_text(value):
    return " ".join(str(value or "").replace("\r", "\n").split()).strip()
