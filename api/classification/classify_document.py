import json
from types import SimpleNamespace

import httpx
from fastapi import APIRouter, HTTPException, Request

from .config import (
    KEY_EMBEDDING_GRAPH_MIN_SIMILARITY,
    KEY_EMBEDDING_GRAPH_PATH,
    KEY_EMBEDDING_GRAPH_SPACE_SCALE,
    KEY_EMBEDDING_GRAPH_TOP_K,
    PROJECTION_BATCH_SIZE,
    PROJECTION_CACHE_FOLDER,
    PROJECTION_DICTIONARY_DIR,
    PROJECTION_DEVICE,
    PROJECTION_ENCODER_MODEL,
    PROJECTION_MODEL_PATH,
    PROJECTION_PRECISION,
    PROJECTION_TRAIN_DICTIONARY_DIR,
    QWEN_INFER_API_BASE_URL,
    QWEN_INFER_API_PATH,
    QWEN_INFER_MAX_NEW_TOKENS,
    QWEN_INFER_TEMPERATURE,
    QWEN_INFER_TIMEOUT_SECONDS,
    SIGNAL_NORMALIZE_INSIDE_SCORE,
    SIGNAL_NORMALIZE_KEY_SCORE,
    SIGNAL_NORMALIZE_MARGIN,
    SIGNAL_NORMALIZE_MAX_OTHER_SCORE,
)
from .utils.key_embedding_graph import (
    build_key_embedding_space_payload,
    load_embedding_model,
    load_trained_projection_model,
    normalize_dictionary,
)
from .model.inference import (
    clear_projection_state,
    run_projection_inference,
)


router = APIRouter()


@router.post("/qwen-infer")
async def qwen_infer(request: Request):
    request_payload = await _load_json_request(request)
    qwen_payload = _build_qwen_infer_payload(request_payload)
    return await _call_qwen_infer_api(qwen_payload)


@router.post("/projection-infer")
async def projection_infer(request: Request):
    request_payload = await _load_json_request(request)
    try:
        return run_projection_inference(request_payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/projection-normalize-signals")
async def projection_normalize_signals():
    try:
        return _normalize_redundant_inside_signals()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/key-embedding-graph")
async def build_key_embedding_graph():
    try:
        return _build_normalized_key_embedding_graph()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def _load_json_request(request):
    try:
        request_payload = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="json body가 필요합니다.") from exc

    if not isinstance(request_payload, dict):
        raise HTTPException(status_code=400, detail="json body는 객체여야 합니다.")
    return request_payload


def _build_qwen_infer_payload(request_payload):
    text = _clean_text(request_payload.get("text"))
    if not text:
        raise HTTPException(status_code=400, detail="text가 필요합니다.")

    return {
        "text": text,
        "max_new_tokens": _positive_int(request_payload.get("max_new_tokens"), QWEN_INFER_MAX_NEW_TOKENS),
        "temperature": _float_value(request_payload.get("temperature"), QWEN_INFER_TEMPERATURE),
        "include_raw": bool(request_payload.get("include_raw", False)),
    }


async def _call_qwen_infer_api(qwen_payload):
    url = _service_url(QWEN_INFER_API_BASE_URL, QWEN_INFER_API_PATH)

    try:
        async with httpx.AsyncClient(timeout=QWEN_INFER_TIMEOUT_SECONDS) as client:
            response = await client.post(url, json=qwen_payload)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        detail = _response_error_detail(exc.response)
        raise HTTPException(status_code=502, detail=f"Qwen infer API failed: {detail}") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Qwen infer API unreachable: {exc}") from exc


def _normalize_redundant_inside_signals():
    graph_args, embedding_model, projection_model, projection_metadata = _load_projection_graph_assets()
    normalization_payload = normalize_dictionary(
        PROJECTION_TRAIN_DICTIONARY_DIR,
        embedding_model,
        projection_model,
        projection_metadata,
        graph_args,
        output_dictionary_dir=PROJECTION_DICTIONARY_DIR,
    )

    if normalization_payload["summary"].get("written_file_count"):
        clear_projection_state()

    return normalization_payload


def _build_normalized_key_embedding_graph():
    graph_args, embedding_model, projection_model, projection_metadata = _load_projection_graph_assets()
    normalization_payload = normalize_dictionary(
        PROJECTION_TRAIN_DICTIONARY_DIR,
        embedding_model,
        projection_model,
        projection_metadata,
        graph_args,
        output_dictionary_dir=PROJECTION_DICTIONARY_DIR,
    )
    if normalization_payload["summary"].get("written_file_count"):
        clear_projection_state()

    graph_payload = build_key_embedding_space_payload(
        PROJECTION_DICTIONARY_DIR,
        embedding_model,
        projection_model,
        projection_metadata,
        graph_args,
        normalization_payload,
    )

    _update_graph_embedding_metadata(graph_payload)
    _write_key_embedding_graph(graph_payload)
    return graph_payload


def _load_projection_graph_assets():
    graph_args = _key_embedding_graph_args()
    embedding_model = load_embedding_model(
        PROJECTION_ENCODER_MODEL,
        PROJECTION_DEVICE,
        PROJECTION_CACHE_FOLDER,
        PROJECTION_PRECISION,
    )
    projection_model, projection_metadata = load_trained_projection_model(PROJECTION_MODEL_PATH, PROJECTION_DEVICE)
    return graph_args, embedding_model, projection_model, projection_metadata


def _key_embedding_graph_args():
    return SimpleNamespace(
        model_name=PROJECTION_ENCODER_MODEL,
        device=PROJECTION_DEVICE,
        precision=PROJECTION_PRECISION,
        batch_size=PROJECTION_BATCH_SIZE,
        top_k=KEY_EMBEDDING_GRAPH_TOP_K,
        min_similarity=KEY_EMBEDDING_GRAPH_MIN_SIMILARITY,
        space_scale=KEY_EMBEDDING_GRAPH_SPACE_SCALE,
        inside_score=SIGNAL_NORMALIZE_INSIDE_SCORE,
        key_score=SIGNAL_NORMALIZE_KEY_SCORE,
        margin=SIGNAL_NORMALIZE_MARGIN,
        max_other_score=SIGNAL_NORMALIZE_MAX_OTHER_SCORE,
    )


def _update_graph_embedding_metadata(graph_payload):
    embedding_payload = graph_payload.get("embedding")
    if not isinstance(embedding_payload, dict):
        return

    embedding_payload["input_model"] = PROJECTION_ENCODER_MODEL
    embedding_payload["projection_model_path"] = str(PROJECTION_MODEL_PATH)
    embedding_payload["device"] = PROJECTION_DEVICE
    embedding_payload["precision"] = PROJECTION_PRECISION
    embedding_payload["batch_size"] = PROJECTION_BATCH_SIZE


def _write_key_embedding_graph(graph_payload):
    graph_source = json.dumps(graph_payload, ensure_ascii=False, indent=2)
    KEY_EMBEDDING_GRAPH_PATH.write_text(graph_source + "\n", encoding="utf-8")


def _service_url(base_url, path):
    return f"{str(base_url).rstrip('/')}/{str(path).lstrip('/')}"


def _response_error_detail(response):
    try:
        error_payload = response.json()
    except ValueError:
        return response.text[:1000]

    if isinstance(error_payload, dict):
        return error_payload.get("detail") or error_payload
    return error_payload


def _positive_int(value, default_value):
    if value is None:
        return int(default_value)
    try:
        parsed_value = int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="max_new_tokens는 정수여야 합니다.") from exc
    if parsed_value <= 0:
        raise HTTPException(status_code=400, detail="max_new_tokens는 1 이상이어야 합니다.")
    return parsed_value


def _float_value(value, default_value):
    if value is None:
        return float(default_value)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="temperature는 숫자여야 합니다.") from exc


def _clean_text(value):
    return " ".join(str(value or "").replace("\r", "\n").split()).strip()
