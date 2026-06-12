import base64
import json
import os
import re
import shutil
from pathlib import Path
from uuid import uuid4

import httpx
from fastapi import APIRouter, File, HTTPException, UploadFile

from convert_to_img import convert_uploaded_file_to_images
from normalize import normalize_ocr_text
from .key_embedding.overlay import build_signal_overlay_images
from .key_embedding.schema import normalize_target_payload, paddle_page_jsons_to_model_text


DATA_DIR = Path(os.environ.get("DOCUMENT_AI_DATA_DIR", "/app/data"))
UPLOAD_DIR = DATA_DIR / "uploads"
IMAGE_DIR = DATA_DIR / "images"
PADDLE_DIR = DATA_DIR / "paddle"
OVERLAY_DIR = DATA_DIR / "overlays"
AXES = ("subject", "document_type", "business_domain", "modifier")
PADDLE_OCR_API_BASE_URL = os.environ.get("PADDLE_OCR_API_BASE_URL", "http://paddle-ocr:8001")
PADDLE_OCR_API_PATH = os.environ.get("PADDLE_OCR_API_PATH", "/inference")
PADDLE_OCR_TIMEOUT_SECONDS = float(os.environ.get("PADDLE_OCR_TIMEOUT_SECONDS", "300"))
KEY_EMBEDDING_API_BASE_URL = os.environ.get("KEY_EMBEDDING_API_BASE_URL", "http://192.168.0.21:8004")
KEY_EMBEDDING_API_PATH = os.environ.get("KEY_EMBEDDING_API_PATH", "/infer")
KEY_EMBEDDING_TIMEOUT_SECONDS = float(os.environ.get("KEY_EMBEDDING_TIMEOUT_SECONDS", "300"))
KEY_EMBEDDING_MAX_NEW_TOKENS = int(os.environ.get("KEY_EMBEDDING_MAX_NEW_TOKENS", "512"))
KEY_EMBEDDING_TEMPERATURE = float(os.environ.get("KEY_EMBEDDING_TEMPERATURE", "0"))


router = APIRouter()


@router.post("/classify-document")
async def classify_document(file: UploadFile = File(...)):
    job_id = uuid4().hex[:12]
    upload_name = _safe_file_name(file.filename)
    source_path = UPLOAD_DIR / job_id / upload_name
    output_dir = IMAGE_DIR / job_id
    paddle_output_dir = PADDLE_DIR / job_id
    overlay_output_dir = OVERLAY_DIR / job_id

    try:
        _save_upload_file(file, source_path)
        image_paths = convert_uploaded_file_to_images(source_path, output_dir)
        paddle_pages = await _extract_paddle_pages(image_paths)
        paddle_page_paths = _save_paddle_pages(paddle_pages, paddle_output_dir)
        normalized_text = _normalize_paddle_pages(paddle_pages)
        inference_response = await _infer_key_embedding(normalized_text) if normalized_text else None
        classification_result = _normalize_key_embedding_response(inference_response)
        overlay_images = build_signal_overlay_images(
            image_paths,
            paddle_pages,
            classification_result,
            overlay_output_dir,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        await file.close()

    return {
        "job_id": job_id,
        "file_name": upload_name,
        "page_count": len(image_paths),
        "images": [_relative_data_path(path) for path in image_paths],
        "paddle_pages": [_relative_data_path(path) for path in paddle_page_paths],
        "overlay_images": [_overlay_image_payload(overlay_image) for overlay_image in overlay_images],
        "ocr_text": normalized_text,
        "result": classification_result,
        "raw_output": _raw_output_from_response(inference_response),
        "warnings": _warnings_from_response(inference_response),
        "status": "classified" if inference_response else "no_ocr_text",
    }


def _save_upload_file(file, source_path):
    source_path.parent.mkdir(parents=True, exist_ok=True)
    with open(source_path, "wb") as output_file:
        shutil.copyfileobj(file.file, output_file)


def _empty_axis_result():
    return {
        axis: {
            "key": "unknown",
            "signals": [],
        }
        for axis in AXES
    }


async def _extract_paddle_pages(image_paths):
    paddle_pages = []
    for image_path in image_paths:
        paddle_pages.append(await _call_paddle_ocr(image_path))
    return paddle_pages


async def _call_paddle_ocr(image_path):
    payload = {
        "byte_img": base64.b64encode(Path(image_path).read_bytes()).decode("ascii"),
        "release_after_inference": True,
        "predict_options": {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "return_word_box": False,
        },
    }
    url = _service_url(PADDLE_OCR_API_BASE_URL, PADDLE_OCR_API_PATH)

    try:
        async with httpx.AsyncClient(timeout=PADDLE_OCR_TIMEOUT_SECONDS) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        detail = _response_error_detail(exc.response)
        raise HTTPException(status_code=502, detail=f"Paddle OCR API failed: {detail}") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Paddle OCR API unreachable: {exc}") from exc


def _save_paddle_pages(paddle_pages, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    for page_index, paddle_page in enumerate(paddle_pages, start=1):
        output_path = output_dir / f"page_{page_index:03d}.json"
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(paddle_page, output_file, ensure_ascii=False, indent=2)
        output_paths.append(output_path)

    return output_paths


def _normalize_paddle_pages(paddle_pages):
    raw_text = paddle_page_jsons_to_model_text(paddle_pages)
    return normalize_ocr_text(raw_text)


async def _infer_key_embedding(text):
    payload = {
        "text": text,
        "max_new_tokens": KEY_EMBEDDING_MAX_NEW_TOKENS,
        "temperature": KEY_EMBEDDING_TEMPERATURE,
        "include_raw": False,
    }
    url = _service_url(KEY_EMBEDDING_API_BASE_URL, KEY_EMBEDDING_API_PATH)

    try:
        async with httpx.AsyncClient(timeout=KEY_EMBEDDING_TIMEOUT_SECONDS) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        detail = _response_error_detail(exc.response)
        raise HTTPException(status_code=502, detail=f"Key embedding API failed: {detail}") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Key embedding API unreachable: {exc}") from exc


def _normalize_key_embedding_response(inference_response):
    if not isinstance(inference_response, dict):
        return _empty_axis_result()

    result_payload = inference_response.get("result")
    if not isinstance(result_payload, dict):
        result_payload = inference_response

    return normalize_target_payload(result_payload)


def _raw_output_from_response(inference_response):
    if isinstance(inference_response, dict):
        return inference_response.get("raw_output")
    return None


def _warnings_from_response(inference_response):
    warnings = inference_response.get("warnings") if isinstance(inference_response, dict) else []
    if not isinstance(warnings, list):
        return []
    return [str(warning) for warning in warnings if str(warning).strip()]


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


def _relative_data_path(path):
    try:
        return str(Path(path).relative_to(DATA_DIR))
    except ValueError:
        return str(path)


def _data_url(path):
    return f"/document-ai-data/{_relative_data_path(path)}"


def _overlay_image_payload(overlay_image):
    path = overlay_image["path"]
    return {
        "image": _relative_data_path(path),
        "url": _data_url(path),
        "matches": overlay_image.get("matches") or [],
    }


def _safe_file_name(file_name):
    file_name = Path(str(file_name or "uploaded")).name
    stem = Path(file_name).stem.strip()
    suffix = Path(file_name).suffix.lower()
    stem = re.sub(r"[^0-9A-Za-z_.-]+", "_", stem).strip("._-")
    return f"{stem or 'uploaded'}{suffix}"
