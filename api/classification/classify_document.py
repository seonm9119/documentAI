import asyncio
import base64
import json
import os
import re
import uuid
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import httpx
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from api.utils.convert_to_img import (
    IMAGE_EXTENSIONS,
    PDF_EXTENSION,
    convert_uploaded_file_to_images,
)
from api.utils.normalize import normalize_ocr_text

from .utils.signal_dictionary import load_dictionary_entries
from .config import (
    KEY_EMBEDDING_GRAPH_MIN_SIMILARITY,
    KEY_EMBEDDING_GRAPH_PATH,
    KEY_EMBEDDING_GRAPH_SPACE_SCALE,
    KEY_EMBEDDING_GRAPH_TOP_K,
    KEY_SIGNAL_AXES,
    KEY_SIGNAL_UNKNOWN_KEY,
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
DOCUMENT_AI_DATA_DIR = Path(os.environ.get("DOCUMENT_AI_DATA_DIR", "/app/data"))
PADDLE_OCR_API_BASE_URL = os.environ.get("PADDLE_OCR_API_BASE_URL", "http://paddle-ocr:8001")
PADDLE_OCR_API_PATH = os.environ.get("PADDLE_OCR_API_PATH", "/inference")
PADDLE_OCR_RELEASE_API_PATH = os.environ.get("PADDLE_OCR_RELEASE_API_PATH", "/release")
PADDLE_OCR_TIMEOUT_SECONDS = float(os.environ.get("PADDLE_OCR_TIMEOUT_SECONDS", "120"))
PADDLE_OCR_RETRY_COUNT = int(os.environ.get("PADDLE_OCR_RETRY_COUNT", "2"))
PADDLE_OCR_RETRY_DELAY_SECONDS = float(os.environ.get("PADDLE_OCR_RETRY_DELAY_SECONDS", "0.75"))
CLASSIFY_QWEN_TIMEOUT_SECONDS = float(os.environ.get("CLASSIFY_QWEN_TIMEOUT_SECONDS", "30"))
CLASSIFY_MAX_UPLOAD_BYTES = int(os.environ.get("CLASSIFY_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
SUPPORTED_UPLOAD_EXTENSIONS = IMAGE_EXTENSIONS | {PDF_EXTENSION}

_KEY_EMBEDDING_GRAPH_MEMORY_CACHE = {
    "signature": None,
    "payload": None,
}
_KEY_EMBEDDING_GRAPH_CACHE_LOCK = Lock()
_CLASSIFICATION_DICTIONARY_MEMORY_CACHE = None
_CLASSIFICATION_DICTIONARY_CACHE_LOCK = Lock()


@router.post("/classify-document")
async def classify_document(file: UploadFile = File(...)):
    original_filename = _safe_upload_filename(file.filename)
    suffix = Path(original_filename).suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=400, detail="PDF 또는 이미지 파일만 업로드할 수 있습니다.")

    job_id = uuid.uuid4().hex[:12]
    upload_path = DOCUMENT_AI_DATA_DIR / "uploads" / f"{job_id}_{original_filename}"
    image_output_dir = DOCUMENT_AI_DATA_DIR / "images" / job_id
    overlay_output_dir = DOCUMENT_AI_DATA_DIR / "overlays" / job_id
    warnings = []

    await _save_upload_file(file, upload_path)

    try:
        image_paths = convert_uploaded_file_to_images(upload_path, image_output_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not image_paths:
        raise HTTPException(status_code=400, detail="분류할 페이지 이미지를 만들지 못했습니다.")

    page_texts = []
    page_records = []
    for page_index, image_path in enumerate(image_paths, start=1):
        ocr_payload = await _run_paddle_ocr(image_path, release_after_inference=page_index == len(image_paths))
        page_text = normalize_ocr_text("\n".join(_extract_paddle_texts(ocr_payload)))
        if page_text:
            page_texts.append(page_text)
        page_records.append({
            "image_path": image_path,
            "ocr_payload": ocr_payload,
        })

    ocr_text = _merge_page_texts(page_texts)
    if not ocr_text:
        raise HTTPException(status_code=422, detail="OCR 텍스트를 추출하지 못했습니다.")

    result, classify_source, qwen_response = await _classify_ocr_text(ocr_text, warnings)
    overlay_images = []
    for page_record in page_records:
        image_path = page_record["image_path"]
        overlay_filename = f"{Path(image_path).stem}_overlay.png"
        overlay_path = overlay_output_dir / overlay_filename
        matched_signal_count = _write_overlay_image(image_path, page_record["ocr_payload"], overlay_path, result)
        overlay_images.append({
            "url": f"/document-ai-data/overlays/{job_id}/{overlay_filename}",
            "matched_signal_count": matched_signal_count,
        })

    return {
        "job_id": job_id,
        "file_name": original_filename,
        "page_count": len(image_paths),
        "overlay_images": overlay_images,
        "result": result,
        "status": "classified",
        "source": classify_source,
        "warnings": warnings,
        "ocr": {
            "text_length": len(ocr_text),
            "page_text_count": len(page_texts),
        },
        "qwen": qwen_response,
    }


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
async def build_key_embedding_graph(refresh: bool = False):
    try:
        if not refresh:
            try:
                return get_cached_key_embedding_graph_payload()
            except FileNotFoundError:
                pass

        return _build_normalized_key_embedding_graph()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def _save_upload_file(file, destination_path):
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    byte_count = 0

    with destination_path.open("wb") as output_file:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break

            byte_count += len(chunk)
            if byte_count > CLASSIFY_MAX_UPLOAD_BYTES:
                destination_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="업로드 파일이 너무 큽니다.")

            output_file.write(chunk)

    if byte_count <= 0:
        destination_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="빈 파일은 분류할 수 없습니다.")

    return byte_count


async def _run_paddle_ocr(image_path, release_after_inference=False):
    url = _service_url(PADDLE_OCR_API_BASE_URL, PADDLE_OCR_API_PATH)
    image_bytes = Path(image_path).read_bytes()
    payload = {
        "byte_img": base64.b64encode(image_bytes).decode("ascii"),
        "release_after_inference": release_after_inference,
        "predict_options": {},
    }

    for attempt_index in range(PADDLE_OCR_RETRY_COUNT + 1):
        try:
            async with httpx.AsyncClient(timeout=PADDLE_OCR_TIMEOUT_SECONDS) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            detail = _response_error_detail(exc.response)
            if _should_retry_paddle_ocr(exc.response, detail, attempt_index):
                await _release_paddle_ocr()
                await asyncio.sleep(PADDLE_OCR_RETRY_DELAY_SECONDS)
                continue
            raise HTTPException(status_code=502, detail=f"Paddle OCR API failed: {detail}") from exc
        except httpx.HTTPError as exc:
            if attempt_index < PADDLE_OCR_RETRY_COUNT:
                await _release_paddle_ocr()
                await asyncio.sleep(PADDLE_OCR_RETRY_DELAY_SECONDS)
                continue
            raise HTTPException(status_code=502, detail=f"Paddle OCR API unreachable: {exc}") from exc

    raise HTTPException(status_code=502, detail="Paddle OCR API retry failed")


async def _release_paddle_ocr():
    url = _service_url(PADDLE_OCR_API_BASE_URL, PADDLE_OCR_RELEASE_API_PATH)
    try:
        async with httpx.AsyncClient(timeout=min(PADDLE_OCR_TIMEOUT_SECONDS, 30)) as client:
            await client.post(url, json={})
        return True
    except httpx.HTTPError:
        return False


def _should_retry_paddle_ocr(response, detail, attempt_index):
    if attempt_index >= PADDLE_OCR_RETRY_COUNT:
        return False
    if response.status_code == 507:
        return True
    return "out of memory" in str(detail).lower()


async def _classify_ocr_text(ocr_text, warnings):
    qwen_response = None

    try:
        qwen_payload = _build_qwen_infer_payload({
            "text": ocr_text,
            "include_raw": True,
        })
        qwen_response = await asyncio.wait_for(
            _call_qwen_infer_api(qwen_payload),
            timeout=CLASSIFY_QWEN_TIMEOUT_SECONDS,
        )
        qwen_result = _extract_qwen_classification_result(qwen_response)
        if qwen_result:
            normalized_qwen_result = _normalize_classification_result(qwen_result)
            fallback_result = _dictionary_and_keyword_classification(ocr_text, warnings)
            merged_result = _fill_unknown_axes(normalized_qwen_result, fallback_result)
            classify_source = "qwen" if merged_result == normalized_qwen_result else "qwen_with_fallback"
            return merged_result, classify_source, qwen_response

        warnings.append("Qwen 결과 JSON을 해석하지 못해 사전 기반 분류를 사용했습니다.")
    except asyncio.TimeoutError:
        warnings.append("Qwen infer API 응답 시간이 초과되어 사전 기반 분류를 사용했습니다.")
    except HTTPException as exc:
        warnings.append(f"Qwen infer API 호출 실패로 사전 기반 분류를 사용했습니다: {exc.detail}")

    return _dictionary_and_keyword_classification(ocr_text, warnings), "dictionary_fallback", qwen_response


def _extract_qwen_classification_result(qwen_response):
    if _looks_like_classification_result(qwen_response):
        return qwen_response

    if isinstance(qwen_response, dict):
        for field_name in ("result", "classification", "target", "output"):
            field_value = qwen_response.get(field_name)
            if _looks_like_classification_result(field_value):
                return field_value
            parsed_value = _parse_jsonish_payload(field_value)
            if _looks_like_classification_result(parsed_value):
                return parsed_value

        choices = qwen_response.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0] if isinstance(choices[0], dict) else {}
            message = first_choice.get("message") if isinstance(first_choice.get("message"), dict) else {}
            choice_text = message.get("content") or first_choice.get("text")
            parsed_choice = _parse_jsonish_payload(choice_text)
            if _looks_like_classification_result(parsed_choice):
                return parsed_choice

        for field_name in ("text", "content", "response", "generated_text", "raw"):
            parsed_value = _parse_jsonish_payload(qwen_response.get(field_name))
            if _looks_like_classification_result(parsed_value):
                return parsed_value

    parsed_response = _parse_jsonish_payload(qwen_response)
    if _looks_like_classification_result(parsed_response):
        return parsed_response
    return None


def _parse_jsonish_payload(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I).strip()
        text = re.sub(r"\s*```$", "", text).strip()

    for candidate in (text, _json_object_slice(text)):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _json_object_slice(text):
    start_index = text.find("{")
    end_index = text.rfind("}")
    if start_index < 0 or end_index <= start_index:
        return ""
    return text[start_index:end_index + 1]


def _looks_like_classification_result(value):
    return isinstance(value, dict) and any(axis in value for axis in KEY_SIGNAL_AXES)


def _normalize_classification_result(raw_result):
    raw_result = raw_result if isinstance(raw_result, dict) else {}
    return {
        axis: _normalize_axis_result(raw_result.get(axis))
        for axis in KEY_SIGNAL_AXES
    }


def _normalize_axis_result(raw_axis_result):
    raw_axis_result = raw_axis_result if isinstance(raw_axis_result, dict) else {}
    axis_key = _clean_text(raw_axis_result.get("key")) or KEY_SIGNAL_UNKNOWN_KEY
    signals = _clean_text_list(raw_axis_result.get("signals"))
    if axis_key == KEY_SIGNAL_UNKNOWN_KEY:
        signals = []
    return {
        "key": axis_key,
        "signals": signals,
    }


def _dictionary_and_keyword_classification(ocr_text, warnings):
    try:
        dictionary_result = _dictionary_classification(ocr_text)
    except (FileNotFoundError, json.JSONDecodeError, RuntimeError) as exc:
        warnings.append(f"분류 사전을 읽지 못해 키워드 기반 분류를 사용했습니다: {exc}")
        dictionary_result = _unknown_classification_result()

    keyword_result = _keyword_classification(ocr_text)
    return _merge_fallback_results(dictionary_result, keyword_result)


def _dictionary_classification(ocr_text):
    dictionary_entries = _load_classification_dictionary_entries()
    normalized_text = _matchable_text(ocr_text)
    compact_text = _compact_text(normalized_text)
    result = {}

    for axis in KEY_SIGNAL_AXES:
        best_entry = None
        for axis_entry in dictionary_entries.get(axis, []):
            score, matched_signals = _score_dictionary_entry(axis_entry, normalized_text, compact_text)
            if score <= 0:
                continue

            if best_entry is None or score > best_entry["score"]:
                best_entry = {
                    "score": score,
                    "key": axis_entry["key"],
                    "signals": matched_signals,
                }

        if best_entry:
            result[axis] = {
                "key": best_entry["key"],
                "signals": best_entry["signals"][:8],
            }
        else:
            result[axis] = _unknown_axis_result()

    return result


def _load_classification_dictionary_entries():
    global _CLASSIFICATION_DICTIONARY_MEMORY_CACHE

    with _CLASSIFICATION_DICTIONARY_CACHE_LOCK:
        if _CLASSIFICATION_DICTIONARY_MEMORY_CACHE is None:
            _CLASSIFICATION_DICTIONARY_MEMORY_CACHE = load_dictionary_entries(PROJECTION_DICTIONARY_DIR)
        return _CLASSIFICATION_DICTIONARY_MEMORY_CACHE


def _score_dictionary_entry(axis_entry, normalized_text, compact_text):
    score = 0
    matched_signals = []

    for signal in axis_entry.get("signals") or []:
        signal_score = _score_text_match(signal, normalized_text, compact_text, signal_weight=True)
        if signal_score <= 0:
            continue
        score += signal_score
        matched_signals.append(signal)

    key_score = _score_text_match(axis_entry.get("key"), normalized_text, compact_text, signal_weight=False)
    if key_score > 0:
        score += key_score
        if not matched_signals:
            matched_signals.append(axis_entry["key"])

    return score, _dedupe_text_list(matched_signals)


def _score_text_match(value, normalized_text, compact_text, signal_weight):
    candidate = _matchable_text(value)
    compact_candidate = _compact_text(candidate)
    if not _is_matchable_candidate(compact_candidate):
        return 0

    if candidate and candidate in normalized_text:
        return min(60, len(compact_candidate) + (12 if signal_weight else 6))
    if compact_candidate and compact_candidate in compact_text:
        return min(48, len(compact_candidate) + (8 if signal_weight else 4))
    return 0


def _keyword_classification(ocr_text):
    return {
        "subject": _keyword_axis_result(ocr_text, [
            ("예금", ("예금", "의뢰대상예금")),
            ("보험금", ("보험금", "보험")),
            ("계좌", ("계좌", "계좌개설")),
            ("상품거래", ("invoice", "seller", "buyer", "goods")),
        ]),
        "document_type": _keyword_axis_result(ocr_text, [
            ("신고서", ("신고서", "신고 서", "제신고")),
            ("상업송장", ("commercial invoice", "invoice")),
            ("증명서", ("certificate", "증명서")),
            ("청구서", ("청구서", "claim")),
            ("계약서", ("계약서", "contract")),
        ]),
        "business_domain": _keyword_axis_result(ocr_text, [
            ("금융", ("금융", "은행", "계좌", "예금")),
            ("보험", ("보험", "보험금")),
            ("무역", ("exporter", "importer", "invoice", "commercial invoice", "bill of lading")),
            ("증권", ("증권", "금융투자")),
            ("카드", ("카드", "card")),
        ]),
        "modifier": _keyword_axis_result(ocr_text, [
            ("사망", ("사망", "상속")),
            ("개인", ("개인", "individual")),
            ("기업", ("기업", "corporate", "company")),
            ("CIP", ("cip",)),
            ("CPT", ("cpt",)),
            ("EXW", ("exw",)),
        ]),
    }


def _keyword_axis_result(ocr_text, keyword_rules):
    normalized_text = _matchable_text(ocr_text)
    compact_text = _compact_text(normalized_text)

    for key, keywords in keyword_rules:
        signals = []
        for keyword in keywords:
            if _score_text_match(keyword, normalized_text, compact_text, signal_weight=True) > 0:
                signals.append(keyword)

        if signals:
            return {
                "key": key,
                "signals": _dedupe_text_list(signals),
            }

    return _unknown_axis_result()


def _merge_fallback_results(dictionary_result, keyword_result):
    merged_result = {}
    dictionary_result = dictionary_result if isinstance(dictionary_result, dict) else {}
    keyword_result = keyword_result if isinstance(keyword_result, dict) else {}

    for axis in KEY_SIGNAL_AXES:
        dictionary_axis = _normalize_axis_result(dictionary_result.get(axis))
        keyword_axis = _normalize_axis_result(keyword_result.get(axis))
        if keyword_axis["key"] != KEY_SIGNAL_UNKNOWN_KEY:
            merged_result[axis] = keyword_axis
        else:
            merged_result[axis] = dictionary_axis

    return merged_result


def _fill_unknown_axes(primary_result, fallback_result):
    merged_result = {}
    primary_result = primary_result if isinstance(primary_result, dict) else {}
    fallback_result = fallback_result if isinstance(fallback_result, dict) else {}

    for axis in KEY_SIGNAL_AXES:
        primary_axis = _normalize_axis_result(primary_result.get(axis))
        fallback_axis = _normalize_axis_result(fallback_result.get(axis))
        if primary_axis["key"] == KEY_SIGNAL_UNKNOWN_KEY and fallback_axis["key"] != KEY_SIGNAL_UNKNOWN_KEY:
            merged_result[axis] = fallback_axis
        else:
            merged_result[axis] = primary_axis

    return merged_result


def _unknown_classification_result():
    return {
        axis: _unknown_axis_result()
        for axis in KEY_SIGNAL_AXES
    }


def _unknown_axis_result():
    return {
        "key": KEY_SIGNAL_UNKNOWN_KEY,
        "signals": [],
    }


def _extract_paddle_texts(paddle_payload):
    paddle_result = _extract_paddle_result(paddle_payload)
    rec_texts = paddle_result.get("rec_texts")
    if isinstance(rec_texts, list):
        return _clean_text_list(rec_texts)

    text = _clean_text(paddle_result.get("text"))
    return [text] if text else []


def _extract_paddle_result(paddle_payload):
    paddle_result = paddle_payload
    if isinstance(paddle_payload, list):
        paddle_result = paddle_payload[0] if paddle_payload else {}
    if not isinstance(paddle_result, dict):
        return {}

    paddle_result = paddle_result.get("res", paddle_result)
    return paddle_result if isinstance(paddle_result, dict) else {}


def _write_overlay_image(image_path, ocr_payload, overlay_path, classification_result):
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required to draw OCR overlays.") from exc

    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    paddle_result = _extract_paddle_result(ocr_payload)
    polygons = _extract_signal_polygons(paddle_result, classification_result)

    with Image.open(image_path) as image:
        overlay_image = image.convert("RGB")
        draw = ImageDraw.Draw(overlay_image, "RGBA")
        for polygon in polygons:
            if len(polygon) < 2:
                continue
            draw.line(polygon + [polygon[0]], fill=(203, 47, 40, 220), width=3)
        overlay_image.save(overlay_path, format="PNG")
    return len(polygons)


def _extract_signal_polygons(paddle_result, classification_result):
    signal_terms = _extract_result_signal_terms(classification_result)
    if not signal_terms:
        return []

    signal_polygons = []
    for ocr_item in _extract_ocr_items(paddle_result):
        if _ocr_text_matches_signal(ocr_item["text"], signal_terms):
            signal_polygons.append(ocr_item["polygon"])
    return signal_polygons


def _extract_result_signal_terms(classification_result):
    if not isinstance(classification_result, dict):
        return []

    signal_terms = []
    seen_terms = set()
    for axis in KEY_SIGNAL_AXES:
        axis_payload = classification_result.get(axis)
        if not isinstance(axis_payload, dict):
            continue

        for signal in _clean_text_list(axis_payload.get("signals")):
            normalized_signal = _matchable_text(signal)
            compact_signal = _compact_text(normalized_signal)
            if not _is_matchable_candidate(compact_signal) or compact_signal in seen_terms:
                continue

            signal_terms.append({
                "text": normalized_signal,
                "compact": compact_signal,
            })
            seen_terms.add(compact_signal)
    return signal_terms


def _extract_ocr_items(paddle_result):
    raw_texts = paddle_result.get("rec_texts")
    if not isinstance(raw_texts, list) or not raw_texts:
        return []

    polygons = _extract_ocr_polygons(paddle_result)
    ocr_items = []
    for text_index, raw_text in enumerate(raw_texts):
        if text_index >= len(polygons):
            break

        text = _clean_text(raw_text)
        polygon = polygons[text_index]
        if text and polygon:
            ocr_items.append({
                "text": text,
                "polygon": polygon,
            })
    return ocr_items


def _ocr_text_matches_signal(ocr_text, signal_terms):
    normalized_text = _matchable_text(ocr_text)
    compact_text = _compact_text(normalized_text)
    if not compact_text:
        return False

    for signal_term in signal_terms:
        signal_text = signal_term["text"]
        signal_compact = signal_term["compact"]
        if signal_text and signal_text in normalized_text:
            return True
        if signal_compact and signal_compact in compact_text:
            return True
    return False


def _extract_ocr_polygons(paddle_result):
    for field_name in ("rec_polys", "polys", "dt_polys"):
        raw_polygons = paddle_result.get(field_name)
        if isinstance(raw_polygons, list) and raw_polygons:
            return [
                polygon
                for polygon in (_coerce_polygon(raw_polygon) for raw_polygon in raw_polygons)
                if polygon
            ]

    raw_boxes = paddle_result.get("rec_boxes")
    if isinstance(raw_boxes, list):
        return [
            polygon
            for polygon in (_coerce_box_polygon(raw_box) for raw_box in raw_boxes)
            if polygon
        ]
    return []


def _coerce_polygon(raw_polygon):
    if not isinstance(raw_polygon, list):
        return []
    if len(raw_polygon) == 4 and all(_is_number(value) for value in raw_polygon):
        return _coerce_box_polygon(raw_polygon)

    polygon = []
    for raw_point in raw_polygon:
        if not isinstance(raw_point, (list, tuple)) or len(raw_point) < 2:
            continue
        x_value, y_value = raw_point[0], raw_point[1]
        if _is_number(x_value) and _is_number(y_value):
            polygon.append((float(x_value), float(y_value)))
    return polygon


def _coerce_box_polygon(raw_box):
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) < 4:
        return []
    if not all(_is_number(value) for value in raw_box[:4]):
        return []

    x1, y1, x2, y2 = [float(value) for value in raw_box[:4]]
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _merge_page_texts(page_texts):
    merged_lines = []
    seen_lines = set()

    for page_text in page_texts:
        for line in normalize_ocr_text(page_text).splitlines():
            if not line or line in seen_lines:
                continue
            merged_lines.append(line)
            seen_lines.add(line)

    return "\n".join(merged_lines).strip()


def _safe_upload_filename(filename):
    raw_name = Path(str(filename or "uploaded")).name
    suffix = Path(raw_name).suffix.lower()
    stem = Path(raw_name).stem.strip()
    safe_stem = re.sub(r"[^0-9A-Za-z_.-]+", "_", stem).strip("._-") or "uploaded"
    return f"{safe_stem}{suffix}"


def _clean_text_list(values):
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []

    return _dedupe_text_list(_clean_text(value) for value in values)


def _dedupe_text_list(values):
    clean_values = []
    seen_values = set()

    for value in values:
        clean_value = _clean_text(value)
        normalized_value = _matchable_text(clean_value)
        if not clean_value or not normalized_value or normalized_value in seen_values:
            continue
        clean_values.append(clean_value)
        seen_values.add(normalized_value)

    return clean_values


def _matchable_text(value):
    return normalize_ocr_text(value).lower()


def _compact_text(value):
    return re.sub(r"\s+", "", str(value or ""))


def _is_matchable_candidate(compact_candidate):
    if not compact_candidate or compact_candidate.isdigit():
        return False
    if compact_candidate in {"and", "for", "the", "of", "to", "by", "no"}:
        return False
    if _contains_hangul(compact_candidate):
        return len(compact_candidate) >= 2
    return len(compact_candidate) >= 3


def _contains_hangul(value):
    return any("\uac00" <= char <= "\ud7a3" for char in str(value or ""))


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


def get_cached_key_embedding_graph_payload():
    if not KEY_EMBEDDING_GRAPH_PATH.exists():
        raise FileNotFoundError("key_embedding_graph.json을 찾을 수 없습니다.")

    graph_signature = _key_embedding_graph_file_signature()

    with _KEY_EMBEDDING_GRAPH_CACHE_LOCK:
        if (
            _KEY_EMBEDDING_GRAPH_MEMORY_CACHE["signature"] == graph_signature and
            _KEY_EMBEDDING_GRAPH_MEMORY_CACHE["payload"] is not None
        ):
            return _KEY_EMBEDDING_GRAPH_MEMORY_CACHE["payload"]

    try:
        graph_payload = json.loads(KEY_EMBEDDING_GRAPH_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("key_embedding_graph.json 파싱에 실패했습니다.") from exc

    _cache_key_embedding_graph_payload(graph_payload)
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
    KEY_EMBEDDING_GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    KEY_EMBEDDING_GRAPH_PATH.write_text(graph_source + "\n", encoding="utf-8")
    _cache_key_embedding_graph_payload(graph_payload)


def _cache_key_embedding_graph_payload(graph_payload):
    graph_signature = _key_embedding_graph_file_signature()

    with _KEY_EMBEDDING_GRAPH_CACHE_LOCK:
        _KEY_EMBEDDING_GRAPH_MEMORY_CACHE["signature"] = graph_signature
        _KEY_EMBEDDING_GRAPH_MEMORY_CACHE["payload"] = graph_payload


def _key_embedding_graph_file_signature():
    graph_stat = KEY_EMBEDDING_GRAPH_PATH.stat()
    return graph_stat.st_mtime_ns, graph_stat.st_size


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
