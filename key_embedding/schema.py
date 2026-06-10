import html
import json
import re
from pathlib import Path


AXES = ("subject", "document_type", "business_domain", "modifier")


def ocr_page_jsons_to_model_text(deepseek_pages, paddle_pages):
    deepseek_page_payloads = load_json_pages(deepseek_pages, "deepseek_pages")
    paddle_page_payloads = load_json_pages(paddle_pages, "paddle_pages")

    if len(deepseek_page_payloads) != len(paddle_page_payloads):
        raise ValueError("deepseek_pages와 paddle_pages의 page 수가 같아야 합니다.")

    page_texts = []
    for deepseek_page_payload, paddle_page_payload in zip(deepseek_page_payloads, paddle_page_payloads):
        deepseek_text = extract_deepseek_text(deepseek_page_payload)
        paddle_text = extract_paddle_text(paddle_page_payload)
        page_texts.append(build_ocr_page_text(deepseek_text, paddle_text))

    return merge_raw_texts(page_texts)


def load_json_pages(json_pages, argument_name):
    if not isinstance(json_pages, list):
        raise ValueError(f"{argument_name}는 json page list여야 합니다.")

    page_payloads = []
    for json_page in json_pages:
        page_payloads.append(load_json_page(json_page))
    return page_payloads


def load_json_page(json_page):
    if isinstance(json_page, (str, Path)):
        with open(json_page, "r", encoding="utf-8") as json_file:
            return json.load(json_file)
    if isinstance(json_page, (dict, list)):
        return json_page
    raise ValueError("json page는 file path, dict, list 중 하나여야 합니다.")


def extract_deepseek_text(deepseek_payload):
    if not isinstance(deepseek_payload, dict):
        return ""

    raw_text = str(deepseek_payload.get("text") or "")
    raw_text = re.sub(r"<\|det\|>.*?<\|/det\|>", "\n", raw_text, flags=re.S)
    raw_text = re.sub(r"<\|ref\|>.*?<\|/ref\|>", "\n", raw_text, flags=re.S)
    raw_text = re.sub(r"</(td|th|tr|table)>", "\n", raw_text, flags=re.I)
    raw_text = re.sub(r"<[^>]+>", " ", raw_text)
    return clean_block_text(html.unescape(raw_text))


def extract_paddle_text(paddle_payload):
    paddle_result = paddle_payload
    if isinstance(paddle_payload, list):
        paddle_result = paddle_payload[0] if paddle_payload else {}
    if not isinstance(paddle_result, dict):
        return ""

    paddle_result = paddle_result.get("res", paddle_result)
    if not isinstance(paddle_result, dict):
        return ""

    rec_texts = paddle_result.get("rec_texts")
    if isinstance(rec_texts, list):
        return clean_ocr_text_lines(rec_texts)
    return clean_block_text(paddle_result.get("text"))


def build_ocr_page_text(deepseek_text, paddle_text):
    return merge_raw_texts([deepseek_text, paddle_text])


def normalize_axis_target(raw_axis_target):
    raw_axis_target = raw_axis_target if isinstance(raw_axis_target, dict) else {}
    axis_key = clean_text(raw_axis_target.get("key")) or "unknown"
    signals = clean_text_list(raw_axis_target.get("signals"))
    if axis_key == "unknown":
        signals = []
    return {
        "key": axis_key,
        "signals": signals,
    }


def normalize_target_payload(raw_target):
    raw_target = raw_target if isinstance(raw_target, dict) else {}
    return {
        axis: normalize_axis_target(raw_target.get(axis))
        for axis in AXES
    }


def merge_raw_texts(raw_texts):
    lines = []
    seen_lines = set()

    for raw_text in raw_texts:
        for line in clean_block_text(raw_text).splitlines():
            if not line or line in seen_lines:
                continue
            lines.append(line)
            seen_lines.add(line)

    return "\n".join(lines).strip()


def clean_text(value):
    return " ".join(str(value or "").replace("\r", "\n").split()).strip()


def clean_block_text(value):
    lines = []
    for line in str(value or "").replace("\r", "\n").split("\n"):
        clean_line = " ".join(line.split()).strip()
        if clean_line:
            lines.append(clean_line)
    return "\n".join(lines).strip()


def clean_text_list(values):
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []

    clean_values = []
    seen_values = set()
    for value in values:
        clean_value = clean_text(value)
        if not clean_value or clean_value in seen_values:
            continue
        clean_values.append(clean_value)
        seen_values.add(clean_value)
    return clean_values


def clean_ocr_text_lines(values):
    lines = []
    for value in values:
        clean_value = clean_text(value)
        if clean_value:
            lines.append(clean_value)
    return "\n".join(lines).strip()
