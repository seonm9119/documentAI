import html
import json
import re
from pathlib import Path


AXES = ("subject", "document_type", "business_domain", "modifier")


def ocr_page_jsons_to_model_text(deepseek_pages, paddle_pages, file_name="", source_type="pdf"):
    document_payload = ocr_page_jsons_to_document_payload(deepseek_pages, paddle_pages, file_name, source_type)
    return document_payload_to_text(document_payload)


def ocr_page_jsons_to_document_payload(deepseek_pages, paddle_pages, file_name="", source_type="pdf"):
    deepseek_page_payloads = load_json_pages(deepseek_pages, "deepseek_pages")
    paddle_page_payloads = load_json_pages(paddle_pages, "paddle_pages")

    if len(deepseek_page_payloads) != len(paddle_page_payloads):
        raise ValueError("deepseek_pages와 paddle_pages의 page 수가 같아야 합니다.")

    pages = []
    for page_index, page_pair in enumerate(zip(deepseek_page_payloads, paddle_page_payloads), start=1):
        deepseek_page_payload, paddle_page_payload = page_pair
        deepseek_text = extract_deepseek_text(deepseek_page_payload)
        paddle_text = extract_paddle_text(paddle_page_payload)
        pages.append({
            "page_index": page_index,
            "page_name": f"page_{page_index}",
            "raw_text": build_ocr_page_text(deepseek_text, paddle_text),
            "tables": [],
        })

    return normalize_document_payload({
        "file_name": file_name,
        "source_type": source_type,
        "pages": pages,
    })


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
    lines = []
    append_section(lines, "DEEPSEEK_OCR_TEXT", deepseek_text)
    append_section(lines, "PADDLE_OCR_TEXT", paddle_text)
    return "\n".join(lines).strip()


def normalize_document_payload(payload):
    payload = payload if isinstance(payload, dict) else {}
    return {
        "file_name": clean_text(payload.get("file_name")),
        "source_type": clean_text(payload.get("source_type")),
        "pages": normalize_pages(payload.get("pages")),
    }


def normalize_pages(raw_pages):
    if not isinstance(raw_pages, list):
        return []

    pages = []
    for page_index, raw_page in enumerate(raw_pages, start=1):
        if not isinstance(raw_page, dict):
            continue
        pages.append({
            "page_index": positive_int(raw_page.get("page_index")) or page_index,
            "page_name": clean_text(raw_page.get("page_name")),
            "raw_text": clean_block_text(raw_page.get("raw_text")),
            "tables": normalize_tables(raw_page.get("tables")),
        })
    return pages


def normalize_tables(raw_tables):
    if not isinstance(raw_tables, list):
        return []

    tables = []
    for raw_table in raw_tables:
        if not isinstance(raw_table, dict):
            continue
        tables.append({
            "columns": clean_text_list(raw_table.get("columns")),
            "rows": normalize_rows(raw_table.get("rows")),
            "raw_text": clean_block_text(raw_table.get("raw_text")),
        })
    return tables


def normalize_rows(raw_rows):
    if not isinstance(raw_rows, list):
        return []

    rows = []
    for raw_row in raw_rows:
        if isinstance(raw_row, list):
            row = clean_text_list(raw_row)
        else:
            row = clean_text_list([raw_row])
        if row:
            rows.append(row)
    return rows


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


def document_payload_to_text(payload):
    document_payload = normalize_document_payload(payload)
    lines = []

    append_section(lines, "FILE_NAME", document_payload["file_name"])
    append_section(lines, "SOURCE_TYPE", document_payload["source_type"])

    for page in document_payload["pages"]:
        append_section(lines, f"PAGE_{page['page_index']}_NAME", page["page_name"])
        append_section(lines, f"PAGE_{page['page_index']}_RAW_TEXT", page["raw_text"])
        append_tables(lines, page["page_index"], page["tables"])

    return "\n".join(lines).strip()


def append_tables(lines, page_index, tables):
    for table_index, table in enumerate(tables, start=1):
        table_prefix = f"PAGE_{page_index}_TABLE_{table_index}"
        if table["columns"]:
            append_section(lines, f"{table_prefix}_COLUMNS", " | ".join(table["columns"]))
        if table["rows"]:
            row_lines = [" | ".join(row) for row in table["rows"] if row]
            append_section(lines, f"{table_prefix}_ROWS", "\n".join(row_lines))
        append_section(lines, f"{table_prefix}_RAW_TEXT", table["raw_text"])


def append_section(lines, section_name, section_text):
    section_text = clean_block_text(section_text)
    if not section_text:
        return
    lines.append(f"[{section_name}]")
    lines.append(section_text)


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


def positive_int(value):
    try:
        parsed_value = int(value)
    except (TypeError, ValueError):
        return None
    return parsed_value if parsed_value > 0 else None
