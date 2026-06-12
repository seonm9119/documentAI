import argparse
import json
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from normalize import normalize_ocr_text

try:
    from .schema import normalize_target_payload
except ImportError:
    from schema import normalize_target_payload


AXES = {"subject", "document_type", "business_domain", "modifier"}
TRAIN_FILE_PATTERN = re.compile(r"^train_(\d+)\.jsonl$")
PROVENANCE_KEYS = {"source_json_path", "_source_json_path"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="/mnt/h")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--train-dir-name", default="train")
    parser.add_argument("--train-name", default="")
    parser.add_argument("--seed", default=17, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    source_root = Path(args.source_root)
    output_dir = Path(args.output_dir)
    train_dir = output_dir / args.train_dir_name
    train_dir.mkdir(parents=True, exist_ok=True)

    train_output_path = resolve_train_output_path(train_dir, args.train_name)
    existing_record_fingerprints = load_existing_record_fingerprints(train_dir, exclude_path=train_output_path)

    train_records = []
    skipped_records = 0
    for sft_folder in find_sft_folders(source_root):
        records, folder_skipped_records = load_sft_records(sft_folder, existing_record_fingerprints)
        train_records.extend(records)
        skipped_records += folder_skipped_records

    random.Random(args.seed).shuffle(train_records)

    write_jsonl(train_output_path, train_records)

    print(f"train={len(train_records)} {train_output_path}")
    print(f"skipped_existing={skipped_records}")


def resolve_train_output_path(train_dir, train_name):
    if train_name:
        train_name_path = Path(train_name)
        if train_name_path.is_absolute():
            return train_name_path
        return train_dir / train_name_path

    next_number, width = find_next_train_file_number(train_dir)
    return train_dir / f"train_{next_number:0{width}d}.jsonl"


def find_next_train_file_number(train_dir):
    max_number = 0
    width = 6
    for train_path in train_dir.glob("train_*.jsonl"):
        match = TRAIN_FILE_PATTERN.match(train_path.name)
        if not match:
            continue
        max_number = max(max_number, int(match.group(1)))
        width = max(width, len(match.group(1)))
    return max_number + 1, width


def load_existing_record_fingerprints(train_dir, exclude_path):
    existing_record_fingerprints = set()
    exclude_path = exclude_path.resolve()
    for train_path in sorted(train_dir.glob("train_*.jsonl")):
        if train_path.resolve() == exclude_path:
            continue
        with open(train_path, "r", encoding="utf-8") as train_file:
            for line_number, raw_line in enumerate(train_file, start=1):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    existing_record_fingerprints.add(record_fingerprint(json.loads(raw_line)))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{train_path}:{line_number} jsonl line 파싱에 실패했습니다.") from exc
    return existing_record_fingerprints


def find_sft_folders(source_root):
    sft_folders = []
    for path in sorted(source_root.rglob("sft")):
        if path.is_dir() and list(path.glob("*.json")):
            sft_folders.append(path)
    return sft_folders


def load_sft_records(sft_folder, existing_record_fingerprints):
    dataset_root = sft_folder.parent
    records = []
    skipped_records = 0
    for sft_json_path in sorted(sft_folder.glob("*.json")):
        record = json.loads(sft_json_path.read_text(encoding="utf-8"))
        validate_record(record, sft_json_path)
        record = embed_page_payloads(record, dataset_root)
        fingerprint = record_fingerprint(record)
        if fingerprint in existing_record_fingerprints:
            skipped_records += 1
            continue
        existing_record_fingerprints.add(fingerprint)
        records.append(record)
    return records, skipped_records


def validate_record(record, sft_json_path):
    if not isinstance(record, dict):
        raise ValueError(f"{sft_json_path} record는 객체여야 합니다.")

    target = record.get("target")
    if not isinstance(target, dict) or set(target) != AXES:
        raise ValueError(f"{sft_json_path} target은 4축을 모두 가져야 합니다.")

    for axis in AXES:
        axis_payload = target.get(axis)
        if not isinstance(axis_payload, dict):
            raise ValueError(f"{sft_json_path} {axis} payload가 잘못됐습니다.")
        if not axis_payload.get("key"):
            raise ValueError(f"{sft_json_path} {axis}.key가 비어 있습니다.")
        if axis_payload.get("key") != "unknown" and not axis_payload.get("signals"):
            raise ValueError(f"{sft_json_path} {axis}.signals가 비어 있습니다.")

    if not isinstance(record.get("paddle_pages"), list) or not record.get("paddle_pages"):
        raise ValueError(f"{sft_json_path} paddle_pages가 비어 있습니다.")


def embed_page_payloads(record, dataset_root):
    normalized_record = dict(record)
    normalized_record["paddle_pages"] = load_page_payloads(record["paddle_pages"], dataset_root)
    return build_normalized_sft_record(normalized_record)


def load_page_payloads(page_paths, dataset_root):
    page_payloads = []
    for page_path in page_paths:
        page_path = Path(page_path)
        if not page_path.is_absolute():
            page_path = dataset_root / page_path
        if not page_path.exists():
            raise FileNotFoundError(str(page_path))
        page_payloads.append(json.loads(page_path.read_text(encoding="utf-8")))
    return page_payloads


def build_normalized_sft_record(record):
    normalized_record = dict(record)
    normalized_record.pop("deepseek_pages", None)
    normalized_record["paddle_pages"] = build_normalized_page_payloads(
        record.get("paddle_pages"),
        extract_paddle_raw_text,
    )

    if "target" in normalized_record:
        normalized_record["target"] = normalize_target_payload(normalized_record.get("target"))
    if "output" in normalized_record:
        normalized_record["output"] = normalize_target_payload(normalized_record.get("output"))

    return normalized_record


def build_normalized_page_payloads(page_payloads, extract_text):
    if not isinstance(page_payloads, list):
        return []

    return [
        {"text": normalize_ocr_text(extract_text(page_payload))}
        for page_payload in page_payloads
    ]


def extract_paddle_raw_text(paddle_payload):
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
        return "\n".join(str(rec_text or "") for rec_text in rec_texts)
    return paddle_result.get("text") or ""


def record_fingerprint(record):
    return json.dumps(
        strip_provenance_keys(build_normalized_sft_record(record)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def strip_provenance_keys(value):
    if isinstance(value, dict):
        return {
            key: strip_provenance_keys(nested_value)
            for key, nested_value in value.items()
            if key not in PROVENANCE_KEYS
        }
    if isinstance(value, list):
        return [strip_provenance_keys(item) for item in value]
    return value


def write_jsonl(output_path, records):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            output_file.write("\n")


if __name__ == "__main__":
    main()
