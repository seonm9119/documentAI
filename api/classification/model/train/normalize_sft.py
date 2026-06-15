import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from ... import config
    from ...utils.embedding_encoder import encode_texts, ensure_cuda_device, load_encoder
    from ...utils.signal_dictionary import clean_text, clean_text_list, load_dictionary_entries, normalize_vector
    from ..inference import load_projection_model, score_signal
except ImportError:
    from api.classification import config
    from api.classification.utils.embedding_encoder import encode_texts, ensure_cuda_device, load_encoder
    from api.classification.utils.signal_dictionary import clean_text, clean_text_list, load_dictionary_entries, normalize_vector
    from api.classification.model.inference import load_projection_model, score_signal


def parse_args():
    parser = argparse.ArgumentParser(description="Normalize SFT target signals with key projection embedding space.")
    parser.add_argument("sft_path")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--model-path", default=str(config.PROJECTION_MODEL_PATH))
    parser.add_argument("--dictionary-dir", default=str(config.PROJECTION_DICTIONARY_DIR))
    parser.add_argument("--encoder-model", default=config.PROJECTION_ENCODER_MODEL)
    parser.add_argument("--cache-folder", default=config.PROJECTION_CACHE_FOLDER)
    parser.add_argument("--device", default=config.PROJECTION_DEVICE)
    parser.add_argument("--precision", default=config.PROJECTION_PRECISION)
    parser.add_argument("--batch-size", default=config.PROJECTION_BATCH_SIZE, type=int)
    parser.add_argument("--keep-statuses", default="keep")
    parser.add_argument("--progress-interval", default=1000, type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    sft_path = Path(args.sft_path)
    if not sft_path.exists():
        raise FileNotFoundError(f"SFT 파일/폴더를 찾을 수 없습니다: {sft_path}")

    allowed_statuses = parse_allowed_statuses(args.keep_statuses)
    projection_state = load_signal_filter_state(args)
    source_paths = collect_sft_source_paths(sft_path)
    summary = empty_normalize_summary()

    for source_index, source_path in enumerate(source_paths, start=1):
        try:
            target_path = resolve_target_path(sft_path, source_path, args.output_dir)
            file_summary = normalize_sft_source_file(source_path, target_path, projection_state, allowed_statuses, args)
            merge_summary(summary, file_summary)
        except Exception as exc:
            summary["error_file_count"] += 1
            print(f"skip file: {source_path} ({type(exc).__name__}: {exc})", file=sys.stderr, flush=True)
        print_progress("normalize files", source_index, len(source_paths), args)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def parse_allowed_statuses(raw_statuses):
    statuses = [
        clean_text(status)
        for status in str(raw_statuses or "").split(",")
    ]
    return {status for status in statuses if status}


def load_signal_filter_state(args):
    ensure_cuda_device(args.device)
    projection_model, checkpoint = load_projection_model(args.model_path, args.device)
    encoder = load_encoder(args.encoder_model, args.device, args.cache_folder, args.precision)
    dictionary_entries = load_dictionary_entries(args.dictionary_dir)
    axis_spaces = build_axis_spaces(dictionary_entries, projection_model, encoder, args)
    return {
        "projection_model": projection_model,
        "projection_metadata": checkpoint.get("metadata") or {},
        "encoder": encoder,
        "axis_spaces": axis_spaces,
        "args": args,
    }


def build_axis_spaces(dictionary_entries, projection_model, encoder, args):
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

        projected_vectors = project_texts(axis, texts, projection_model, encoder, args)
        vectors_by_key = {}
        key_vectors = {}

        for text_ref, projected_vector in zip(text_refs, projected_vectors):
            axis_key, text_type = text_ref
            vectors_by_key.setdefault(axis_key, []).append(projected_vector)
            if text_type == "key":
                key_vectors[axis_key] = projected_vector

        prototypes = {}
        for axis_key, key_vectors_and_signals in vectors_by_key.items():
            prototypes[axis_key] = normalize_vector(np.vstack(key_vectors_and_signals).mean(axis=0))

        axis_spaces[axis] = {
            "key_vectors": key_vectors,
            "prototypes": prototypes,
        }

    return axis_spaces


def project_texts(axis, texts, projection_model, encoder, args):
    embeddings = encode_texts(encoder, texts, args.batch_size, config.PROJECTION_MODEL_CONFIG["input_dim"])
    with torch.no_grad():
        embedding_tensor = torch.as_tensor(embeddings, dtype=torch.float32, device=args.device)
        projected_tensor = projection_model(axis, embedding_tensor)
        projected_vectors = projected_tensor.detach().cpu().numpy().astype(np.float32)
    return [normalize_vector(projected_vector) for projected_vector in projected_vectors]


def collect_sft_source_paths(sft_path):
    if sft_path.is_file():
        if is_supported_source_path(sft_path):
            return [sft_path]
        raise ValueError(f"지원하지 않는 SFT 파일 형식입니다. json 파일만 지원합니다: {sft_path}")

    source_paths = sorted(sft_path.glob("*.json"))
    if not source_paths:
        raise FileNotFoundError(f"SFT 폴더 안에서 json 파일을 찾지 못했습니다: {sft_path}")
    return source_paths


def print_progress(label, current_count, total_count, args):
    progress_interval = max(0, int(args.progress_interval or 0))
    if progress_interval <= 0:
        return
    if current_count != total_count and current_count % progress_interval != 0:
        return

    print(f"{label}: {current_count}/{total_count}", flush=True)


def is_supported_source_path(source_path):
    return source_path.suffix.lower() == ".json"


def resolve_target_path(sft_path, source_path, output_dir):
    if not output_dir:
        return source_path

    output_dir = Path(output_dir)
    if sft_path.is_file():
        return output_dir / source_path.name

    return output_dir / source_path.relative_to(sft_path)


def normalize_sft_source_file(source_path, target_path, projection_state, allowed_statuses, args):
    return normalize_json_source_file(source_path, target_path, projection_state, allowed_statuses, args)


def normalize_json_source_file(source_path, target_path, projection_state, allowed_statuses, args):
    file_summary = empty_normalize_summary()
    file_summary["file_count"] = 1
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    changed = normalize_json_payload(source_payload, projection_state, allowed_statuses, file_summary)
    output_text = json.dumps(source_payload, ensure_ascii=False, indent=2) + "\n"
    write_source_file(target_path, output_text, changed, args)
    if changed:
        file_summary["changed_file_count"] = 1
    return file_summary


def normalize_json_payload(source_payload, projection_state, allowed_statuses, file_summary):
    changed = False

    if isinstance(source_payload, list):
        for source_record in source_payload:
            record_changed, record_summary = normalize_sft_record(source_record, projection_state, allowed_statuses)
            merge_summary(file_summary, record_summary)
            changed = changed or record_changed
        return changed

    if not isinstance(source_payload, dict):
        return False

    for record_field in ("records", "data"):
        source_records = source_payload.get(record_field)
        if not isinstance(source_records, list):
            continue

        for source_record in source_records:
            record_changed, record_summary = normalize_sft_record(source_record, projection_state, allowed_statuses)
            merge_summary(file_summary, record_summary)
            changed = changed or record_changed
        return changed

    record_changed, record_summary = normalize_sft_record(source_payload, projection_state, allowed_statuses)
    merge_summary(file_summary, record_summary)
    return record_changed


def normalize_sft_record(source_record, projection_state, allowed_statuses):
    record_summary = empty_normalize_summary()
    record_summary["record_count"] = 1
    if not isinstance(source_record, dict):
        return False, record_summary

    target_field = find_target_field(source_record)
    if not target_field:
        return False, record_summary

    target_payload = source_record[target_field]
    if not isinstance(target_payload, dict):
        return False, record_summary

    record_summary["target_count"] = 1
    changed = False
    for axis in config.KEY_SIGNAL_AXES:
        raw_axis_target = target_payload.get(axis)
        if not isinstance(raw_axis_target, dict):
            continue

        axis_changed, axis_summary = normalize_axis_target_signals(
            axis,
            raw_axis_target,
            projection_state,
            allowed_statuses,
        )
        merge_summary(record_summary, axis_summary)
        changed = changed or axis_changed

    return changed, record_summary


def find_target_field(source_record):
    if isinstance(source_record.get("target"), dict):
        return "target"
    if isinstance(source_record.get("output"), dict):
        return "output"
    return ""


def normalize_axis_target_signals(axis, raw_axis_target, projection_state, allowed_statuses):
    axis_summary = empty_normalize_summary()
    axis_summary["axis_count"] = 1
    axis_key = clean_text(raw_axis_target.get("key")) or config.KEY_SIGNAL_UNKNOWN_KEY
    raw_signals = raw_axis_target.get("signals")
    signals = clean_text_list(raw_signals)
    axis_summary["signal_count_before"] = len(signals)

    kept_signals = filter_axis_signals(axis, axis_key, signals, projection_state, allowed_statuses, axis_summary)
    axis_summary["signal_count_after"] = len(kept_signals)
    axis_summary["removed_signal_count"] = len(signals) - len(kept_signals)

    if raw_signals == kept_signals:
        return False, axis_summary

    raw_axis_target["signals"] = kept_signals
    return True, axis_summary


def filter_axis_signals(axis, axis_key, signals, projection_state, allowed_statuses, axis_summary):
    if not signals:
        return []

    axis_space = projection_state["axis_spaces"].get(axis) or {}
    if axis_key == config.KEY_SIGNAL_UNKNOWN_KEY or axis_key not in (axis_space.get("prototypes") or {}):
        axis_summary["status_counts"]["unknown_key"] = axis_summary["status_counts"].get("unknown_key", 0) + len(signals)
        return []

    projected_vectors = get_projected_signal_vectors(axis, signals, projection_state)
    kept_signals = []

    for signal, signal_vector in zip(signals, projected_vectors):
        signal_report = score_signal(axis_key, signal, signal_vector, axis_space)
        signal_status = signal_report["status"]
        axis_summary["status_counts"][signal_status] = axis_summary["status_counts"].get(signal_status, 0) + 1
        if signal_status in allowed_statuses:
            kept_signals.append(signal)

    return kept_signals


def get_projected_signal_vectors(axis, signals, projection_state):
    signal_vector_cache = projection_state.setdefault("signal_vector_cache", {})
    missing_signals = [
        signal
        for signal in signals
        if (axis, signal) not in signal_vector_cache
    ]

    if missing_signals:
        projected_vectors = project_texts(
            axis,
            missing_signals,
            projection_state["projection_model"],
            projection_state["encoder"],
            projection_state["args"],
        )
        for signal, projected_vector in zip(missing_signals, projected_vectors):
            signal_vector_cache[(axis, signal)] = projected_vector

    return [signal_vector_cache[(axis, signal)] for signal in signals]


def write_source_file(target_path, output_text, changed, args):
    if args.dry_run:
        return

    if not changed and not args.output_dir:
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(output_text, encoding="utf-8")


def empty_normalize_summary():
    return {
        "file_count": 0,
        "changed_file_count": 0,
        "error_file_count": 0,
        "record_count": 0,
        "target_count": 0,
        "axis_count": 0,
        "signal_count_before": 0,
        "signal_count_after": 0,
        "removed_signal_count": 0,
        "status_counts": {},
    }


def merge_summary(summary, next_summary):
    for summary_key, summary_value in next_summary.items():
        if summary_key == "status_counts":
            merge_status_counts(summary["status_counts"], summary_value)
            continue

        summary[summary_key] += int(summary_value or 0)


def merge_status_counts(status_counts, next_status_counts):
    for status, count in next_status_counts.items():
        status_counts[status] = status_counts.get(status, 0) + int(count or 0)


if __name__ == "__main__":
    main()
