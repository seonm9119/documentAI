import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

UTILS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = UTILS_DIR.parents[2]
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    from ..config import (
        KEY_EMBEDDING_GRAPH_BATCH_SIZE,
        KEY_EMBEDDING_GRAPH_COLOR_HUE,
        KEY_EMBEDDING_GRAPH_COLOR_LIGHTNESS,
        KEY_EMBEDDING_GRAPH_COLOR_SATURATION,
        KEY_EMBEDDING_GRAPH_COORD_ROUND,
        KEY_EMBEDDING_GRAPH_DISTANCE_ROUND,
        KEY_EMBEDDING_GRAPH_KEY_SIZE,
        KEY_EMBEDDING_GRAPH_MIN_SIMILARITY,
        KEY_EMBEDDING_GRAPH_OUTPUT_NAME,
        KEY_EMBEDDING_GRAPH_SIGNAL_SIZE,
        KEY_EMBEDDING_GRAPH_SPACE_SCALE,
        KEY_EMBEDDING_GRAPH_TOP_K,
        KEY_EMBEDDING_NORMALIZE_REPORT_NAME,
        KEY_SIGNAL_AXIS_SOURCES,
        PROJECTION_CACHE_FOLDER,
        PROJECTION_DEVICE,
        PROJECTION_DICTIONARY_DIR,
        PROJECTION_ENCODER_MODEL,
        PROJECTION_ENCODER_TRUST_REMOTE_CODE,
        PROJECTION_MODEL_PATH,
        PROJECTION_PRECISION,
        PROJECTION_TRAIN_DICTIONARY_DIR,
        SIGNAL_NORMALIZE_INSIDE_SCORE,
        SIGNAL_NORMALIZE_KEY_SCORE,
        SIGNAL_NORMALIZE_MARGIN,
        SIGNAL_NORMALIZE_MAX_OTHER_SCORE,
    )
    from .signal_dictionary import (
        calculate_similarity,
        clean_text,
        find_inside_signal_decisions,
        normalize_text,
        normalize_vector,
    )
except ImportError:
    from api.classification.config import (
        KEY_EMBEDDING_GRAPH_BATCH_SIZE,
        KEY_EMBEDDING_GRAPH_COLOR_HUE,
        KEY_EMBEDDING_GRAPH_COLOR_LIGHTNESS,
        KEY_EMBEDDING_GRAPH_COLOR_SATURATION,
        KEY_EMBEDDING_GRAPH_COORD_ROUND,
        KEY_EMBEDDING_GRAPH_DISTANCE_ROUND,
        KEY_EMBEDDING_GRAPH_KEY_SIZE,
        KEY_EMBEDDING_GRAPH_MIN_SIMILARITY,
        KEY_EMBEDDING_GRAPH_OUTPUT_NAME,
        KEY_EMBEDDING_GRAPH_SIGNAL_SIZE,
        KEY_EMBEDDING_GRAPH_SPACE_SCALE,
        KEY_EMBEDDING_GRAPH_TOP_K,
        KEY_EMBEDDING_NORMALIZE_REPORT_NAME,
        KEY_SIGNAL_AXIS_SOURCES,
        PROJECTION_CACHE_FOLDER,
        PROJECTION_DEVICE,
        PROJECTION_DICTIONARY_DIR,
        PROJECTION_ENCODER_MODEL,
        PROJECTION_ENCODER_TRUST_REMOTE_CODE,
        PROJECTION_MODEL_PATH,
        PROJECTION_PRECISION,
        PROJECTION_TRAIN_DICTIONARY_DIR,
        SIGNAL_NORMALIZE_INSIDE_SCORE,
        SIGNAL_NORMALIZE_KEY_SCORE,
        SIGNAL_NORMALIZE_MARGIN,
        SIGNAL_NORMALIZE_MAX_OTHER_SCORE,
    )
    from signal_dictionary import (
        calculate_similarity,
        clean_text,
        find_inside_signal_decisions,
        normalize_text,
        normalize_vector,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Normalize dictionaries and build key embedding graph space.")
    parser.add_argument("--dictionary-dir", default=PROJECTION_DICTIONARY_DIR)
    parser.add_argument("--source-dictionary-dir", default=PROJECTION_TRAIN_DICTIONARY_DIR)
    parser.add_argument("--output", default=KEY_EMBEDDING_GRAPH_OUTPUT_NAME)
    parser.add_argument("--report-output", default=KEY_EMBEDDING_NORMALIZE_REPORT_NAME)
    parser.add_argument("--model-name", default=PROJECTION_ENCODER_MODEL)
    parser.add_argument("--device", default=PROJECTION_DEVICE)
    parser.add_argument("--cache-folder", default=PROJECTION_CACHE_FOLDER)
    parser.add_argument("--precision", default=PROJECTION_PRECISION)
    parser.add_argument("--batch-size", default=KEY_EMBEDDING_GRAPH_BATCH_SIZE, type=int)
    parser.add_argument("--top-k", default=KEY_EMBEDDING_GRAPH_TOP_K, type=int)
    parser.add_argument("--min-similarity", default=KEY_EMBEDDING_GRAPH_MIN_SIMILARITY, type=float)
    parser.add_argument("--space-scale", default=KEY_EMBEDDING_GRAPH_SPACE_SCALE, type=float)
    parser.add_argument("--inside-score", default=SIGNAL_NORMALIZE_INSIDE_SCORE, type=float)
    parser.add_argument("--key-score", default=SIGNAL_NORMALIZE_KEY_SCORE, type=float)
    parser.add_argument("--margin", default=SIGNAL_NORMALIZE_MARGIN, type=float)
    parser.add_argument("--max-other-score", default=SIGNAL_NORMALIZE_MAX_OTHER_SCORE, type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    dictionary_dir = Path(args.dictionary_dir)
    source_dictionary_dir = Path(args.source_dictionary_dir)
    graph_output_path = resolve_output_path(dictionary_dir, args.output)
    normalize_report_path = resolve_output_path(dictionary_dir, args.report_output)
    embedding_model = load_embedding_model(args.model_name, args.device, args.cache_folder, args.precision)
    projection_model, projection_metadata = load_trained_projection_model(PROJECTION_MODEL_PATH, args.device)
    normalized_payload = normalize_dictionary(
        source_dictionary_dir,
        embedding_model,
        projection_model,
        projection_metadata,
        args,
        output_dictionary_dir=dictionary_dir,
    )
    graph_payload = build_key_embedding_space_payload(
        dictionary_dir,
        embedding_model,
        projection_model,
        projection_metadata,
        args,
        normalized_payload,
    )

    normalize_report_source = json.dumps(normalized_payload, ensure_ascii=False, indent=2)
    graph_source = json.dumps(graph_payload, ensure_ascii=False, indent=2)
    normalize_report_path.write_text(normalize_report_source + "\n", encoding="utf-8")
    graph_output_path.write_text(graph_source + "\n", encoding="utf-8")
    print(f"created {normalize_report_path}")
    print(f"created {graph_output_path}")
    return 0


def resolve_output_path(dictionary_dir, output):
    output_path = Path(output)
    if output_path.is_absolute():
        return output_path
    return dictionary_dir / output_path


def normalize_dictionary(source_dictionary_dir, embedding_model, projection_model, projection_metadata, args, output_dictionary_dir=None):
    source_dictionary_dir = Path(source_dictionary_dir)
    output_dictionary_dir = Path(output_dictionary_dir or source_dictionary_dir)
    output_dictionary_dir.mkdir(parents=True, exist_ok=True)

    axes_report = {}
    thresholds = build_signal_normalize_thresholds(args)
    updated_file_count = 0
    written_file_count = 0

    for axis_name, axis_source in KEY_SIGNAL_AXIS_SOURCES.items():
        source_path = source_dictionary_dir / axis_source["file_name"]
        output_path = output_dictionary_dir / axis_source["file_name"]
        source_records = json.loads(source_path.read_text(encoding="utf-8"))
        axis_entries = build_axis_entries_from_records(source_records, axis_name, axis_source)
        axis_vectors = build_axis_vectors(embedding_model, projection_model, axis_name, axis_entries, args)
        remove_decisions = find_inside_signal_decisions(axis_entries, axis_vectors, thresholds)
        remove_norms_by_key = group_remove_norms(remove_decisions)

        if remove_norms_by_key:
            updated_file_count += 1

        if remove_norms_by_key or output_path != source_path:
            write_normalized_axis_records(output_path, source_records, axis_source, remove_norms_by_key)
            written_file_count += 1

        axes_report[axis_name] = build_axis_normalize_report(source_path, output_path, axis_entries, remove_decisions)

    summary = summarize_axes_normalize_report(axes_report)
    summary["updated_file_count"] = updated_file_count
    summary["written_file_count"] = written_file_count

    return {
        "model": "projection_signal_normalizer",
        "method": "remove_redundant_inside_key_area",
        "generated_at": current_utc_timestamp(),
        "source_dictionary_dir": str(source_dictionary_dir),
        "dictionary_dir": str(output_dictionary_dir),
        "output_dictionary_dir": str(output_dictionary_dir),
        "projection_model_path": str(PROJECTION_MODEL_PATH),
        "projection_metadata": projection_metadata,
        "thresholds": thresholds,
        "summary": summary,
        "axes": axes_report,
    }


def build_signal_normalize_thresholds(args):
    return {
        "inside_score": float(args.inside_score),
        "key_score": float(args.key_score),
        "margin": float(args.margin),
        "max_other_score": float(args.max_other_score),
    }


def build_axis_entries_from_records(source_records, axis_name, axis_source):
    key_field = axis_source["key_field"]
    axis_entries = []
    seen_keys = set()

    for record_index, source_record in enumerate(source_records):
        if not isinstance(source_record, dict):
            continue

        key = normalize_display_text(source_record.get(key_field))
        if not key or key in seen_keys:
            continue

        axis_entries.append({
            "axis": axis_name,
            "key": key,
            "record_index": record_index,
            "signals": build_signal_entries(clean_raw_signals(source_record.get("signals"))),
        })
        seen_keys.add(key)

    return axis_entries


def build_signal_entries(raw_signals):
    signal_entries = []
    seen_signals = set()

    for signal_index, raw_signal in enumerate(raw_signals):
        signal = normalize_display_text(raw_signal)
        signal_norm = normalize_embedding_text(signal)
        if not signal or signal_norm in seen_signals:
            continue

        signal_entries.append({
            "text": signal,
            "text_norm": signal_norm,
            "signal_index": signal_index,
        })
        seen_signals.add(signal_norm)

    return signal_entries


def clean_raw_signals(raw_signals):
    if isinstance(raw_signals, str):
        return [raw_signals]
    if isinstance(raw_signals, list):
        return raw_signals
    return []


def build_axis_vectors(embedding_model, projection_model, axis_name, axis_entries, args):
    texts = []
    for axis_entry in axis_entries:
        texts.append(axis_entry["key"])
        texts.extend(signal_entry["text"] for signal_entry in axis_entry["signals"])

    if not texts:
        return []

    encoded_vectors = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=max(1, int(args.batch_size)),
    )
    projected_vectors = project_encoded_vectors(projection_model, axis_name, encoded_vectors, args.device)
    axis_vectors = []
    offset = 0

    for axis_entry in axis_entries:
        signal_count = len(axis_entry["signals"])
        axis_vectors.append({
            "key_vector": projected_vectors[offset],
            "signal_vectors": projected_vectors[offset + 1:offset + 1 + signal_count],
        })
        offset += signal_count + 1

    return axis_vectors


def group_remove_norms(remove_decisions):
    remove_norms_by_key = {}

    for remove_decision in remove_decisions:
        remove_norms_by_key.setdefault(remove_decision["key"], set()).add(remove_decision["signal_norm"])

    return remove_norms_by_key


def write_normalized_axis_records(output_path, source_records, axis_source, remove_norms_by_key):
    key_field = axis_source["key_field"]

    for source_record in source_records:
        if not isinstance(source_record, dict) or "signals" not in source_record:
            continue

        key = normalize_display_text(source_record.get(key_field))
        remove_norms = remove_norms_by_key.get(key)
        if not remove_norms:
            continue

        source_record["signals"] = [
            signal
            for signal in clean_raw_signals(source_record.get("signals"))
            if normalize_embedding_text(signal) not in remove_norms
        ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(source_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_axis_normalize_report(source_path, output_path, axis_entries, remove_decisions):
    signal_count = sum(len(axis_entry["signals"]) for axis_entry in axis_entries)
    removed_count_by_key = {}

    for remove_decision in remove_decisions:
        removed_count_by_key[remove_decision["key"]] = removed_count_by_key.get(remove_decision["key"], 0) + 1

    return {
        "source_file": source_path.name,
        "source_path": str(source_path),
        "output_path": str(output_path),
        "summary": {
            "key_count": len(axis_entries),
            "signal_count_before": signal_count,
            "removed_signal_count": len(remove_decisions),
            "signal_count_after": signal_count - len(remove_decisions),
            "updated_key_count": len(removed_count_by_key),
        },
        "removed_signals": remove_decisions,
    }


def summarize_axes_normalize_report(axes_report):
    summary = {
        "key_count": 0,
        "signal_count_before": 0,
        "removed_signal_count": 0,
        "signal_count_after": 0,
        "updated_key_count": 0,
    }

    for axis_report in axes_report.values():
        axis_summary = axis_report["summary"]
        for summary_key in summary:
            summary[summary_key] += int(axis_summary.get(summary_key) or 0)

    return summary


def build_key_embedding_space_payload(dictionary_dir, embedding_model, projection_model, projection_metadata, args, normalized_payload):
    axes_payload = {}
    total_key_count = 0
    total_signal_count = 0
    total_edge_count = 0
    total_space_node_count = 0

    for axis_name, axis_source in KEY_SIGNAL_AXIS_SOURCES.items():
        axis_entries = load_axis_entries(dictionary_dir, axis_name, axis_source)
        axis_nodes = build_axis_nodes(axis_name, axis_entries)
        axis_embedding_data = build_axis_embedding_data(
            embedding_model,
            projection_model,
            axis_name,
            axis_entries,
            axis_nodes,
            args.batch_size,
            args.space_scale,
            args.device,
        )
        axis_vectors = axis_embedding_data["key_vectors"]
        axis_edges = build_similarity_edges(axis_name, axis_nodes, axis_vectors, args.top_k, args.min_similarity)
        axis_signal_count = sum(axis_node["signal_count"] for axis_node in axis_nodes)
        axis_space = axis_embedding_data["embedding_space"]

        axes_payload[axis_name] = {
            "source_file": axis_source["file_name"],
            "summary": {
                "key_count": len(axis_nodes),
                "signal_count": axis_signal_count,
                "edge_count": len(axis_edges),
                "space_node_count": len(axis_space["nodes"]),
                "space_link_count": len(axis_space["links"]),
            },
            "nodes": axis_nodes,
            "edges": axis_edges,
            "embedding_space": axis_space,
        }

        total_key_count += len(axis_nodes)
        total_signal_count += axis_signal_count
        total_edge_count += len(axis_edges)
        total_space_node_count += len(axis_space["nodes"])

    return {
        "version": 4,
        "generated_at": current_utc_timestamp(),
        "embedding": {
            "method": "sentence_transformer_projection_head",
            "input_model": args.model_name,
            "projection_model": projection_metadata.get("model") or "key_embedding_projection",
            "projection_model_path": str(PROJECTION_MODEL_PATH),
            "projection_best_valid_loss": projection_metadata.get("best_valid_loss"),
            "projection_train_epochs": projection_metadata.get("train_config", {}).get("epochs"),
            "projected_dim": projection_metadata.get("model_config", {}).get("output_dim"),
            "device": args.device,
            "precision": args.precision,
            "text_source": "key_and_signals",
            "normalized_embeddings": True,
            "include_canonical_key": True,
            "batch_size": args.batch_size,
            "top_k": args.top_k,
            "min_similarity": args.min_similarity,
            "space_projection": "pca_svd_3d",
            "space_scale": args.space_scale,
        },
        "normalization": {
            "method": "remove_redundant_inside_key_area",
            "thresholds": normalized_payload["thresholds"],
            "summary": normalized_payload["summary"],
        },
        "summary": {
            "axis_count": len(KEY_SIGNAL_AXIS_SOURCES),
            "key_count": total_key_count,
            "signal_count": total_signal_count,
            "edge_count": total_edge_count,
            "space_node_count": total_space_node_count,
        },
        "axes": axes_payload,
    }


def load_trained_projection_model(model_path, device):
    if not model_path.exists():
        raise FileNotFoundError(f"projection model이 없습니다: {model_path}")

    try:
        from api.classification.model.inference import load_projection_model
    except ImportError as exc:
        raise ImportError("key embedding projection model loader를 import할 수 없습니다.") from exc

    projection_model, checkpoint = load_projection_model(model_path, device)
    return projection_model, checkpoint.get("metadata") or {}


def load_embedding_model(model_name, device, cache_folder, precision):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("sentence-transformers가 설치되어 있지 않습니다.") from exc

    if is_cuda_device(device) and str(precision).lower() == "fp16":
        embedding_model = SentenceTransformer(
            model_name,
            device="cpu",
            cache_folder=cache_folder,
            trust_remote_code=PROJECTION_ENCODER_TRUST_REMOTE_CODE,
        )
        embedding_model.half()
        embedding_model.to(device)
        return embedding_model

    return SentenceTransformer(
        model_name,
        device=device,
        cache_folder=cache_folder,
        trust_remote_code=PROJECTION_ENCODER_TRUST_REMOTE_CODE,
    )


def is_cuda_device(device):
    return str(device or "").strip().lower().startswith("cuda")


def load_axis_entries(dictionary_dir, axis_name, axis_source):
    source_path = dictionary_dir / axis_source["file_name"]
    source_records = json.loads(source_path.read_text(encoding="utf-8"))
    key_field = axis_source["key_field"]
    axis_entries = []
    seen_keys = set()

    for source_record in source_records:
        if not isinstance(source_record, dict):
            continue

        canonical_key = normalize_display_text(source_record.get(key_field))
        if not canonical_key or canonical_key in seen_keys:
            continue

        axis_entries.append({
            "axis": axis_name,
            "key": canonical_key,
            "signals": dedupe_signals(clean_raw_signals(source_record.get("signals"))),
        })
        seen_keys.add(canonical_key)

    return axis_entries


def dedupe_signals(raw_signals):
    signals = []
    seen_signals = set()

    for raw_signal in raw_signals:
        signal = normalize_display_text(raw_signal)
        signal_key = normalize_embedding_text(signal)
        if not signal or signal_key in seen_signals:
            continue

        signals.append(signal)
        seen_signals.add(signal_key)

    return signals


def build_axis_nodes(axis_name, axis_entries):
    axis_nodes = []

    for entry_index, axis_entry in enumerate(axis_entries, start=1):
        canonical_key = axis_entry["key"]
        signals = axis_entry["signals"]
        axis_nodes.append({
            "id": f"key::{axis_name}::{canonical_key}",
            "type": "key",
            "axis": axis_name,
            "label": canonical_key,
            "key": canonical_key,
            "signal_count": len(signals),
            "signals": signals,
            "rank": entry_index,
        })

    return axis_nodes


def build_axis_embedding_data(embedding_model, projection_model, axis_name, axis_entries, axis_nodes, batch_size, space_scale, device):
    entry_text_groups = [entry_texts(axis_entry) for axis_entry in axis_entries]
    flat_texts = [text for text_group in entry_text_groups for text in text_group]
    if not flat_texts:
        return {
            "key_vectors": np.zeros((0, 0), dtype=np.float32),
            "embedding_space": empty_embedding_space(),
        }

    encoded_vectors = embedding_model.encode(
        flat_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=max(1, int(batch_size)),
    )
    projected_vectors = project_encoded_vectors(projection_model, axis_name, encoded_vectors, device)
    key_vectors = []
    space_records = []
    space_vectors = []
    offset = 0
    total_key_count = len(axis_entries)

    for entry_index, (axis_entry, axis_node, text_group) in enumerate(zip(axis_entries, axis_nodes, entry_text_groups), start=1):
        text_count = len(text_group)
        text_vectors = projected_vectors[offset:offset + text_count]
        offset += text_count
        key_vector = normalize_vector(text_vectors.mean(axis=0))
        key_vectors.append(key_vector)
        key_color = axis_graph_color(entry_index, total_key_count)
        space_records.append(build_space_key_node(axis_entry, axis_node, key_color))
        space_vectors.append(key_vector)

        for signal_index, signal in enumerate(axis_entry["signals"], start=1):
            signal_vector = normalize_vector(text_vectors[signal_index])
            space_records.append(build_space_signal_node(axis_entry, signal, signal_index, key_color))
            space_vectors.append(signal_vector)

    key_matrix = np.vstack(key_vectors).astype(np.float32) if key_vectors else np.zeros((0, 0), dtype=np.float32)
    space_matrix = np.vstack(space_vectors).astype(np.float32) if space_vectors else np.zeros((0, 0), dtype=np.float32)

    return {
        "key_vectors": key_matrix,
        "embedding_space": project_embedding_space(space_records, space_matrix, axis_entries, space_scale),
    }


def project_encoded_vectors(projection_model, axis_name, encoded_vectors, device):
    import torch

    encoded_vectors = np.asarray(encoded_vectors, dtype=np.float32)
    if encoded_vectors.size == 0:
        return encoded_vectors

    with torch.no_grad():
        encoded_tensor = torch.as_tensor(encoded_vectors, dtype=torch.float32, device=device)
        projected_tensor = projection_model(axis_name, encoded_tensor)
        projected_vectors = projected_tensor.detach().cpu().numpy().astype(np.float32)

    return projected_vectors


def empty_embedding_space():
    return {
        "nodes": [],
        "links": [],
        "groups": [],
        "note": "PCA 3D projection is unavailable because there are no embeddings.",
    }


def build_space_key_node(axis_entry, axis_node, key_color):
    return {
        "id": axis_node["id"],
        "label": axis_node["label"],
        "title": f"{axis_entry['axis']} key: {axis_entry['key']}",
        "type": "key",
        "axis": axis_entry["axis"],
        "key": axis_entry["key"],
        "parent": axis_entry["key"],
        "signal_count": len(axis_entry["signals"]),
        "size": KEY_EMBEDDING_GRAPH_KEY_SIZE,
        "color": key_color,
    }


def build_space_signal_node(axis_entry, signal, signal_index, key_color):
    return {
        "id": f"signal::{axis_entry['axis']}::{axis_entry['key']}::{signal_index:04d}",
        "label": signal,
        "title": f"{axis_entry['key']} / signal: {signal}",
        "type": "signal",
        "axis": axis_entry["axis"],
        "key": axis_entry["key"],
        "parent": axis_entry["key"],
        "size": KEY_EMBEDDING_GRAPH_SIGNAL_SIZE,
        "color": key_color,
    }


def axis_graph_color(index, total):
    hue = int((KEY_EMBEDDING_GRAPH_COLOR_HUE * (index - 1)) / max(total, 1))
    return f"hsl({hue}, {KEY_EMBEDDING_GRAPH_COLOR_SATURATION}%, {KEY_EMBEDDING_GRAPH_COLOR_LIGHTNESS}%)"


def project_embedding_space(space_records, space_matrix, axis_entries, space_scale):
    if not space_records or space_matrix.size == 0:
        return empty_embedding_space()

    centered_matrix = space_matrix - space_matrix.mean(axis=0, keepdims=True)
    _left_vectors, _singular_values, right_vectors = np.linalg.svd(centered_matrix, full_matrices=False)

    if right_vectors.shape[0] >= 3:
        components = right_vectors[:3]
    elif right_vectors.shape[0] == 2:
        components = np.vstack([right_vectors[0], right_vectors[1], np.zeros_like(right_vectors[0])])
    elif right_vectors.shape[0] == 1:
        components = np.vstack([right_vectors[0], np.zeros_like(right_vectors[0]), np.zeros_like(right_vectors[0])])
    else:
        components = np.zeros((3, space_matrix.shape[1]), dtype=np.float32)

    coordinates = centered_matrix @ components.T
    max_abs_coordinate = float(np.max(np.abs(coordinates))) if coordinates.size else 1.0
    if max_abs_coordinate <= 0:
        max_abs_coordinate = 1.0
    coordinates = (coordinates / max_abs_coordinate) * float(space_scale)

    space_nodes = []
    key_id_by_key = {}
    group_indices_by_key = {axis_entry["key"]: [] for axis_entry in axis_entries}

    for node_index, (space_record, coordinate) in enumerate(zip(space_records, coordinates)):
        node = dict(space_record)
        node["x"] = round(float(coordinate[0]), KEY_EMBEDDING_GRAPH_COORD_ROUND)
        node["y"] = round(float(coordinate[1]), KEY_EMBEDDING_GRAPH_COORD_ROUND)
        node["z"] = round(float(coordinate[2]), KEY_EMBEDDING_GRAPH_COORD_ROUND)
        space_nodes.append(node)

        if node["type"] == "key":
            key_id_by_key[node["key"]] = node["id"]
        group_indices_by_key.setdefault(node["key"], []).append(node_index)

    space_links = build_embedding_space_links(space_nodes, key_id_by_key)
    space_groups = build_embedding_space_groups(space_nodes, group_indices_by_key, axis_entries)

    return {
        "nodes": space_nodes,
        "links": space_links,
        "groups": space_groups,
        "note": "PCA 3D projection of key/signal embedding vectors.",
    }


def build_embedding_space_links(space_nodes, key_id_by_key):
    space_links = []

    for node in space_nodes:
        if node["type"] != "signal":
            continue

        key_id = key_id_by_key.get(node["key"])
        if not key_id:
            continue

        space_links.append({
            "id": f"{key_id}--{node['id']}",
            "source": key_id,
            "target": node["id"],
            "relationship": "membership",
            "parent": node["key"],
            "color": node["color"],
            "width": 0.6,
        })

    return space_links


def build_embedding_space_groups(space_nodes, group_indices_by_key, axis_entries):
    space_groups = []

    for axis_entry in axis_entries:
        key = axis_entry["key"]
        member_indices = group_indices_by_key.get(key, [])
        if not member_indices:
            continue

        member_nodes = [space_nodes[member_index] for member_index in member_indices]
        signal_nodes = [node for node in member_nodes if node["type"] == "signal"]
        shell_nodes = signal_nodes if len(signal_nodes) >= 3 else member_nodes
        shell_coordinates = np.asarray(
            [[node["x"], node["y"], node["z"]] for node in shell_nodes],
            dtype=np.float32,
        )
        centroid = shell_coordinates.mean(axis=0)
        distances = np.linalg.norm(shell_coordinates - centroid, axis=1)
        key_node = next((node for node in member_nodes if node["type"] == "key"), None)

        space_groups.append({
            "key": key,
            "color": key_node["color"] if key_node else "#334155",
            "count": len(axis_entry["signals"]),
            "x": round(float(centroid[0]), KEY_EMBEDDING_GRAPH_COORD_ROUND),
            "y": round(float(centroid[1]), KEY_EMBEDDING_GRAPH_COORD_ROUND),
            "z": round(float(centroid[2]), KEY_EMBEDDING_GRAPH_COORD_ROUND),
            "radius": round(float(distances.max()) if len(distances) else 0.0, KEY_EMBEDDING_GRAPH_DISTANCE_ROUND),
            "max_distance": round(float(distances.max()) if len(distances) else 0.0, KEY_EMBEDDING_GRAPH_DISTANCE_ROUND),
        })

    return space_groups


def entry_texts(axis_entry):
    texts = [axis_entry["key"]]
    texts.extend(axis_entry["signals"])
    return texts


def build_similarity_edges(axis_name, axis_nodes, axis_vectors, top_k, min_similarity):
    candidate_scores_by_index = [[] for _axis_node in axis_nodes]

    for left_index in range(len(axis_nodes)):
        for right_index in range(left_index + 1, len(axis_nodes)):
            similarity = calculate_similarity(axis_vectors[left_index], axis_vectors[right_index])
            if similarity < min_similarity:
                continue

            candidate_scores_by_index[left_index].append((right_index, similarity))
            candidate_scores_by_index[right_index].append((left_index, similarity))

    selected_pair_scores = {}
    for source_index, candidate_scores in enumerate(candidate_scores_by_index):
        ranked_scores = sorted(candidate_scores, key=lambda score_entry: (-score_entry[1], score_entry[0]))
        for target_index, similarity in ranked_scores[:top_k]:
            pair_key = tuple(sorted((source_index, target_index)))
            selected_pair_scores[pair_key] = max(selected_pair_scores.get(pair_key, 0), similarity)

    edges = []
    sorted_pair_scores = sorted(selected_pair_scores.items(), key=lambda pair_score: (-pair_score[1], pair_score[0]))

    for edge_index, ((left_index, right_index), similarity) in enumerate(sorted_pair_scores, start=1):
        edges.append({
            "id": f"edge::{axis_name}::{edge_index:04d}",
            "source": axis_nodes[left_index]["id"],
            "target": axis_nodes[right_index]["id"],
            "type": "embedding_similarity",
            "weight": round(similarity, 6),
        })

    return edges


def normalize_display_text(raw_text):
    return clean_text(raw_text)


def normalize_embedding_text(raw_text):
    return normalize_text(raw_text)


def current_utc_timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
