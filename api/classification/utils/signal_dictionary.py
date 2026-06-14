import json
import random
import re
import unicodedata
from pathlib import Path

import numpy as np

try:
    from ..config import (
        KEY_SIGNAL_AXIS_SOURCES,
        SIGNAL_NORMALIZE_INSIDE_SCORE,
        SIGNAL_NORMALIZE_KEY_SCORE,
        SIGNAL_NORMALIZE_MARGIN,
        SIGNAL_NORMALIZE_MAX_OTHER_SCORE,
        SIGNAL_NORMALIZE_SCORE_ROUND,
    )
except ImportError:
    from api.classification.config import (
        KEY_SIGNAL_AXIS_SOURCES,
        SIGNAL_NORMALIZE_INSIDE_SCORE,
        SIGNAL_NORMALIZE_KEY_SCORE,
        SIGNAL_NORMALIZE_MARGIN,
        SIGNAL_NORMALIZE_MAX_OTHER_SCORE,
        SIGNAL_NORMALIZE_SCORE_ROUND,
    )


def clean_text(value):
    return " ".join(str(value or "").replace("\r", "\n").split()).strip()


def clean_text_list(values):
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []

    clean_values = []
    seen_values = set()

    for raw_text in values:
        clean_value = clean_text(raw_text)
        normalized_value = normalize_text(clean_value)
        if not clean_value or normalized_value in seen_values:
            continue

        clean_values.append(clean_value)
        seen_values.add(normalized_value)

    return clean_values


def normalize_text(value):
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.strip().lower()
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = text.replace(":", " ")
    text = text.replace("：", " ")
    text = text.replace("|", " ")
    text = text.replace("_", " ")
    text = re.sub(r'^[\s,;:/|()\[\]{}<>"]+|[\s,;:/|()\[\]{}<>"]+$', "", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_vector(vector):
    vector = np.asarray(vector, dtype=np.float32)
    vector_norm = np.linalg.norm(vector)
    if vector_norm <= 0:
        return vector
    return vector / vector_norm


def load_axis_entries(dictionary_dir, axis_name):
    axis_source = KEY_SIGNAL_AXIS_SOURCES[axis_name]
    source_path = Path(dictionary_dir) / axis_source["file_name"]
    source_records = json.loads(source_path.read_text(encoding="utf-8"))
    key_field = axis_source["key_field"]
    entries = []
    seen_keys = set()

    for source_record in source_records:
        if not isinstance(source_record, dict):
            continue

        canonical_key = clean_text(source_record.get(key_field))
        if not canonical_key or canonical_key in seen_keys:
            continue

        entries.append({
            "axis": axis_name,
            "key": canonical_key,
            "signals": clean_text_list(source_record.get("signals") or []),
        })
        seen_keys.add(canonical_key)

    return entries


def load_dictionary_entries(dictionary_dir):
    return {
        axis_name: load_axis_entries(dictionary_dir, axis_name)
        for axis_name in KEY_SIGNAL_AXIS_SOURCES
    }


def build_axis_samples(axis_entries, include_key=True):
    samples = []

    for entry_index, axis_entry in enumerate(axis_entries):
        texts = []
        if include_key:
            texts.append({"text": axis_entry["key"], "text_type": "canonical_key"})
        texts.extend({"text": signal, "text_type": "signal"} for signal in axis_entry["signals"])

        seen_texts = set()
        for text_index, text_payload in enumerate(texts):
            normalized_text = normalize_text(text_payload["text"])
            if not normalized_text or normalized_text in seen_texts:
                continue

            samples.append({
                "axis": axis_entry["axis"],
                "key": axis_entry["key"],
                "text": clean_text(text_payload["text"]),
                "text_norm": normalized_text,
                "text_type": text_payload["text_type"],
                "entry_index": entry_index,
                "text_index": text_index,
            })
            seen_texts.add(normalized_text)

    return samples


def validate_axis_samples(axis_samples):
    collision_map = {}
    for sample in axis_samples:
        collision_map.setdefault(sample["text_norm"], set()).add(sample["key"])

    collisions = [
        {"text": text_norm, "keys": sorted(keys)}
        for text_norm, keys in collision_map.items()
        if len(keys) > 1
    ]
    return {
        "sample_count": len(axis_samples),
        "key_count": len({sample["key"] for sample in axis_samples}),
        "collision_count": len(collisions),
        "collisions": collisions,
    }


def filter_colliding_signals(axis_samples):
    keys_by_text = {}
    for sample in axis_samples:
        keys_by_text.setdefault(sample["text_norm"], set()).add(sample["key"])

    colliding_texts = {
        text_norm
        for text_norm, keys in keys_by_text.items()
        if len(keys) > 1
    }
    return [
        sample
        for sample in axis_samples
        if sample["text_type"] != "signal" or sample["text_norm"] not in colliding_texts
    ]


def split_axis_samples(axis_samples, valid_ratio, seed):
    rows = list(axis_samples)
    random.Random(seed).shuffle(rows)
    valid_count = int(round(len(rows) * float(valid_ratio)))
    if valid_count <= 0 or valid_count >= len(rows):
        return {"train": rows, "valid": []}
    return {
        "train": rows[valid_count:],
        "valid": rows[:valid_count],
    }


def build_label_options(axis_entries):
    return [axis_entry["key"] for axis_entry in axis_entries]


def attach_embeddings(axis_samples, embeddings, label_options):
    label_id_by_key = {key: label_id for label_id, key in enumerate(label_options)}
    rows = []

    for sample_index, sample in enumerate(axis_samples):
        label_id = label_id_by_key.get(sample["key"])
        if label_id is None:
            continue

        row = dict(sample)
        row["embedding"] = np.asarray(embeddings[sample_index], dtype=np.float32)
        row["label_id"] = label_id
        rows.append(row)

    return rows


def build_hard_negative_map(rows, top_k):
    key_vectors = {}
    for key, key_rows in rows_by_key(rows).items():
        vectors = np.vstack([row["embedding"] for row in key_rows]).astype(np.float32)
        key_vectors[key] = normalize_vector(vectors.mean(axis=0))

    keys = list(key_vectors)
    hard_map = {}
    for key in keys:
        scores = []
        for other_key in keys:
            if other_key == key:
                continue
            scores.append((other_key, float(key_vectors[key] @ key_vectors[other_key])))
        scores.sort(key=lambda score_entry: (-score_entry[1], score_entry[0]))
        hard_map[key] = [other_key for other_key, _score in scores[:int(top_k)]]

    return hard_map


def iter_balanced_batches(rows, batch_size, samples_per_key, seed, shuffle=True):
    grouped_rows = rows_by_key(rows)
    keys = list(grouped_rows)
    if shuffle:
        random.Random(seed).shuffle(keys)

    keys_per_batch = max(1, int(batch_size) // max(1, int(samples_per_key)))
    for start in range(0, len(keys), keys_per_batch):
        batch_rows = []
        for key in keys[start:start + keys_per_batch]:
            key_rows = list(grouped_rows[key])
            if shuffle:
                random.Random(seed + start + len(batch_rows)).shuffle(key_rows)
            batch_rows.extend(key_rows[:max(1, int(samples_per_key))])

        if batch_rows:
            yield batch_rows


def rows_by_key(rows):
    grouped_rows = {}
    for row in rows:
        grouped_rows.setdefault(row["key"], []).append(row)
    return grouped_rows


def find_inside_signal_decisions(axis_entries, axis_vectors, thresholds):
    prototypes = build_key_prototypes(axis_entries, axis_vectors)
    remove_decisions = []

    for entry_index, axis_entry in enumerate(axis_entries):
        entry_vectors = axis_vectors[entry_index]
        signal_vectors = entry_vectors["signal_vectors"]

        for signal_index, signal_entry in enumerate(axis_entry["signals"]):
            signal_vector = signal_vectors[signal_index]
            own_score = calculate_own_score(entry_vectors, signal_index, signal_vector)
            key_score = calculate_similarity(signal_vector, entry_vectors["key_vector"])
            best_other_key, best_other_score = find_best_other_key(axis_entry["key"], signal_vector, prototypes)
            margin = own_score - best_other_score

            if not is_redundant_inside_signal(own_score, key_score, best_other_score, margin, thresholds):
                continue

            remove_decisions.append({
                "key": axis_entry["key"],
                "signal": signal_entry["text"],
                "signal_norm": signal_entry["text_norm"],
                "own_score": round(float(own_score), SIGNAL_NORMALIZE_SCORE_ROUND),
                "key_score": round(float(key_score), SIGNAL_NORMALIZE_SCORE_ROUND),
                "best_other_key": best_other_key,
                "best_other_score": round(float(best_other_score), SIGNAL_NORMALIZE_SCORE_ROUND),
                "margin": round(float(margin), SIGNAL_NORMALIZE_SCORE_ROUND),
                "reason": "redundant_inside_key_area",
            })

    return remove_decisions


def build_key_prototypes(axis_entries, axis_vectors):
    prototypes = {}

    for axis_entry, entry_vectors in zip(axis_entries, axis_vectors):
        key_and_signal_vectors = [entry_vectors["key_vector"]]
        key_and_signal_vectors.extend(entry_vectors["signal_vectors"])
        prototypes[axis_entry["key"]] = normalize_vector(np.vstack(key_and_signal_vectors).mean(axis=0))

    return prototypes


def calculate_own_score(entry_vectors, signal_index, signal_vector):
    remaining_vectors = [entry_vectors["key_vector"]]

    for current_index, current_vector in enumerate(entry_vectors["signal_vectors"]):
        if current_index != signal_index:
            remaining_vectors.append(current_vector)

    leave_one_out_prototype = normalize_vector(np.vstack(remaining_vectors).mean(axis=0))
    return calculate_similarity(signal_vector, leave_one_out_prototype)


def find_best_other_key(own_key, signal_vector, prototypes):
    best_other_key = ""
    best_other_score = -1.0

    for key, prototype in prototypes.items():
        if key == own_key:
            continue

        score = calculate_similarity(signal_vector, prototype)
        if score > best_other_score:
            best_other_key = key
            best_other_score = score

    return best_other_key, best_other_score


def is_redundant_inside_signal(own_score, key_score, best_other_score, margin, thresholds):
    return (
        own_score >= float(thresholds["inside_score"])
        and key_score >= float(thresholds["key_score"])
        and best_other_score <= float(thresholds["max_other_score"])
        and margin >= float(thresholds["margin"])
    )


def calculate_similarity(left_vector, right_vector):
    return float(np.asarray(left_vector, dtype=np.float32) @ np.asarray(right_vector, dtype=np.float32))
