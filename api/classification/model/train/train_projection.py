import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from ...config import (
    KEY_SIGNAL_AXIS_SOURCES,
    PROJECTION_CACHE_FOLDER,
    PROJECTION_DEVICE,
    PROJECTION_ENCODER_MODEL,
    PROJECTION_MODEL_CONFIG,
    PROJECTION_MODEL_PATH,
    PROJECTION_PRECISION,
    PROJECTION_TRAIN_DICTIONARY_DIR,
    PROJECTION_TRAIN_CONFIG,
)
from ...utils.signal_dictionary import (
    attach_embeddings,
    build_axis_samples,
    build_hard_negative_map,
    build_label_options,
    filter_colliding_signals,
    iter_balanced_batches,
    load_dictionary_entries,
    split_axis_samples,
    validate_axis_samples,
)
from ...utils.embedding_encoder import encode_texts, ensure_cuda_device, load_encoder
from .losses import total_loss
from .network import KeyEmbeddingProjectionModel, save_projection_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train key-signal projection heads for dictionary quality gates.")
    parser.add_argument("--dictionary-dir", default=str(PROJECTION_TRAIN_DICTIONARY_DIR))
    parser.add_argument("--output", default=str(PROJECTION_MODEL_PATH))
    parser.add_argument("--encoder-model", default=PROJECTION_ENCODER_MODEL)
    parser.add_argument("--cache-folder", default=PROJECTION_CACHE_FOLDER)
    parser.add_argument("--device", default=PROJECTION_DEVICE)
    parser.add_argument("--precision", default=PROJECTION_PRECISION)
    parser.add_argument("--epochs", default=PROJECTION_TRAIN_CONFIG["epochs"], type=int)
    parser.add_argument("--batch-size", default=PROJECTION_TRAIN_CONFIG["batch_size"], type=int)
    parser.add_argument("--samples-per-key", default=PROJECTION_TRAIN_CONFIG["samples_per_key"], type=int)
    parser.add_argument("--lr", default=PROJECTION_TRAIN_CONFIG["lr"], type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = train_projection_model(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def train_projection_model(args):
    set_seed(PROJECTION_TRAIN_CONFIG["seed"])
    ensure_cuda_device(args.device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dictionary_entries = load_dictionary_entries(args.dictionary_dir)
    encoder = load_encoder(args.encoder_model, args.device, args.cache_folder, args.precision)
    axis_payloads = build_training_payloads(dictionary_entries, encoder, args)

    model = KeyEmbeddingProjectionModel(KEY_SIGNAL_AXIS_SOURCES.keys(), PROJECTION_MODEL_CONFIG).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=PROJECTION_TRAIN_CONFIG["weight_decay"])

    started_at = time.perf_counter()
    history = []
    best_valid_loss = None
    best_state = None

    for epoch in range(1, int(args.epochs) + 1):
        train_loss = run_epoch(model, optimizer, axis_payloads, args, epoch)
        valid_loss = evaluate_model(model, axis_payloads, args)
        score = valid_loss if valid_loss is not None else train_loss

        if best_valid_loss is None or score < best_valid_loss:
            best_valid_loss = score
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

        history.append({
            "epoch": epoch,
            "train_loss": round(float(train_loss), 6),
            "valid_loss": None if valid_loss is None else round(float(valid_loss), 6),
        })

        if epoch == 1 or epoch % 10 == 0 or epoch == int(args.epochs):
            print(f"epoch={epoch} train_loss={train_loss:.5f} valid_loss={format_loss(valid_loss)}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    metadata = {
        "model": "key_embedding_projection",
        "encoder_model": args.encoder_model,
        "device": args.device,
        "precision": args.precision,
        "dictionary_dir": str(args.dictionary_dir),
        "model_config": dict(PROJECTION_MODEL_CONFIG),
        "train_config": {
            **dict(PROJECTION_TRAIN_CONFIG),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "samples_per_key": int(args.samples_per_key),
            "lr": float(args.lr),
        },
        "axis_summary": {
            axis: payload["summary"]
            for axis, payload in axis_payloads.items()
        },
        "best_valid_loss": best_valid_loss,
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "history": history,
    }
    save_projection_model(output_path, model, metadata)
    metadata["checkpoint_path"] = str(output_path)
    return metadata


def build_training_payloads(dictionary_entries, encoder, args):
    axis_payloads = {}

    for axis, axis_entries in dictionary_entries.items():
        axis_samples = build_axis_samples(axis_entries, include_key=True)
        source_report = validate_axis_samples(axis_samples)
        training_samples = filter_colliding_signals(axis_samples)

        texts = [sample["text"] for sample in training_samples]
        embeddings = encode_texts(encoder, texts, args.batch_size, PROJECTION_MODEL_CONFIG["input_dim"])
        label_options = build_label_options(axis_entries)
        rows = attach_embeddings(training_samples, embeddings, label_options)
        split = split_axis_samples(rows, PROJECTION_TRAIN_CONFIG["valid_ratio"], PROJECTION_TRAIN_CONFIG["seed"])
        hard_map = build_hard_negative_map(rows, PROJECTION_TRAIN_CONFIG["hard_negative_top"])

        axis_payloads[axis] = {
            "label_options": label_options,
            "split": split,
            "hard_map": hard_map,
            "summary": {
                **source_report,
                "training_sample_count": len(training_samples),
                "excluded_collision_signal_count": len(axis_samples) - len(training_samples),
                "train_count": len(split["train"]),
                "valid_count": len(split["valid"]),
            },
        }

    return axis_payloads


def run_epoch(model, optimizer, axis_payloads, args, epoch):
    model.train()
    losses = []

    for axis, axis_payload in axis_payloads.items():
        rows = axis_payload["split"]["train"]
        batches = iter_balanced_batches(
            rows,
            args.batch_size,
            args.samples_per_key,
            PROJECTION_TRAIN_CONFIG["seed"] + epoch,
            shuffle=True,
        )
        for batch_rows in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(model, axis, batch_rows, axis_payload["hard_map"], args.device)
            loss.backward()
            if PROJECTION_TRAIN_CONFIG["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), PROJECTION_TRAIN_CONFIG["grad_clip"])
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses)) if losses else 0.0


def evaluate_model(model, axis_payloads, args):
    model.eval()
    losses = []

    with torch.no_grad():
        for axis, axis_payload in axis_payloads.items():
            rows = axis_payload["split"]["valid"]
            if not rows:
                continue

            batches = iter_balanced_batches(
                rows,
                args.batch_size,
                args.samples_per_key,
                PROJECTION_TRAIN_CONFIG["seed"],
                shuffle=False,
            )
            for batch_rows in batches:
                loss = batch_loss(model, axis, batch_rows, axis_payload["hard_map"], args.device)
                losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses)) if losses else None


def batch_loss(model, axis, batch_rows, hard_map, device):
    embeddings = torch.as_tensor(np.vstack([row["embedding"] for row in batch_rows]), dtype=torch.float32, device=device)
    labels = torch.as_tensor([row["label_id"] for row in batch_rows], dtype=torch.long, device=device)
    keys = [row["key"] for row in batch_rows]
    features = model(axis, embeddings)
    loss, _parts = total_loss(features, labels, keys, hard_map)
    return loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_loss(value):
    if value is None:
        return "none"
    return f"{float(value):.5f}"


if __name__ == "__main__":
    raise SystemExit(main())
