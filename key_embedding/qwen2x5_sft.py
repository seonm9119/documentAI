import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset, disable_progress_bar
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback, TrainerCallback, TrainingArguments
from trl import SFTTrainer

try:
    from . import config
    from .schema import normalize_target_payload, paddle_page_jsons_to_model_text
except ImportError:
    import config
    from schema import normalize_target_payload, paddle_page_jsons_to_model_text


SYSTEM_PROMPT = """You are the key-embedding-graph model.
Extract document-level key/signal values from OCR text.
Return exactly one JSON object and no extra text.
The JSON object must include four axes: subject, document_type, business_domain, and modifier.
Each axis must contain a key and signals.
The key must be a short, stable concept that represents the document.
Signals must be evidence phrases that actually appear in the input OCR text.
If evidence is weak or an axis cannot be determined, set key to "unknown" and signals to an empty array.
Even when the document has multiple pages, return one final document-level JSON object."""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR)
    parser.add_argument("--base-model", default=os.environ.get("KEY_EMBEDDING_BASE_MODEL", config.BASE_MODEL))
    parser.add_argument("--max-seq-length", default=config.MAX_SEQ_LENGTH, type=int)
    parser.add_argument("--epochs", default=100, type=float)
    parser.add_argument("--learning-rate", default=2e-4, type=float)
    parser.add_argument("--eval-ratio", default=0.05, type=float)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--grad-accum", default=16, type=int)
    parser.add_argument("--lora-r", default=8, type=int)
    parser.add_argument("--lora-alpha", default=16, type=int)
    parser.add_argument("--lora-dropout", default=0.05, type=float)
    parser.add_argument("--target-modules", default="q_proj,v_proj")
    parser.add_argument("--save-steps", default=100, type=int)
    parser.add_argument("--logging-steps", default=10, type=int)
    parser.add_argument("--early-stopping-patience", default=5, type=int)
    parser.add_argument("--early-stopping-threshold", default=3e-4, type=float)
    parser.add_argument("--eval-accumulation-steps", default=1, type=int)
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


def main():
    disable_progress_bar()
    args = parse_args()
    tokenizer = load_qwen_tokenizer(args)
    full_dataset = load_sft_dataset(args.train_data, tokenizer)
    train_dataset, eval_dataset = split_train_eval_dataset(full_dataset, args.eval_ratio)
    model = load_qwen_model(args)

    lora_config = build_lora_config(args)
    training_args = build_training_args(args, eval_dataset)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
        compute_metrics=compute_token_accuracy if eval_dataset is not None else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if eval_dataset is not None else None,
        callbacks=build_callbacks(args, eval_dataset),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def load_qwen_model_and_tokenizer(args):
    tokenizer = load_qwen_tokenizer(args)
    model = load_qwen_model(args)
    return tokenizer, model


def load_qwen_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_qwen_model(args):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        device_map=args.device_map,
        trust_remote_code=False,
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    return model


def load_sft_dataset(train_data_path, tokenizer):
    train_records = []
    source_records = load_source_records(train_data_path)
    for line_number, source_record in enumerate(source_records, start=1):
        train_records.append({
            "text": build_sft_text(source_record, tokenizer, line_number),
        })

    if not train_records:
        raise ValueError("학습 데이터가 비어 있습니다.")
    return Dataset.from_list(train_records)


def load_source_records(train_data_path):
    train_data_path = Path(train_data_path)
    if train_data_path.is_dir():
        source_records = []
        for record_path in sorted(train_data_path.glob("*.json")):
            source_records.append(json.loads(record_path.read_text(encoding="utf-8")))
        return source_records

    if train_data_path.suffix.lower() == ".json":
        source_records = json.loads(train_data_path.read_text(encoding="utf-8"))
        if isinstance(source_records, dict):
            source_records = source_records.get("records") or source_records.get("data")
            if source_records is None:
                source_records = [json.loads(train_data_path.read_text(encoding="utf-8"))]
        if not isinstance(source_records, list):
            raise ValueError("json 학습 데이터는 record list이거나 records/data list를 가진 객체여야 합니다.")
        return source_records

    source_records = []
    for raw_line in train_data_path.read_text(encoding="utf-8").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        source_records.append(json.loads(raw_line))
    return source_records


def build_sft_text(source_record, tokenizer, line_number):
    target_payload = get_target_payload(source_record, line_number)
    model_input_text = build_model_input_text(source_record, line_number)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(model_input_text)},
        {"role": "assistant", "content": json.dumps(target_payload, ensure_ascii=False, indent=2)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def get_target_payload(source_record, line_number):
    raw_target = source_record.get("target") or source_record.get("output")
    if not isinstance(raw_target, dict):
        raise ValueError(f"{line_number}번째 record에 target 또는 output 객체가 필요합니다.")
    return normalize_target_payload(raw_target)


def build_model_input_text(source_record, line_number):
    paddle_pages = source_record.get("paddle_pages")
    if not isinstance(paddle_pages, list):
        raise ValueError(f"{line_number}번째 record에 paddle_pages list가 필요합니다.")

    return paddle_page_jsons_to_model_text(paddle_pages)


def build_user_prompt(model_input_text):
    return (
        "Read the following OCR text and return one document-level four-axis key/signal JSON object.\n\n"
        f"{model_input_text}"
    )


def split_train_eval_dataset(full_dataset, eval_ratio):
    if eval_ratio <= 0 or len(full_dataset) < 2:
        return full_dataset, None

    split_dataset = full_dataset.train_test_split(test_size=eval_ratio, seed=17)
    return split_dataset["train"], split_dataset["test"]


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.argmax(logits, dim=-1)


def compute_token_accuracy(eval_prediction):
    predictions, labels = eval_prediction
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    shifted_predictions = predictions[:, :-1]
    shifted_labels = labels[:, 1:]
    label_mask = shifted_labels != -100
    label_count = label_mask.sum()

    if label_count == 0:
        return {"token_accuracy": 0.0}

    correct_count = (shifted_predictions[label_mask] == shifted_labels[label_mask]).sum()
    return {"token_accuracy": float(correct_count) / float(label_count)}


class EpochProgressCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        current_epoch = int(state.epoch or 0) + 1
        total_epochs = int(args.num_train_epochs)
        print(f"[epoch {current_epoch}/{total_epochs}] train begin", flush=True)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return

        metric_parts = []
        for metric_name in ["eval_loss", "eval_token_accuracy"]:
            if metric_name in metrics:
                metric_parts.append(f"{metric_name}={metrics[metric_name]:.6f}")
        if metric_parts:
            print(f"[epoch {state.epoch:.2f}] validation {' '.join(metric_parts)}", flush=True)


def build_callbacks(args, eval_dataset):
    callbacks = [EpochProgressCallback()]
    if eval_dataset is None or args.early_stopping_patience <= 0:
        return callbacks
    callbacks.append(EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
    ))
    return callbacks


def build_lora_config(args):
    target_modules = [
        target_module.strip()
        for target_module in args.target_modules.split(",")
        if target_module.strip()
    ]
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )


def build_training_args(args, eval_dataset):
    evaluation_strategy = "epoch" if eval_dataset is not None else "no"
    save_strategy = "epoch" if eval_dataset is not None else "steps"
    load_best_model = eval_dataset is not None
    return TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_strategy="epoch",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        evaluation_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        load_best_model_at_end=load_best_model,
        metric_for_best_model="eval_loss" if load_best_model else None,
        greater_is_better=False if load_best_model else None,
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        eval_accumulation_steps=args.eval_accumulation_steps,
        remove_unused_columns=True,
        disable_tqdm=True,
        report_to=[],
    )


if __name__ == "__main__":
    main()
