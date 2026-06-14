import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from datasets import Dataset, disable_progress_bar
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

try:
    from ... import config
    from ...utils.signal_dictionary import clean_text, clean_text_list
except ImportError:
    from api.classification import config
    from api.classification.utils.signal_dictionary import clean_text, clean_text_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", default=str(config.QWEN_TRAIN_DATA_PATH))
    parser.add_argument("--val-data", default=str(config.QWEN_VAL_DATA_PATH))
    parser.add_argument("--output-dir", default=config.QWEN_OUTPUT_DIR)
    parser.add_argument("--base-model", default=os.environ.get("KEY_EMBEDDING_BASE_MODEL", config.QWEN_BASE_MODEL))
    parser.add_argument("--max-seq-length", default=config.QWEN_MAX_SEQ_LENGTH, type=int)
    parser.add_argument("--learning-rate", default=config.QWEN_SFT_TRAIN_CONFIG["learning_rate"], type=float)
    parser.add_argument("--eval-ratio", default=config.QWEN_VAL_RATIO, type=float)
    parser.add_argument("--batch-size", default=config.QWEN_SFT_TRAIN_CONFIG["batch_size"], type=int)
    parser.add_argument("--grad-accum", default=config.QWEN_SFT_TRAIN_CONFIG["grad_accum"], type=int)
    parser.add_argument("--lora-r", default=config.QWEN_LORA_CONFIG["r"], type=int)
    parser.add_argument("--lora-alpha", default=config.QWEN_LORA_CONFIG["alpha"], type=int)
    parser.add_argument("--lora-dropout", default=config.QWEN_LORA_CONFIG["dropout"], type=float)
    parser.add_argument("--target-modules", default=config.QWEN_LORA_CONFIG["target_modules"])
    parser.add_argument("--save-steps", default=config.QWEN_SFT_TRAIN_CONFIG["save_steps"], type=int)
    parser.add_argument("--logging-steps", default=config.QWEN_SFT_TRAIN_CONFIG["logging_steps"], type=int)
    parser.add_argument(
        "--early-stopping-patience",
        default=config.QWEN_SFT_TRAIN_CONFIG["early_stopping_patience"],
        type=int,
    )
    parser.add_argument(
        "--early-stopping-threshold",
        default=config.QWEN_SFT_TRAIN_CONFIG["early_stopping_threshold"],
        type=float,
    )
    parser.add_argument(
        "--eval-accumulation-steps",
        default=config.QWEN_SFT_TRAIN_CONFIG["eval_accumulation_steps"],
        type=int,
    )
    parser.add_argument("--device-map", default=config.QWEN_SFT_TRAIN_CONFIG["device_map"])
    return parser.parse_args()


def main():
    disable_progress_bar()
    args = parse_args()
    prepare_existing_best_adapter(args)
    tokenizer = load_qwen_tokenizer(args)
    train_dataset = load_sft_dataset(args.train_data, tokenizer)
    eval_dataset = load_validation_dataset(args.val_data, tokenizer)
    if eval_dataset is None:
        train_dataset, eval_dataset = split_train_eval_dataset(train_dataset, args.eval_ratio)
    model = load_qwen_model(args)

    lora_config = None if isinstance(model, PeftModel) else build_lora_config(args)
    training_args = build_training_args(args)

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
    )

    train_until_early_stop(trainer, tokenizer, args, eval_dataset)


def load_qwen_model_and_tokenizer(args):
    tokenizer = load_qwen_tokenizer(args)
    model = load_qwen_model(args)
    return tokenizer, model


def load_qwen_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=config.QWEN_TOKENIZER_CONFIG["trust_remote_code"],
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = config.QWEN_TOKENIZER_CONFIG["padding_side"]
    return tokenizer


def load_qwen_model(args):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=config.QWEN_QUANT_CONFIG["load_in_4bit"],
        bnb_4bit_quant_type=config.QWEN_QUANT_CONFIG["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=get_torch_dtype(config.QWEN_QUANT_CONFIG["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=config.QWEN_QUANT_CONFIG["bnb_4bit_use_double_quant"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        device_map=args.device_map,
        trust_remote_code=config.QWEN_MODEL_LOAD_CONFIG["trust_remote_code"],
        torch_dtype=get_torch_dtype(config.QWEN_MODEL_LOAD_CONFIG["torch_dtype"]),
    )
    model.config.use_cache = config.QWEN_MODEL_LOAD_CONFIG["use_cache"]
    model = prepare_model_for_kbit_training(model)
    adapter_path = resolve_existing_adapter_path(args)
    if adapter_path is not None:
        print(f"adapter load: {adapter_path}", flush=True)
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    return model


def get_torch_dtype(dtype_name):
    return getattr(torch, str(dtype_name))


def prepare_existing_best_adapter(args):
    output_dir = Path(args.output_dir)
    best_adapter_dir = output_dir / "best"
    if adapter_checkpoint_exists(best_adapter_dir):
        return
    if not adapter_checkpoint_exists(output_dir):
        return

    copy_adapter_checkpoint(output_dir, best_adapter_dir)
    print(f"adapter migrate: {output_dir} -> {best_adapter_dir}", flush=True)


def resolve_existing_adapter_path(args):
    output_dir = Path(args.output_dir)
    best_adapter_dir = output_dir / "best"
    if adapter_checkpoint_exists(best_adapter_dir):
        return best_adapter_dir
    if adapter_checkpoint_exists(output_dir):
        return output_dir
    return None


def adapter_checkpoint_exists(adapter_dir):
    adapter_dir = Path(adapter_dir)
    return (adapter_dir / "adapter_config.json").exists() and (adapter_dir / "adapter_model.safetensors").exists()


def copy_adapter_checkpoint(source_dir, target_dir):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for source_path in source_dir.iterdir():
        if source_path.is_file():
            shutil.copy2(source_path, target_dir / source_path.name)


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


def load_validation_dataset(val_data_path, tokenizer):
    if not val_data_path:
        return None

    val_data_path = Path(val_data_path)
    if not val_data_path.exists():
        print(f"validation data not found, split train dataset instead: {val_data_path}", flush=True)
        return None

    return load_sft_dataset(val_data_path, tokenizer)


def load_source_records(train_data_path):
    train_data_path = Path(train_data_path)
    if not train_data_path.exists():
        raise FileNotFoundError(f"학습 데이터 파일/폴더를 찾을 수 없습니다: {train_data_path}")

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


def paddle_page_jsons_to_model_text(paddle_pages):
    paddle_page_payloads = load_json_pages(paddle_pages, "paddle_pages")
    page_texts = []
    for paddle_page_payload in paddle_page_payloads:
        page_texts.append(extract_paddle_text(paddle_page_payload))
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
        return json.loads(Path(json_page).read_text(encoding="utf-8"))
    if isinstance(json_page, (dict, list)):
        return json_page
    raise ValueError("json page는 file path, dict, list 중 하나여야 합니다.")


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
        return clean_text_lines(rec_texts)
    return clean_block_text(paddle_result.get("text"))


def normalize_target_payload(raw_target):
    raw_target = raw_target if isinstance(raw_target, dict) else {}
    return {
        axis: normalize_axis_target(raw_target.get(axis))
        for axis in config.KEY_SIGNAL_AXES
    }


def normalize_axis_target(raw_axis_target):
    raw_axis_target = raw_axis_target if isinstance(raw_axis_target, dict) else {}
    axis_key = clean_text(raw_axis_target.get("key")) or config.KEY_SIGNAL_UNKNOWN_KEY
    signals = clean_text_list(raw_axis_target.get("signals"))
    if axis_key == config.KEY_SIGNAL_UNKNOWN_KEY:
        signals = []
    return {
        "key": axis_key,
        "signals": signals,
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


def clean_block_text(value):
    lines = []
    for line in str(value or "").replace("\r", "\n").split("\n"):
        clean_line = " ".join(line.split()).strip()
        if clean_line:
            lines.append(clean_line)
    return "\n".join(lines).strip()


def clean_text_lines(values):
    lines = []
    for value in values:
        clean_value = clean_text(value)
        if clean_value:
            lines.append(clean_value)
    return "\n".join(lines).strip()


def build_sft_text(source_record, tokenizer, line_number):
    target_payload = get_target_payload(source_record, line_number)
    model_input_text = build_model_input_text(source_record, line_number)

    messages = [
        {"role": "system", "content": config.QWEN_SYSTEM_PROMPT},
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

    split_dataset = full_dataset.train_test_split(test_size=eval_ratio, seed=config.QWEN_RANDOM_SEED)
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


def train_until_early_stop(trainer, tokenizer, args, eval_dataset):
    if eval_dataset is None:
        raise ValueError("early stop 학습에는 validation dataset이 필요합니다.")

    best_eval_loss = None
    stale_epoch_count = 0
    epoch_number = 0

    while True:
        epoch_number += 1
        print(f"[epoch {epoch_number}] train begin", flush=True)
        trainer.train()

        metrics = trainer.evaluate()
        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            raise ValueError("validation eval_loss를 계산하지 못했습니다.")

        metric_parts = [f"eval_loss={eval_loss:.6f}"]
        if "eval_token_accuracy" in metrics:
            metric_parts.append(f"eval_token_accuracy={metrics['eval_token_accuracy']:.6f}")

        if is_eval_loss_improved(best_eval_loss, eval_loss, args.early_stopping_threshold):
            best_eval_loss = eval_loss
            stale_epoch_count = 0
            save_best_adapter(trainer, tokenizer, args)
            metric_parts.append("best=true")
        else:
            stale_epoch_count += 1
            metric_parts.append(f"best=false stale={stale_epoch_count}/{args.early_stopping_patience}")

        print(f"[epoch {epoch_number}] validation {' '.join(metric_parts)}", flush=True)
        if stale_epoch_count >= args.early_stopping_patience:
            print(
                f"early stop: best_eval_loss={best_eval_loss:.6f} "
                f"threshold={args.early_stopping_threshold} "
                f"patience={args.early_stopping_patience}",
                flush=True,
            )
            return


def is_eval_loss_improved(best_eval_loss, eval_loss, early_stopping_threshold):
    if best_eval_loss is None:
        return True
    return best_eval_loss - eval_loss >= early_stopping_threshold


def save_best_adapter(trainer, tokenizer, args):
    output_dir = Path(args.output_dir)
    best_adapter_dir = output_dir / "best"
    previous_best_adapter_dir = output_dir / "previous_best"
    temporary_best_adapter_dir = output_dir / "best_tmp"

    if temporary_best_adapter_dir.exists():
        shutil.rmtree(temporary_best_adapter_dir)

    trainer.save_model(str(temporary_best_adapter_dir))
    tokenizer.save_pretrained(temporary_best_adapter_dir)

    if previous_best_adapter_dir.exists():
        shutil.rmtree(previous_best_adapter_dir)
    if best_adapter_dir.exists():
        shutil.move(str(best_adapter_dir), str(previous_best_adapter_dir))
    shutil.move(str(temporary_best_adapter_dir), str(best_adapter_dir))


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
        bias=config.QWEN_LORA_CONFIG["bias"],
        task_type=config.QWEN_LORA_CONFIG["task_type"],
        target_modules=target_modules,
    )


def build_training_args(args):
    training_arguments = dict(config.QWEN_TRAINING_ARGUMENTS)
    training_arguments.update(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
    )
    return TrainingArguments(**training_arguments)


if __name__ == "__main__":
    main()
