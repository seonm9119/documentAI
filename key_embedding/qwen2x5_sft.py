import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

try:
    from .schema import normalize_target_payload, ocr_page_jsons_to_model_text
except ImportError:
    from schema import normalize_target_payload, ocr_page_jsons_to_model_text


SYSTEM_PROMPT = """너는 OCR 결과에서 문서 전체 기준 4축 key/signal을 추출하는 key-embedding-graph 모델이다.
출력은 반드시 JSON 객체 하나만 작성한다.
축은 subject, document_type, business_domain, modifier 네 개를 모두 포함한다.
각 축은 key와 signals를 가진다.
key는 문서를 대표하는 짧고 안정적인 개념이다.
signals는 입력 OCR 텍스트에 실제로 등장한 근거 표현만 사용한다.
근거가 약하거나 판단이 어려운 축은 key를 "unknown"으로 두고 signals는 빈 배열로 둔다.
문서가 여러 페이지여도 최종 결과는 문서 단위 JSON 하나다."""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--output-dir", default="output/key-embedding-graph-qwen2x5")
    parser.add_argument("--base-model", default=os.environ.get("KEY_EMBEDDING_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    parser.add_argument("--max-seq-length", default=4096, type=int)
    parser.add_argument("--epochs", default=3, type=float)
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
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer, model = load_qwen_model_and_tokenizer(args)
    full_dataset = load_sft_dataset(args.train_data, tokenizer)
    train_dataset, eval_dataset = split_train_eval_dataset(full_dataset, args.eval_ratio)

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
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def load_qwen_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

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
    return tokenizer, model


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
    deepseek_pages = source_record.get("deepseek_pages")
    paddle_pages = source_record.get("paddle_pages")
    if not isinstance(deepseek_pages, list) or not isinstance(paddle_pages, list):
        raise ValueError(f"{line_number}번째 record에 deepseek_pages와 paddle_pages list가 필요합니다.")

    return ocr_page_jsons_to_model_text(deepseek_pages, paddle_pages)


def build_user_prompt(model_input_text):
    return (
        "다음 OCR 결과를 보고 문서 전체 기준 4축 key/signal JSON 하나만 출력하세요.\n\n"
        f"{model_input_text}"
    )


def split_train_eval_dataset(full_dataset, eval_ratio):
    if eval_ratio <= 0 or len(full_dataset) < 2:
        return full_dataset, None

    split_dataset = full_dataset.train_test_split(test_size=eval_ratio, seed=17)
    return split_dataset["train"], split_dataset["test"]


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
    evaluation_strategy = "steps" if eval_dataset is not None else "no"
    return TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.save_steps,
        save_total_limit=2,
        evaluation_strategy=evaluation_strategy,
        save_strategy="steps",
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to=[],
    )


if __name__ == "__main__":
    main()
