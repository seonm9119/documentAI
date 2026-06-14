import os
from pathlib import Path


CLASSIFICATION_DIR = Path(__file__).resolve().parent
CLASSIFICATION_MODEL_DIR = CLASSIFICATION_DIR / "model"
CLASSIFICATION_TRAIN_DIR = CLASSIFICATION_MODEL_DIR / "train"
CLASSIFICATION_TRAIN_DATA_DIR = CLASSIFICATION_TRAIN_DIR / "data"
CLASSIFICATION_TRAIN_OUTPUT_DIR = CLASSIFICATION_TRAIN_DIR / "output"


QWEN_INFER_API_BASE_URL = os.environ.get("QWEN_INFER_API_BASE_URL", "http://192.168.0.21:8004")
QWEN_INFER_API_PATH = os.environ.get("QWEN_INFER_API_PATH", "/infer")
QWEN_INFER_TIMEOUT_SECONDS = float(os.environ.get("QWEN_INFER_TIMEOUT_SECONDS", "300"))
QWEN_INFER_MAX_NEW_TOKENS = int(os.environ.get("QWEN_INFER_MAX_NEW_TOKENS", "512"))
QWEN_INFER_TEMPERATURE = float(os.environ.get("QWEN_INFER_TEMPERATURE", "0"))

QWEN_MAX_SEQ_LENGTH = 12288
QWEN_RE_TRAIN_TOKEN_MARGIN = 1000
QWEN_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_OUTPUT_DIR = str(CLASSIFICATION_TRAIN_OUTPUT_DIR / "key-embedding-graph-qwen2x5-1_5b")
QWEN_TRAIN_SOURCE_DATA_NAME = "train/train_000001.jsonl"
QWEN_FILTERED_TRAIN_DATA_NAME = "train.jsonl"
QWEN_VAL_DATA_NAME = "val.jsonl"
QWEN_TRAIN_DATA_PATH = Path(os.environ.get(
    "QWEN_TRAIN_DATA_PATH",
    str(CLASSIFICATION_TRAIN_DATA_DIR / QWEN_FILTERED_TRAIN_DATA_NAME),
))
QWEN_VAL_DATA_PATH = Path(os.environ.get(
    "QWEN_VAL_DATA_PATH",
    str(CLASSIFICATION_TRAIN_DATA_DIR / QWEN_VAL_DATA_NAME),
))
QWEN_RE_TRAIN_DATA_DIR = "re_train"
QWEN_VAL_RATIO = 0.05
QWEN_RANDOM_SEED = 17
KEY_SIGNAL_UNKNOWN_KEY = "unknown"
QWEN_SYSTEM_PROMPT = f"""You are the key-embedding-graph model.
Extract document-level key/signal values from OCR text.
Return exactly one JSON object and no extra text.
The JSON object must include four axes: subject, document_type, business_domain, and modifier.
Each axis must contain a key and signals.
The key must be a short, stable concept that represents the document.
Signals must be evidence phrases that actually appear in the input OCR text.
If evidence is weak or an axis cannot be determined, set key to "{KEY_SIGNAL_UNKNOWN_KEY}" and signals to an empty array.
Even when the document has multiple pages, return one final document-level JSON object."""
QWEN_SFT_TRAIN_CONFIG = {
    "learning_rate": 2e-4,
    "batch_size": 1,
    "grad_accum": 16,
    "save_steps": 100,
    "logging_steps": 10,
    "early_stopping_patience": 5,
    "early_stopping_threshold": 3e-4,
    "eval_accumulation_steps": 1,
    "device_map": "auto",
}
QWEN_LORA_CONFIG = {
    "r": 8,
    "alpha": 16,
    "dropout": 0.05,
    "target_modules": "q_proj,v_proj",
    "bias": "none",
    "task_type": "CAUSAL_LM",
}
QWEN_TOKENIZER_CONFIG = {
    "trust_remote_code": False,
    "padding_side": "right",
}
QWEN_QUANT_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_use_double_quant": True,
}
QWEN_MODEL_LOAD_CONFIG = {
    "trust_remote_code": False,
    "torch_dtype": "float16",
    "use_cache": False,
}
QWEN_TRAINING_ARGUMENTS = {
    "num_train_epochs": 1,
    "per_device_eval_batch_size": 1,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "logging_strategy": "no",
    "save_total_limit": 2,
    "evaluation_strategy": "no",
    "save_strategy": "no",
    "load_best_model_at_end": False,
    "fp16": True,
    "bf16": False,
    "optim": "paged_adamw_8bit",
    "gradient_checkpointing": True,
    "remove_unused_columns": True,
    "disable_tqdm": True,
    "report_to": [],
}

KEY_SIGNAL_AXES = ("subject", "document_type", "business_domain", "modifier")
KEY_SIGNAL_AXIS_SOURCES = {
    "subject": {"file_name": "subject.json", "key_field": "subject_key"},
    "document_type": {"file_name": "document_type.json", "key_field": "document_type_key"},
    "business_domain": {"file_name": "business_domain.json", "key_field": "business_domain_key"},
    "modifier": {"file_name": "modifier.json", "key_field": "modifier_key"},
}

PROJECTION_MODEL_PATH = Path(os.environ.get(
    "PROJECTION_MODEL_PATH",
    str(CLASSIFICATION_TRAIN_OUTPUT_DIR / "key_projection_model.pt"),
))
PROJECTION_TRAIN_DICTIONARY_DIR = Path(os.environ.get(
    "PROJECTION_TRAIN_DICTIONARY_DIR",
    str(CLASSIFICATION_TRAIN_DATA_DIR),
))
PROJECTION_DICTIONARY_DIR = Path(os.environ.get("PROJECTION_DICTIONARY_DIR", str(CLASSIFICATION_DIR / "dictionary")))
KEY_EMBEDDING_GRAPH_PATH = Path(os.environ.get(
    "KEY_EMBEDDING_GRAPH_PATH",
    str(PROJECTION_DICTIONARY_DIR / "key_embedding_graph.json"),
))
PROJECTION_ENCODER_MODEL = os.environ.get("PROJECTION_ENCODER_MODEL", "nlpai-lab/KURE-v1")
PROJECTION_ENCODER_TRUST_REMOTE_CODE = False
PROJECTION_DEVICE = os.environ.get("PROJECTION_DEVICE", "cuda")
PROJECTION_CACHE_FOLDER = os.environ.get(
    "PROJECTION_CACHE_FOLDER",
    str(CLASSIFICATION_MODEL_DIR / ".cache/sentence-transformers"),
)
PROJECTION_PRECISION = os.environ.get("PROJECTION_PRECISION", "fp16")
PROJECTION_BATCH_SIZE = int(os.environ.get("PROJECTION_BATCH_SIZE", "64"))
PROJECTION_KEEP_SCORE = float(os.environ.get("PROJECTION_KEEP_SCORE", "0.70"))
PROJECTION_KEEP_MARGIN = float(os.environ.get("PROJECTION_KEEP_MARGIN", "0.08"))
PROJECTION_REVIEW_SCORE = float(os.environ.get("PROJECTION_REVIEW_SCORE", "0.65"))
PROJECTION_CONFLICT_MARGIN = float(os.environ.get("PROJECTION_CONFLICT_MARGIN", "0.05"))
PROJECTION_DROP_SCORE = float(os.environ.get("PROJECTION_DROP_SCORE", "0.55"))
PROJECTION_EMBEDDING_DIM = 1024

PROJECTION_MODEL_CONFIG = {
    "input_dim": PROJECTION_EMBEDDING_DIM,
    "hidden_dim": 768,
    "output_dim": 384,
    "dropout": 0.1,
}

PROJECTION_TRAIN_CONFIG = {
    "seed": 17,
    "batch_size": 64,
    "samples_per_key": 4,
    "lr": 0.0003,
    "early_stopping_patience": 5,
    "early_stopping_min_delta": 0.0005,
    "weight_decay": 0.01,
    "valid_ratio": 0.15,
    "temperature": 0.07,
    "grad_clip": 1.0,
    "hard_negative_top": 8,
    "hard_margin": 0.12,
    "loss_weights": {
        "supcon": 1.0,
        "prototype": 0.35,
        "hard_margin": 0.25,
        "center": 0.1,
    },
}

SIGNAL_NORMALIZE_INSIDE_SCORE = 0.86
SIGNAL_NORMALIZE_KEY_SCORE = 0.76
SIGNAL_NORMALIZE_MARGIN = 0.10
SIGNAL_NORMALIZE_MAX_OTHER_SCORE = 0.74
SIGNAL_NORMALIZE_SCORE_ROUND = 6

KEY_EMBEDDING_GRAPH_TOP_K = 8
KEY_EMBEDDING_GRAPH_MIN_SIMILARITY = 0.70
KEY_EMBEDDING_GRAPH_BATCH_SIZE = 32
KEY_EMBEDDING_GRAPH_SPACE_SCALE = 180.0
KEY_EMBEDDING_GRAPH_OUTPUT_NAME = "key_embedding_graph.json"
KEY_EMBEDDING_NORMALIZE_REPORT_NAME = "embedding_normalize_report.json"
KEY_EMBEDDING_GRAPH_COLOR_HUE = 360
KEY_EMBEDDING_GRAPH_COLOR_SATURATION = 72
KEY_EMBEDDING_GRAPH_COLOR_LIGHTNESS = 54
KEY_EMBEDDING_GRAPH_KEY_SIZE = 26
KEY_EMBEDDING_GRAPH_SIGNAL_SIZE = 8
KEY_EMBEDDING_GRAPH_COORD_ROUND = 5
KEY_EMBEDDING_GRAPH_DISTANCE_ROUND = 4
