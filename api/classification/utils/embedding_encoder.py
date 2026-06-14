import numpy as np
import torch

try:
    from ..config import PROJECTION_EMBEDDING_DIM, PROJECTION_ENCODER_TRUST_REMOTE_CODE
except ImportError:
    from api.classification.config import PROJECTION_EMBEDDING_DIM, PROJECTION_ENCODER_TRUST_REMOTE_CODE


def load_encoder(model_name, device, cache_folder, precision):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("sentence-transformers가 설치되어 있지 않습니다.") from exc

    if str(device).startswith("cuda") and str(precision).lower() == "fp16":
        encoder = SentenceTransformer(
            model_name,
            device="cpu",
            cache_folder=cache_folder,
            trust_remote_code=PROJECTION_ENCODER_TRUST_REMOTE_CODE,
        )
        encoder.half()
        encoder.to(device)
        return encoder

    return SentenceTransformer(
        model_name,
        device=device,
        cache_folder=cache_folder,
        trust_remote_code=PROJECTION_ENCODER_TRUST_REMOTE_CODE,
    )


def encode_texts(encoder, texts, batch_size, embedding_dim=PROJECTION_EMBEDDING_DIM):
    if not texts:
        return np.zeros((0, int(embedding_dim)), dtype=np.float32)

    embeddings = encoder.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=max(1, int(batch_size)),
    )
    return np.asarray(embeddings, dtype=np.float32)


def ensure_cuda_device(device):
    if not str(device or "").startswith("cuda"):
        return
    if not torch.cuda.is_available():
        raise RuntimeError("cuda device가 필요하지만 torch.cuda.is_available()이 False입니다.")
