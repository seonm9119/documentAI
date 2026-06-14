import torch
import torch.nn.functional as F
from torch import nn

from ...config import PROJECTION_MODEL_CONFIG


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=None, hidden_dim=None, output_dim=None, dropout=None):
        super().__init__()
        input_dim = PROJECTION_MODEL_CONFIG["input_dim"] if input_dim is None else int(input_dim)
        hidden_dim = PROJECTION_MODEL_CONFIG["hidden_dim"] if hidden_dim is None else int(hidden_dim)
        output_dim = PROJECTION_MODEL_CONFIG["output_dim"] if output_dim is None else int(output_dim)
        dropout = PROJECTION_MODEL_CONFIG["dropout"] if dropout is None else float(dropout)

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, embeddings):
        return F.normalize(self.layers(embeddings), dim=-1)


class KeyEmbeddingProjectionModel(nn.Module):
    def __init__(self, axes, model_config=None):
        super().__init__()
        self.axes = list(axes)
        config = dict(PROJECTION_MODEL_CONFIG)
        config.update(model_config or {})
        self.model_config = config
        self.heads = nn.ModuleDict({
            axis: ProjectionHead(
                config["input_dim"],
                config["hidden_dim"],
                config["output_dim"],
                config["dropout"],
            )
            for axis in self.axes
        })

    def forward(self, axis, embeddings):
        return self.heads[str(axis)](embeddings)


def save_projection_model(path, model, metadata):
    torch.save({
        "state_dict": model.state_dict(),
        "axes": list(model.axes),
        "model_config": dict(model.model_config),
        "metadata": metadata,
    }, path)
