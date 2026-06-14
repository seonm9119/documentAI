import torch
import torch.nn.functional as F

from ...config import PROJECTION_TRAIN_CONFIG


def supervised_contrastive_loss(features, labels):
    temperature = float(PROJECTION_TRAIN_CONFIG["temperature"])
    labels = labels.view(-1, 1)
    positive_mask = torch.eq(labels, labels.T).float()
    logits = torch.matmul(features, features.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    self_mask = torch.ones_like(positive_mask) - torch.eye(positive_mask.shape[0], device=positive_mask.device)
    positive_mask = positive_mask * self_mask
    exp_logits = torch.exp(logits) * self_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))

    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not torch.any(valid):
        return features.sum() * 0.0

    loss = -(positive_mask * log_prob).sum(dim=1) / positive_count.clamp_min(1.0)
    return loss[valid].mean()


def prototype_loss(features, labels):
    temperature = float(PROJECTION_TRAIN_CONFIG["temperature"])
    unique_labels = torch.unique(labels)
    if unique_labels.numel() < 2:
        return features.sum() * 0.0

    prototypes = []
    target = torch.zeros_like(labels)
    for label_index, label in enumerate(unique_labels):
        prototypes.append(F.normalize(features[labels == label].mean(dim=0), dim=0))
        target[labels == label] = label_index

    logits = torch.matmul(features, torch.stack(prototypes).T) / temperature
    return F.cross_entropy(logits, target)


def center_compactness_loss(features, labels):
    loss = features.sum() * 0.0
    center_count = 0

    for label in torch.unique(labels):
        rows = features[labels == label]
        if rows.shape[0] < 2:
            continue

        center = F.normalize(rows.mean(dim=0), dim=0)
        loss = loss + (1.0 - torch.matmul(rows, center)).mean()
        center_count += 1

    if center_count == 0:
        return features.sum() * 0.0
    return loss / center_count


def hard_margin_loss(features, labels, keys, hard_map):
    margin = float(PROJECTION_TRAIN_CONFIG["hard_margin"])
    loss = features.sum() * 0.0
    sample_count = 0

    for row_index, key in enumerate(keys):
        positive_indexes = [
            index for index, other_key in enumerate(keys)
            if other_key == key and index != row_index
        ]
        if not positive_indexes:
            continue

        hard_keys = set(hard_map.get(key, []))
        negative_indexes = [
            index for index, other_key in enumerate(keys)
            if other_key in hard_keys
        ]
        if not negative_indexes:
            continue

        positive_similarity = torch.matmul(features[row_index], features[positive_indexes].T).max()
        negative_similarity = torch.matmul(features[row_index], features[negative_indexes].T).max()
        loss = loss + F.relu(margin + negative_similarity - positive_similarity)
        sample_count += 1

    if sample_count == 0:
        return features.sum() * 0.0
    return loss / sample_count


def total_loss(features, labels, keys, hard_map):
    weights = PROJECTION_TRAIN_CONFIG["loss_weights"]
    supcon = supervised_contrastive_loss(features, labels)
    proto = prototype_loss(features, labels)
    hard = hard_margin_loss(features, labels, keys, hard_map)
    center = center_compactness_loss(features, labels)
    total = (
        float(weights["supcon"]) * supcon
        + float(weights["prototype"]) * proto
        + float(weights["hard_margin"]) * hard
        + float(weights["center"]) * center
    )
    return total, {
        "supcon": supcon,
        "prototype": proto,
        "hard_margin": hard,
        "center": center,
    }
