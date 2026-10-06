import torch
import torch.nn.functional as F


def contrastive_loss_func(features, labels, logit_scale):
    """
    Exact implementation of TrafficCLIP's L_CL.
    Optimizes fused vision features V_fused.
    """
    # Compute similarity matrix (V_i · V_a)
    logits = torch.matmul(features, features.T) * logit_scale

    # For numerical stability (subtract max)
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    # Create masks for positive samples P(i)
    # labels.view(-1, 1) == labels.view(1, -1) finds all matching classes in batch
    mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float().to(features.device)

    # Exclude self-similarity
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(features.shape[0]).view(-1, 1).to(features.device),
        0,
    )
    mask = mask * logits_mask

    # Compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # Mean log-likelihood over positive samples
    # (1/|P(i)|) * sum(-log(...))
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)

    # Final loss: mean over all N samples
    loss = -mean_log_prob_pos.mean()
    return loss
