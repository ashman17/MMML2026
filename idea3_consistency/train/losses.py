import torch
import torch.nn.functional as F


def letter_logits(logits_last, letter_token_ids, n_options):
    token_ids = torch.as_tensor(letter_token_ids, device=logits_last.device)
    selected = logits_last.index_select(1, token_ids)
    counts = torch.as_tensor(n_options, device=logits_last.device).reshape(-1, 1)
    valid = torch.arange(selected.shape[1], device=logits_last.device).reshape(1, -1) < counts
    return selected.masked_fill(~valid, float("-inf"))


def ce_loss(z, labels):
    return F.nll_loss(F.log_softmax(z, dim=-1), labels)


def learned_mixin_loss(z, bias, labels, g, ent_weight):
    valid = torch.isfinite(z) & torch.isfinite(bias)
    safe_bias = torch.where(valid, bias, torch.zeros_like(bias))
    scale = g.reshape(-1, 1)
    combined_logits = (z + scale * safe_bias).masked_fill(~valid, float("-inf"))
    combined_logprobs = F.log_softmax(combined_logits, dim=-1)
    prediction_loss = F.nll_loss(combined_logprobs, labels)

    bias_logits = (scale * safe_bias).masked_fill(~valid, float("-inf"))
    bias_logprobs = F.log_softmax(bias_logits, dim=-1)
    bias_probs = bias_logprobs.exp()
    safe_logprobs = torch.where(valid, bias_logprobs, torch.zeros_like(bias_logprobs))
    entropy = -(bias_probs * safe_logprobs).sum(dim=-1).mean()
    return prediction_loss + ent_weight * entropy, combined_logprobs
