import torch


def median_aggregate(updates):
    return {k: torch.median(torch.stack([u[k] for u in updates], dim=0), dim=0).values for k in updates[0]}


def trimmed_mean_aggregate(updates, trim_ratio=0.1):
    out = {}
    n = len(updates)
    trim = int(n * trim_ratio)
    for k in updates[0]:
        vals = torch.stack([u[k] for u in updates], dim=0).sort(dim=0).values
        if trim > 0 and n > 2 * trim:
            vals = vals[trim:-trim]
        out[k] = vals.mean(dim=0)
    return out
