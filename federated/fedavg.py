import torch


def fedavg(updates, weights=None):
    if not updates:
        raise ValueError('No updates to aggregate')
    if weights is None:
        weights = [1.0 / len(updates)] * len(updates)
    s = sum(weights)
    weights = [w / s for w in weights]
    agg = {k: torch.zeros_like(v) for k, v in updates[0].items()}
    for upd, w in zip(updates, weights):
        for k in agg:
            agg[k] += upd[k] * w
    return agg


def apply_update(model, update):
    state = model.state_dict()
    for k in state:
        state[k] = state[k] + update[k].to(state[k].device)
    model.load_state_dict(state)
    return model
