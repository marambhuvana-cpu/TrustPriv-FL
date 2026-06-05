import numpy as np
import torch


def build_state(round_idx, total_rounds, val_acc, prev_acc, loss, prev_loss, accountant, participation_rate, straggler_rate, noniid_severity, group_tel):
    acc_trend = val_acc - prev_acc
    loss_slope = prev_loss - loss
    tel = np.mean(group_tel, axis=0) if len(group_tel) else np.zeros(5)
    state = np.array([
        round_idx / max(1, total_rounds), acc_trend, loss_slope,
        accountant.remaining / max(accountant.total_epsilon, 1e-9), accountant.used_epsilon / max(accountant.total_epsilon, 1e-9),
        noniid_severity, participation_rate, straggler_rate,
        tel[0], tel[1], tel[2], tel[4]
    ], dtype=np.float32)
    return np.clip(state, -1, 1)


def static_allocate(active_groups, remaining, min_eps, max_round_eps):
    eps_t = min(max_round_eps, remaining)
    if not active_groups or eps_t <= 0:
        return 0.0, {}
    share = eps_t / len(active_groups)
    return eps_t, {g: max(min_eps, share) for g in active_groups}


def heuristic_allocate(active_groups, group_tel, remaining, min_eps, max_round_eps):
    eps_t = min(max_round_eps, remaining)
    if not active_groups or eps_t <= 0:
        return 0.0, {}
    scores = []
    for tel in group_tel:
        scores.append(0.4*tel[0] + 0.3*tel[1] + 0.3*tel[4] + 1e-6)
    scores = np.array(scores); weights = scores / scores.sum()
    alloc = {g: max(min_eps, eps_t*w) for g, w in zip(active_groups, weights)}
    s = sum(alloc.values())
    alloc = {g: v * eps_t / s for g, v in alloc.items()}
    return eps_t, alloc


def policy_allocate(policy_net, state, active_groups, remaining, min_eps, max_round_eps, device):
    if not active_groups or remaining <= 0:
        return 0.0, {}, None
    st = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    round_frac, weights, value, logits = policy_net(st)
    eps_t = float(round_frac.item()) * max_round_eps
    eps_t = min(max(eps_t, min_eps * len(active_groups)), max_round_eps, remaining)
    w = weights.detach().cpu().numpy().ravel()
    selected = np.array([w[g] for g in active_groups], dtype=np.float64)
    selected = selected / max(selected.sum(), 1e-12)
    alloc = {g: max(min_eps, eps_t * float(a)) for g, a in zip(active_groups, selected)}
    s = sum(alloc.values())
    alloc = {g: v * eps_t / s for g, v in alloc.items()}
    return eps_t, alloc, (st, logits, value)
