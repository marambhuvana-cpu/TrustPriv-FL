import random, copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from federated.fedavg import fedavg, apply_update
from federated.robust_aggregation import median_aggregate, trimmed_mean_aggregate
from telemetry.telemetry_extractor import TelemetryExtractor
from telemetry.client_grouping import assign_groups, active_groups as get_active_groups
from policy.allocation import build_state, static_allocate, heuristic_allocate, policy_allocate
from policy.reward import compute_reward, fairness_variance
from privacy.privacy_accountant import PrivacyAccountant


class FLServer:
    def __init__(self, model, clients, test_dataset, cfg, device, policy_net=None, ppo_agent=None):
        self.model = model.to(device)
        self.clients = clients
        self.test_dataset = test_dataset
        self.cfg = cfg
        self.device = device
        self.policy_net = policy_net
        self.ppo_agent = ppo_agent
        self.accountant = PrivacyAccountant(cfg['privacy']['total_epsilon'], cfg['privacy']['delta'])
        self.group_map = assign_groups(len(clients), cfg['federated']['num_groups'])
        self.telemetry = TelemetryExtractor(cfg['privacy'].get('telemetry_noise_std', 0.0))
        self.prev_acc = 0.0
        self.prev_loss = 10.0
        self.client_last_acc = {i: 0.0 for i in range(len(clients))}

    def evaluate(self):
        self.model.eval()
        loader = DataLoader(self.test_dataset, batch_size=256, shuffle=False)
        correct, total, loss_sum = 0, 0, 0.0
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss_sum += float(criterion(logits, y))
                pred = logits.argmax(1)
                correct += int((pred == y).sum())
                total += y.numel()
        return correct / max(1, total), loss_sum / max(1, total)

    def _select_clients(self):
        n = len(self.clients)
        k = max(1, int(n * self.cfg['federated']['participation_rate']))
        selected = random.sample(range(n), k)
        for cid in selected:
            self.clients[cid].selected += 1
        return selected

    def _simulate_stragglers(self, selected):
        rate = self.cfg['federated']['straggler_rate']
        active = []
        for cid in selected:
            if random.random() >= rate:
                active.append(cid)
        return active if active else selected[:1]

    def _aggregate(self, updates, weights):
        agg = self.cfg['federated'].get('aggregator', 'fedavg').lower()
        if agg == 'median':
            return median_aggregate(updates)
        if agg == 'trimmed_mean':
            return trimmed_mean_aggregate(updates)
        return fedavg(updates, weights)

    def _prepare_group_telemetry(self, active, prior_loss=1.0, prior_norm=1.0):
        num_groups = self.cfg['federated']['num_groups']
        grouped = {g: [] for g in range(num_groups)}
        for cid in active:
            c = self.clients[cid]
            reliability = c.participations / max(1, c.selected)
            straggler_score = 1.0 - reliability
            vec = self.telemetry.client_telemetry(prior_loss, prior_norm, reliability, straggler_score, c.skew)
            grouped[self.group_map[cid]].append(vec)
        active_g = get_active_groups(active, self.group_map)
        group_tel = [self.telemetry.group_aggregate(grouped[g]) for g in active_g]
        return active_g, group_tel

    def _allocate(self, round_idx, active, active_g, group_tel, acc, loss):
        mode = self.cfg['policy']['mode'].lower()
        p = self.cfg['privacy']; f = self.cfg['federated']
        if mode == 'fedavg':
            return 0.0, {g: 0.0 for g in active_g}, None, None
        if mode == 'static':
            eps_t, alloc = static_allocate(active_g, self.accountant.remaining, p['min_epsilon'], p['max_round_epsilon'])
            return eps_t, alloc, None, None
        if mode in ['heuristic', 'adaptive_noise']:
            eps_t, alloc = heuristic_allocate(active_g, group_tel, self.accountant.remaining, p['min_epsilon'], p['max_round_epsilon'])
            return eps_t, alloc, None, None
        if mode == 'personalized':
            eps_t, alloc = heuristic_allocate(active_g, group_tel, self.accountant.remaining, p['min_epsilon'], p['max_round_epsilon'])
            return eps_t, alloc, None, None
        noniid = float(np.mean([self.clients[cid].skew for cid in active])) if active else 0.0
        state = build_state(round_idx, f['num_rounds'], acc, self.prev_acc, loss, self.prev_loss, self.accountant,
                            f['participation_rate'], f['straggler_rate'], noniid, group_tel)
        eps_t, alloc, pol_data = policy_allocate(self.policy_net, state, active_g, self.accountant.remaining,
                                                 p['min_epsilon'], p['max_round_epsilon'], self.device)
        return eps_t, alloc, state, pol_data

    def run_round(self, round_idx):
        selected = self._select_clients()
        active = self._simulate_stragglers(selected)
        pre_acc, pre_loss = self.evaluate()
        active_g, group_tel = self._prepare_group_telemetry(active, pre_loss, 1.0)
        eps_t, group_alloc, state, pol_data = self._allocate(round_idx, active, active_g, group_tel, pre_acc, pre_loss)
        spent = self.accountant.spend(eps_t, round_idx) if eps_t > 0 else 0.0
        updates, weights, losses, norms = [], [], [], []
        enable_dp = self.cfg['privacy']['enable_dp'] and self.cfg['policy']['mode'].lower() != 'fedavg'
        for cid in active:
            g = self.group_map[cid]
            eps_client = group_alloc.get(g, 0.0) / max(1, sum(1 for x in active if self.group_map[x] == g))
            upd, closs, unorm, n = self.clients[cid].train(
                self.model, eps_client, self.cfg['privacy']['clip_norm'], self.cfg['privacy']['base_noise_multiplier'], enable_dp=enable_dp)
            updates.append(upd); weights.append(n); losses.append(closs); norms.append(unorm)
        if updates:
            agg_update = self._aggregate(updates, weights)
            self.model = apply_update(self.model, agg_update)
        post_acc, post_loss = self.evaluate()
        gains = [post_acc - self.client_last_acc[cid] for cid in active]
        for cid in active:
            self.client_last_acc[cid] = post_acc
        fair = fairness_variance(gains)
        reward = compute_reward(post_acc - pre_acc, spent, fair,
                                self.cfg['policy']['privacy_coef'], self.cfg['policy']['fairness_coef'])
        ppo_loss = None
        if pol_data is not None and self.ppo_agent is not None:
            st, logits, value = pol_data
            self.ppo_agent.store(st, logits, value, active_g, list(group_alloc.values()), reward)
            if round_idx % self.cfg['policy']['update_every'] == 0:
                ppo_loss = self.ppo_agent.update()
        self.prev_acc, self.prev_loss = post_acc, post_loss
        return {
            'round': round_idx, 'selected_clients': len(selected), 'active_clients': len(active),
            'accuracy': post_acc, 'loss': post_loss, 'acc_gain': post_acc - pre_acc,
            'epsilon_spent': spent, 'epsilon_used': self.accountant.used_epsilon,
            'epsilon_remaining': self.accountant.remaining, 'fairness_variance': fair,
            'reward': reward, 'ppo_loss': ppo_loss, 'avg_client_loss': float(np.mean(losses)) if losses else None,
            'avg_update_norm': float(np.mean(norms)) if norms else None
        }

    def train(self):
        rows = []
        for r in range(1, self.cfg['federated']['num_rounds'] + 1):
            if self.accountant.exhausted() and self.cfg['policy']['mode'].lower() != 'fedavg':
                break
            rows.append(self.run_round(r))
            print(f"Round {r:03d} | acc={rows[-1]['accuracy']:.4f} loss={rows[-1]['loss']:.4f} eps_used={rows[-1]['epsilon_used']:.3f}")
        return rows
