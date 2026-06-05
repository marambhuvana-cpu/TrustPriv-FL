import torch
import torch.nn.functional as F


class PPOAgent:
    def __init__(self, policy_net, lr=1e-3, gamma=0.99, clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01, ppo_epochs=2):
        self.net = policy_net
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-5)
        self.gamma = gamma; self.clip_ratio = clip_ratio; self.value_coef = value_coef; self.entropy_coef = entropy_coef; self.ppo_epochs = ppo_epochs
        self.buffer = []

    def store(self, state_tensor, logits, value, active_groups, chosen_weights, reward):
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            logprob = torch.log(probs + 1e-8).mean()
        self.buffer.append((state_tensor.detach(), logprob.detach(), value.detach(), float(reward)))

    def update(self):
        if not self.buffer:
            return 0.0
        states = torch.cat([b[0] for b in self.buffer], dim=0)
        old_logp = torch.stack([b[1] for b in self.buffer]).view(-1)
        rewards = torch.tensor([b[3] for b in self.buffer], dtype=torch.float32, device=states.device)
        returns = []
        g = 0.0
        for r in reversed(rewards.tolist()):
            g = r + self.gamma * g
            returns.insert(0, g)
        returns = torch.tensor(returns, dtype=torch.float32, device=states.device)
        if returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        last_loss = 0.0
        for _ in range(self.ppo_epochs):
            _, _, values, logits = self.net(states)
            probs = F.softmax(logits, dim=-1)
            logp = torch.log(probs + 1e-8).mean(dim=-1)
            ratio = torch.exp(logp - old_logp)
            adv = returns - values.squeeze(-1)
            pg_loss = -torch.min(ratio * adv, torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv).mean()
            v_loss = F.mse_loss(values.squeeze(-1), returns)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            loss = pg_loss + self.value_coef*v_loss - self.entropy_coef*entropy
            self.opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0); self.opt.step()
            last_loss = float(loss.detach())
        self.buffer.clear()
        return last_loss
