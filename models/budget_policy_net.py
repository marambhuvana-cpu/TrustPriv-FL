import torch
import torch.nn as nn
import torch.nn.functional as F


class BudgetPolicyNet(nn.Module):
    def __init__(self, state_dim=12, num_groups=4, hidden1=128, hidden2=64, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Dropout(dropout)
        )
        self.round_head = nn.Linear(hidden2, 1)
        self.policy_head = nn.Linear(hidden2, num_groups)
        self.value_head = nn.Linear(hidden2, 1)

    def forward(self, state):
        z = self.encoder(state)
        round_frac = torch.sigmoid(self.round_head(z))
        logits = self.policy_head(z)
        weights = F.softmax(logits, dim=-1)
        value = self.value_head(z)
        return round_frac, weights, value, logits
