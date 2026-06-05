import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from privacy.dp_sgd import clip_update, add_gaussian_noise, epsilon_to_noise


class FLClient:
    def __init__(self, client_id, dataset, device, batch_size=64, lr=0.001, local_epochs=1, skew=0.0):
        self.client_id = client_id
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.local_epochs = local_epochs
        self.skew = skew
        self.participations = 0
        self.selected = 0

    def train(self, global_model, epsilon, clip_norm, base_noise, enable_dp=True):
        self.participations += 1
        local_model = type(global_model)(*getattr(global_model, '_init_args', ())).to(self.device) if hasattr(global_model, '_init_args') else None
        if local_model is None:
            import copy; local_model = copy.deepcopy(global_model).to(self.device)
        local_model.load_state_dict(global_model.state_dict())
        local_model.train()
        opt = torch.optim.Adam(local_model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        total_loss, total_n = 0.0, 0
        for _ in range(self.local_epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad(set_to_none=True)
                loss = criterion(local_model(x), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), clip_norm)
                opt.step()
                total_loss += float(loss.detach()) * y.size(0); total_n += y.size(0)
        before = global_model.state_dict(); after = local_model.state_dict()
        update = {k: (after[k].detach() - before[k].detach()).to(self.device) for k in before}
        update, raw_norm = clip_update(update, clip_norm)
        if enable_dp:
            nm = epsilon_to_noise(epsilon, base_noise=base_noise)
            update = add_gaussian_noise(update, clip_norm, nm, num_samples=max(1, len(self.dataset)))
        avg_loss = total_loss / max(1, total_n)
        return update, avg_loss, raw_norm, len(self.dataset)
