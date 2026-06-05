import argparse, os, yaml, torch
import pandas as pd
from data.dataset_loader import load_dataset
from data.noniid_partition import dirichlet_partition, label_skew_score
from models.cnn_mnist import SmallCNN
from models.cnn_cifar import CIFARCNN
from models.budget_policy_net import BudgetPolicyNet
from federated.client import FLClient
from federated.server import FLServer
from policy.ppo_agent import PPOAgent
from plotting import make_plots
from utils import set_seed, get_device, ensure_dir, save_json


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def apply_overrides(cfg, args):
    if args.dataset: cfg['experiment']['dataset'] = args.dataset
    if args.mode: cfg['policy']['mode'] = args.mode
    if args.clients: cfg['federated']['num_clients'] = args.clients
    if args.rounds: cfg['federated']['num_rounds'] = args.rounds
    if args.alpha is not None: cfg['federated']['noniid_alpha'] = args.alpha
    if args.epsilon is not None: cfg['privacy']['total_epsilon'] = args.epsilon
    return cfg


def build_model(dataset, in_channels, num_classes):
    if dataset.lower() in ['cifar10','cifar-10']:
        return CIFARCNN(in_channels, num_classes)
    return SmallCNN(in_channels, num_classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--mode', default=None, choices=['trustpriv','static','heuristic','personalized','fedavg'])
    ap.add_argument('--clients', type=int, default=None)
    ap.add_argument('--rounds', type=int, default=None)
    ap.add_argument('--alpha', type=float, default=None)
    ap.add_argument('--epsilon', type=float, default=None)
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args)
    set_seed(cfg['experiment']['seed'])
    device = get_device(cfg['experiment']['device'])
    out_dir = cfg['experiment']['output_dir']
    ensure_dir(out_dir); ensure_dir(os.path.join(out_dir, 'figures')); ensure_dir(os.path.join(out_dir, 'saved_models'))

    train_ds, test_ds, in_channels, num_classes = load_dataset(cfg['experiment']['dataset'])
    parts, indices = dirichlet_partition(train_ds, cfg['federated']['num_clients'], cfg['federated']['noniid_alpha'], cfg['experiment']['seed'])
    skews = [label_skew_score(idx, train_ds) for idx in indices]
    clients = [FLClient(i, part, device, cfg['federated']['batch_size'], cfg['federated']['lr'], cfg['federated']['local_epochs'], skews[i]) for i, part in enumerate(parts)]

    model = build_model(cfg['experiment']['dataset'], in_channels, num_classes)
    policy_net = None; ppo_agent = None
    if cfg['policy']['mode'].lower() == 'trustpriv':
        policy_net = BudgetPolicyNet(12, cfg['federated']['num_groups'], cfg['policy']['hidden1'], cfg['policy']['hidden2'], cfg['policy']['dropout']).to(device)
        ppo_agent = PPOAgent(policy_net, cfg['policy']['lr'], cfg['policy']['gamma'], cfg['policy']['clip_ratio'], cfg['policy']['value_coef'], cfg['policy']['entropy_coef'], cfg['policy']['ppo_epochs'])

    server = FLServer(model, clients, test_ds, cfg, device, policy_net, ppo_agent)
    rows = server.train()
    df = pd.DataFrame(rows)
    metrics_path = os.path.join(out_dir, 'metrics.csv')
    df.to_csv(metrics_path, index=False)
    make_plots(metrics_path, os.path.join(out_dir, 'figures'))
    torch.save(server.model.state_dict(), os.path.join(out_dir, 'saved_models', 'global_model.pt'))
    summary = {
        'dataset': cfg['experiment']['dataset'], 'mode': cfg['policy']['mode'],
        'rounds_completed': len(rows), 'final_accuracy': float(df['accuracy'].iloc[-1]) if len(df) else None,
        'final_loss': float(df['loss'].iloc[-1]) if len(df) else None,
        'epsilon_used': float(server.accountant.used_epsilon), 'epsilon_remaining': float(server.accountant.remaining),
        'num_clients': cfg['federated']['num_clients'], 'noniid_alpha': cfg['federated']['noniid_alpha']
    }
    save_json(summary, os.path.join(out_dir, 'summary.json'))
    print('Saved results to', out_dir)


if __name__ == '__main__':
    main()
