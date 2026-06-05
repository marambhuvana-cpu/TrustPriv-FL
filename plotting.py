import os
import pandas as pd
import matplotlib.pyplot as plt


def make_plots(metrics_csv, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(metrics_csv)
    plots = [
        ('accuracy', 'Accuracy vs Communication Rounds', 'Accuracy', 'accuracy_vs_rounds.png'),
        ('loss', 'Loss vs Communication Rounds', 'Loss', 'loss_vs_rounds.png'),
        ('epsilon_used', 'Cumulative Privacy Budget Consumption', 'Used Epsilon', 'epsilon_consumption.png'),
        ('fairness_variance', 'Fairness Variance vs Rounds', 'Fairness Variance', 'fairness_variance.png'),
        ('reward', 'BudgetPolicyNet Reward vs Rounds', 'Reward', 'policy_reward.png'),
    ]
    for col, title, ylabel, fname in plots:
        if col in df:
            plt.figure(figsize=(7,4.5))
            plt.plot(df['round'], df[col], marker='o', linewidth=1.5)
            plt.xlabel('Communication Round')
            plt.ylabel(ylabel)
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, fname), dpi=300)
            plt.close()
    if {'epsilon_used','accuracy'}.issubset(df.columns):
        plt.figure(figsize=(7,4.5))
        plt.plot(df['epsilon_used'], df['accuracy'], marker='o', linewidth=1.5)
        plt.xlabel('Used Epsilon')
        plt.ylabel('Accuracy')
        plt.title('Privacy Utility Trade-off')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'privacy_utility_tradeoff.png'), dpi=300)
        plt.close()
