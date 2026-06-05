import numpy as np


def fairness_variance(client_gains):
    if len(client_gains) == 0:
        return 0.0
    return float(np.var(client_gains))


def compute_reward(acc_gain, eps_spent, fair_var, privacy_coef=0.25, fairness_coef=0.5):
    return float(acc_gain - privacy_coef * eps_spent - fairness_coef * fair_var)
