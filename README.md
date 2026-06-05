# TrustPriv-FL

## TrustPriv-FL: An Artificial Intelligence Framework with BudgetPolicyNet for Adaptive Differential Privacy Budget Optimisation in Federated Learning

TrustPriv-FL is an intelligent privacy-preserving federated learning framework that dynamically allocates differential privacy budgets during training using a reinforcement learning–based policy network called BudgetPolicyNet. Unlike conventional DP-FL methods that employ static privacy budgets, TrustPriv-FL continuously adapts privacy expenditure according to training progress, client heterogeneity, participation reliability, fairness requirements, and remaining privacy resources.

The framework combines:

* Federated Learning (FL)
* Differential Privacy (DP)
* Deep Reinforcement Learning (PPO)
* Adaptive Privacy Budget Allocation
* Privacy-Safe Telemetry
* Fairness-Aware Optimization
* Straggler-Aware Resource Management

---

# Key Features

## Adaptive Privacy Budget Allocation

BudgetPolicyNet dynamically determines privacy budgets for each communication round and client group.

## Reinforcement Learning-Based Policy Optimization

Uses Proximal Policy Optimization (PPO) to learn privacy allocation strategies.

## Differential Privacy Protection

Implements DP-SGD with:

* Gradient clipping
* Gaussian noise injection
* Privacy accounting

## Privacy-Safe Telemetry

Collects only low-dimensional privacy-preserving signals:

* Clipped loss
* Update norm
* Reliability score
* Data skew indicator

## Fairness-Aware Learning

Reduces utility disparities among heterogeneous clients.

## Straggler-Aware Scheduling

Handles intermittent participation and client dropouts.

## Scalable Federated Learning

Supports:

* MNIST
* Fashion-MNIST
* CIFAR-10

under highly heterogeneous non-IID environments.

---

# System Architecture

```text
+----------------------+
|     TrustPriv-FL     |
|        Server        |
+----------------------+
           |
           v
+------------------------------------+
| Adaptive Privacy Allocator (APA)   |
|                                    |
| BudgetPolicyNet (PPO)              |
| Privacy Accountant                 |
+------------------------------------+
           |
           v
  Privacy Budget Allocation
           |
---------------------------------------------
|                    |                      |
v                    v                      v

Client 1          Client 2             Client N
   |                 |                    |
 DP-SGD           DP-SGD               DP-SGD
   |                 |                    |
---------------------------------------------
           |
           v
  Federated Aggregation
           |
           v
      Global Model
```

---

# Repository Structure

```text
TrustPriv_FL/

├── main.py
├── config.yaml
├── requirements.txt
├── README.md

├── data/
│   ├── dataset_loader.py
│   └── noniid_partition.py

├── models/
│   ├── cnn_mnist.py
│   ├── cnn_cifar.py
│   └── budget_policy_net.py

├── federated/
│   ├── client.py
│   ├── server.py
│   ├── fedavg.py
│   └── robust_aggregation.py

├── privacy/
│   ├── dp_sgd.py
│   └── privacy_accountant.py

├── telemetry/
│   ├── telemetry_extractor.py
│   └── client_grouping.py

├── policy/
│   ├── ppo_agent.py
│   ├── reward.py
│   └── allocation.py

├── experiments/
│   ├── run_mnist.py
│   ├── run_fashion_mnist.py
│   ├── run_cifar10.py
│   ├── scalability_test.py
│   └── ablation_study.py

└── results/
    ├── figures/
    ├── metrics/
    └── saved_models/
```

---

# Datasets

The framework supports the following benchmark datasets.

| Dataset       | Classes | Samples | Purpose                    |
| ------------- | ------- | ------- | -------------------------- |
| MNIST         | 10      | 70,000  | Baseline evaluation        |
| Fashion-MNIST | 10      | 70,000  | Intermediate complexity    |
| CIFAR-10      | 10      | 60,000  | Robustness and scalability |

## Dataset Sources

### MNIST

http://yann.lecun.com/exdb/mnist/

### Fashion-MNIST

https://github.com/zalandoresearch/fashion-mnist

### CIFAR-10

https://www.cs.toronto.edu/~kriz/cifar.html

---

# Installation

## Clone Repository

```bash
git clone https://github.com/marambhuvana-cpu/TrustPriv-FL
cd TrustPriv_FL
```

## Create Environment

```bash
conda create -n trustpriv python=3.10
conda activate trustpriv
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

### Main Dependencies

* torch
* torchvision
* numpy
* pandas
* matplotlib
* seaborn
* scikit-learn
* gymnasium
* stable-baselines3
* opacus
* tqdm
* pyyaml

---

# Running Experiments

## MNIST

```bash
python experiments/run_mnist.py
```

## Fashion-MNIST

```bash
python experiments/run_fashion_mnist.py
```

## CIFAR-10

```bash
python experiments/run_cifar10.py
```

## Scalability Analysis

```bash
python experiments/scalability_test.py
```

Evaluates:

* 20 clients
* 40 clients
* 60 clients
* 80 clients
* 100 clients

## Ablation Study

```bash
python experiments/ablation_study.py
```

Evaluates:

1. Full TrustPriv-FL
2. Without BudgetPolicyNet
3. Without Fairness Module
4. Static DP Allocation
5. Uniform Budget Allocation

---

# Baselines

The following methods are implemented.

## FedAvg

Standard federated learning without privacy.

## DP-FedAvg

Static differential privacy budget allocation.

## Adaptive Noise DP-FL

Noise scheduling strategy.

## Personalised DP-FL

Client-specific privacy allocation.

## TrustPriv-FL (Proposed)

BudgetPolicyNet-based adaptive privacy optimization.

---

# BudgetPolicyNet

BudgetPolicyNet receives the following state vector.

### Training Progress

* Validation Accuracy Trend
* Loss Slope
* Current Round

### Privacy State

* Remaining Budget
* Used Budget
* Privacy Consumption

### Heterogeneity Statistics

* Participation Rate
* Straggler Ratio
* Non-IID Severity

### Telemetry Summary

* Clipped Loss
* Update Norm
* Reliability Score

### Network Architecture

```text
Input Layer (12)

Hidden Layer 1 (128)
ReLU
Dropout(0.2)

Hidden Layer 2 (64)
ReLU

Policy Head
Softmax

Value Head
Linear
```

### Optimizer

Adam

Learning Rate = 0.001

### RL Algorithm

Proximal Policy Optimization (PPO)

---

# Differential Privacy Configuration

```yaml
epsilon_total: 8.0
delta: 1e-5
clip_norm: 1.0
initial_noise_multiplier: 1.1
privacy_accountant: RDP
```

---

# Evaluation Metrics

## Utility Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* Convergence Rounds

## Privacy Metrics

* Privacy Budget Consumption
* Remaining Privacy Budget
* Membership Inference Resistance

## Fairness Metrics

* Utility Variance
* Client Fairness Score

## Robustness Metrics

* Straggler Tolerance
* Adversarial Noise Resilience

## Scalability Metrics

* Communication Cost
* Runtime
* Throughput

---

# Example Results

| Method               | Accuracy |
| -------------------- | -------- |
| FedAvg               | 93.2%    |
| DP-FedAvg            | 89.7%    |
| Adaptive Noise DP-FL | 91.4%    |
| Personalised DP-FL   | 92.1%    |
| TrustPriv-FL         | 95.8%    |

---

# Reproducibility

Random seeds are fixed for:

* NumPy
* PyTorch
* Python Random

Configuration file:

```text
config.yaml
```

contains all hyperparameters required to reproduce experiments.

---

# Citation

```bibtex
@article{TrustPrivFL2026,
 title={TrustPriv-FL: An Artificial Intelligence Framework with BudgetPolicyNet for Adaptive Differential Privacy Budget Optimisation in Federated Learning},
 year={2026}
}
```

---

# Future Extensions

* Secure Aggregation Integration
* Blockchain-Based Privacy Accounting
* Cross-Silo Federated Learning
* Hierarchical Federated Learning
* Multi-Agent Reinforcement Learning
* Transformer-Based BudgetPolicyNet
* Personalized Privacy Guarantees
* Real-World Healthcare Deployment

---

# License

MIT License

---


