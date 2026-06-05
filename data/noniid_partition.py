import numpy as np
from torch.utils.data import Subset


def dirichlet_partition(dataset, num_clients=20, alpha=0.5, seed=42, min_size=10):
    rng = np.random.default_rng(seed)
    labels = np.array(dataset.targets if hasattr(dataset, 'targets') else dataset.labels)
    num_classes = int(labels.max()) + 1
    while True:
        client_indices = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            idx_c = np.where(labels == c)[0]
            rng.shuffle(idx_c)
            props = rng.dirichlet(np.repeat(alpha, num_clients))
            cuts = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
            split = np.split(idx_c, cuts)
            for i, part in enumerate(split):
                client_indices[i].extend(part.tolist())
        sizes = [len(x) for x in client_indices]
        if min(sizes) >= min_size:
            break
    return [Subset(dataset, inds) for inds in client_indices], client_indices


def label_skew_score(indices, dataset):
    labels = np.array(dataset.targets if hasattr(dataset, 'targets') else dataset.labels)
    num_classes = int(labels.max()) + 1
    vals = labels[indices]
    hist = np.bincount(vals, minlength=num_classes).astype(float)
    p = hist / max(hist.sum(), 1.0)
    uniform = np.ones(num_classes) / num_classes
    return float(np.abs(p - uniform).sum() / 2.0)
