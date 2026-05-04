"""
Test 6 feature-distance improvement ideas for class-conditional OOD detection.

1. Local density ratio
2. KNN neighbor variance
3. PCA subspace residual
4. Neighbor label agreement
5. Class contrast ratio
6. Mahalanobis distance
"""
import sys, os, time
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT_DIR)

import numpy as np
import torch
from tqdm import tqdm
from sklearn import metrics
from sklearn.decomposition import PCA
from queue import PriorityQueue
from openood.utils.config import Config
from openood.datasets import get_dataloader, get_ood_dataloader
from openood.networks import get_network
from openood.postprocessors.class_cond_knn_postprocessor import (
    ClassConditionalKNNPostprocessor, batched_matrix_multiply, ScoreData)


def compute_auroc(id_conf, ood_conf):
    conf = np.concatenate([id_conf, ood_conf])
    label = np.concatenate([np.ones(len(id_conf)), -np.ones(len(ood_conf))])
    fpr, tpr, _ = metrics.roc_curve(label, conf)
    return max(metrics.auc(fpr, tpr), metrics.auc(tpr, fpr))


def run_ood_eval(pp, id_feats, id_preds, ood_feats, ood_preds,
                 variant, precomputed):
    """Generic OOD eval pipeline. `variant` function computes intra_scores."""
    queue = PriorityQueue()
    all_data = np.concatenate([id_feats, ood_feats], axis=0)
    all_preds = np.concatenate([id_preds, ood_preds], axis=0)
    label_id = np.concatenate([np.ones(len(id_feats)), np.zeros(len(ood_feats))])

    np.random.seed(100)
    idx = np.random.permutation(len(all_data))
    all_data, all_preds, label_id = all_data[idx], all_preds[idx], label_id[idx]

    batch_size = 512
    n_batches = len(all_data) // batch_size + (len(all_data) % batch_size > 0)
    scores_list = []

    for i in tqdm(range(n_batches), desc=f"  {variant.__name__}", leave=False):
        start, end = i * batch_size, min((i + 1) * batch_size, len(all_data))
        batch_data = all_data[start:end]
        batch_preds = all_preds[start:end]
        intra_scores = variant(pp, batch_data, batch_preds, precomputed)

        for j in range(len(batch_data)):
            queue.put(ScoreData(-intra_scores[j], batch_data[j]))
            if queue.qsize() > pp.queue_size:
                queue.get()

        data_list = [item.data for item in list(queue.queue)]
        new_food = np.array(data_list)
        ood_batch = batched_matrix_multiply(new_food, batch_data, pp.K2)
        batch_score = -intra_scores - ood_batch
        scores_list.append(batch_score)

    scores_all = np.concatenate(scores_list)
    return compute_auroc(scores_all[label_id == 1], scores_all[label_id == 0])


# ─── Variant functions ───────────────────────────────────────────────

def baseline(pp, batch_data, batch_preds, precomputed):
    """K1-th neighbor in all features of predicted class."""
    scores = np.zeros(len(batch_data))
    for c in np.unique(batch_preds):
        mask = batch_preds == c
        if c not in pp.cluster_features or len(pp.cluster_features[c]) == 0:
            continue
        feats_c = np.concatenate([f for f in pp.cluster_features[c].values()], axis=0)
        k = min(pp.K1, len(feats_c))
        scores[np.where(mask)[0]] = batched_matrix_multiply(feats_c, batch_data[mask], K=k)
    return scores


def density_ratio(pp, batch_data, batch_preds, precomputed):
    """Local density ratio: sample's local density / cluster's avg local density."""
    scores = np.zeros(len(batch_data))
    for c in np.unique(batch_preds):
        mask = batch_preds == c
        if c not in pp.cluster_features or len(pp.cluster_features[c]) == 0:
            continue
        for nc, cluster_feats in pp.cluster_features[c].items():
            if len(cluster_feats) < 2:
                continue
            k = min(pp.K1, len(cluster_feats))
            avg_density = precomputed['cluster_density'].get((c, nc))
            if avg_density is None or avg_density == 0:
                continue
            c_data = batch_data[mask]
            # K-th neighbor distance to cluster features (cosine sim)
            sims = batched_matrix_multiply(cluster_feats, c_data, K=k)
            scores[np.where(mask)[0]] = sims / avg_density
    return scores


def knn_variance(pp, batch_data, batch_preds, precomputed):
    """Std of distances to all K nearest neighbors."""
    scores = np.zeros(len(batch_data))
    for c in np.unique(batch_preds):
        mask = batch_preds == c
        idxs = np.where(mask)[0]
        if c not in pp.cluster_features or len(pp.cluster_features[c]) == 0:
            continue
        feats_c = np.concatenate([f for f in pp.cluster_features[c].values()], axis=0)
        feats_t = torch.tensor(feats_c, dtype=torch.float32)
        c_data_t = torch.tensor(batch_data[mask], dtype=torch.float32)
        all_sims = c_data_t @ feats_t.T  # (n, N_c)
        # Top-K similarities (highest K)
        topk_vals, _ = torch.topk(all_sims, k=min(pp.K1 * 2, len(feats_c)), dim=1)
        # Std of top-K similarities
        scores[idxs] = topk_vals.std(dim=1).cpu().numpy()
    return scores


def pca_residual(pp, batch_data, batch_preds, precomputed):
    """L2 norm of PCA reconstruction residual."""
    pca_models = precomputed['pca_models']
    scores = np.zeros(len(batch_data))
    for c in np.unique(batch_preds):
        mask = batch_preds == c
        c_data = batch_data[mask]
        if c not in pca_models or pca_models[c] is None:
            continue
        pca, mean = pca_models[c]
        c_centered = c_data - mean
        c_recon = pca.inverse_transform(pca.transform(c_centered)) + mean
        residuals = np.linalg.norm(c_data - c_recon, axis=1)
        scores[np.where(mask)[0]] = residuals
    return scores


def neighbor_agreement(pp, batch_data, batch_preds, precomputed):
    """Fraction of K1 neighbors that share the same X-Means cluster as the nearest neighbor."""
    scores = np.zeros(len(batch_data))
    for c in np.unique(batch_preds):
        mask = batch_preds == c
        idxs = np.where(mask)[0]
        if c not in pp.cluster_features or len(pp.cluster_features[c]) == 0:
            continue
        # Gather all features + their cluster labels
        all_feats_list, all_labels_list = [], []
        for nc, f in pp.cluster_features[c].items():
            all_feats_list.append(f)
            all_labels_list.append(np.full(len(f), nc))
        feats_c = np.concatenate(all_feats_list, axis=0)
        labels_c = np.concatenate(all_labels_list, axis=0)

        feats_t = torch.tensor(feats_c, dtype=torch.float32)
        c_data_t = torch.tensor(batch_data[mask], dtype=torch.float32)
        all_sims = c_data_t @ feats_t.T  # (n, N_c)
        k = min(pp.K1, len(feats_c))
        _, topk_idx = torch.topk(all_sims, k=k, dim=1)  # (n, k)
        topk_labels = labels_c[topk_idx.cpu().numpy()]  # (n, k)
        # Agreement: fraction sharing majority cluster (HIGH=ID, negate so HIGH=OOD)
        for i, row_labels in enumerate(topk_labels):
            _, counts = np.unique(row_labels, return_counts=True)
            scores[idxs[i]] = -counts.max() / k
    return scores


def class_contrast(pp, batch_data, batch_preds, precomputed):
    """Sim to predicted class / max(sim to other classes)."""
    class_centroids = precomputed['class_centroids']  # (100, D)
    scores = np.zeros(len(batch_data))
    batch_t = torch.tensor(batch_data, dtype=torch.float32)
    centroids_t = torch.tensor(class_centroids, dtype=torch.float32)
    all_sims = batch_t @ centroids_t.T  # (N, 100)

    for i, c in enumerate(batch_preds):
        if c >= 100:
            continue
        sim_self = all_sims[i, c].item()
        other_mask = np.ones(100, dtype=bool)
        other_mask[c] = False
        sim_other_max = all_sims[i, other_mask].max().item()
        if sim_other_max > 1e-10:
            scores[i] = -(sim_self / sim_other_max)  # negate: HIGH ratio=ID → LOW score
        else:
            scores[i] = -1.0
    return scores


def mahalanobis(pp, batch_data, batch_preds, precomputed):
    """Mahalanobis distance to predicted class (diagonal precision)."""
    means = precomputed['mahala_means']
    inv_vars = precomputed['mahala_inv_vars']  # dict: c -> (D,) inverse variance
    scores = np.zeros(len(batch_data))
    for c in np.unique(batch_preds):
        mask = batch_preds == c
        c_data = batch_data[mask]
        if c not in means or means[c] is None:
            continue
        centered = c_data - means[c]
        # Diagonal Mahalanobis: sqrt(sum((x-μ)² / σ²))
        dist = np.sqrt(np.sum(centered ** 2 * inv_vars[c], axis=1))
        scores[np.where(mask)[0]] = dist
    return scores


# ─── Precomputation ──────────────────────────────────────────────────

def precompute_all(pp, id_feats, id_preds):
    pre = {}

    # Cluster densities
    pre['cluster_density'] = {}
    for c, clusters in pp.cluster_features.items():
        for nc, feats in clusters.items():
            if len(feats) < 2:
                continue
            k = min(pp.K1, len(feats))
            densities = batched_matrix_multiply(feats, feats, K=k)
            pre['cluster_density'][(c, nc)] = densities.mean()

    # PCA models per class
    pre['pca_models'] = {}
    for c in range(100):
        if c not in pp.cluster_features or len(pp.cluster_features[c]) == 0:
            pre['pca_models'][c] = None
            continue
        feats = np.concatenate([f for f in pp.cluster_features[c].values()], axis=0)
        if len(feats) < 20:
            pre['pca_models'][c] = None
            continue
        mean = feats.mean(axis=0)
        pca = PCA(n_components=min(32, len(feats) - 1, feats.shape[1] - 1))
        pca.fit(feats - mean)
        pre['pca_models'][c] = (pca, mean)

    # Class centroids (mean feature per class)
    centroids = np.zeros((100, id_feats.shape[1]))
    for c in range(100):
        if c in pp.cluster_features and len(pp.cluster_features[c]) > 0:
            feats = np.concatenate([f for f in pp.cluster_features[c].values()], axis=0)
            centroids[c] = feats.mean(axis=0)
    pre['class_centroids'] = centroids

    # Mahalanobis means + diagonal inverse variance
    pre['mahala_means'] = {}
    pre['mahala_inv_vars'] = {}
    for c in range(100):
        if c not in pp.cluster_features or len(pp.cluster_features[c]) == 0:
            pre['mahala_means'][c] = None
            pre['mahala_inv_vars'][c] = None
            continue
        feats = np.concatenate([f for f in pp.cluster_features[c].values()], axis=0)
        if len(feats) < 20:
            pre['mahala_means'][c] = None
            pre['mahala_inv_vars'][c] = None
            continue
        mean = feats.mean(axis=0)
        var = feats.var(axis=0) + 1e-4
        pre['mahala_means'][c] = mean
        pre['mahala_inv_vars'][c] = 1.0 / var

    return pre


def main():
    config = Config(
        os.path.join(ROOT_DIR, 'configs/pipelines/test/test_ood.yml'),
        os.path.join(ROOT_DIR, 'configs/datasets/cifar100/cifar100.yml'),
        os.path.join(ROOT_DIR, 'configs/datasets/cifar100/cifar100_ood.yml'),
        os.path.join(ROOT_DIR, 'configs/networks/resnet18_32x32.yml'),
        os.path.join(ROOT_DIR, 'configs/postprocessors/class_cond_knn.yml'),
        os.path.join(ROOT_DIR, 'configs/preprocessors/base_preprocessor.yml'),
    )
    config.parse_refs()
    config.network.pretrained = True
    config.network.checkpoint = os.path.join(ROOT_DIR, 'checkpoint/cifar100_res18/s0/best.ckpt')

    pp = ClassConditionalKNNPostprocessor(config)
    id_loader_dict = get_dataloader(config)
    ood_loader_dict = get_ood_dataloader(config)
    net = get_network(config.network)
    net.cuda()
    net.eval()

    pp.setup(net, id_loader_dict, ood_loader_dict)

    # Extract ID features
    print("Extracting ID features...")
    id_feats_list, id_preds_list = [], []
    for batch in tqdm(id_loader_dict['test'], desc='ID test'):
        data = batch['data'].cuda()
        output, feature = net(data, return_feature=True)
        feat = feature.data.cpu().numpy()
        feat = feat / (np.linalg.norm(feat, axis=-1, keepdims=True) + 1e-10)
        pred = output.argmax(dim=1).cpu()
        id_feats_list.append(feat)
        id_preds_list.append(pred)
    id_feats = np.concatenate(id_feats_list, axis=0)
    id_preds = torch.cat(id_preds_list).numpy().astype(int)

    # Extract near-OOD
    print("Extracting OOD features...")
    ood_data = {}
    for ood_name, ood_dl in ood_loader_dict['nearood'].items():
        of_list, op_list = [], []
        for batch in tqdm(ood_dl, desc=f'  {ood_name}', leave=False):
            data = batch['data'].cuda()
            output, feature = net(data, return_feature=True)
            feat = feature.data.cpu().numpy()
            feat = feat / (np.linalg.norm(feat, axis=-1, keepdims=True) + 1e-10)
            pred = output.argmax(dim=1).cpu()
            of_list.append(feat)
            op_list.append(pred)
        ood_data[ood_name] = {
            'feats': np.concatenate(of_list, axis=0),
            'preds': torch.cat(op_list).numpy().astype(int),
        }

    # Precompute all needed structures
    print("Precomputing PCA, GMM, densities...")
    pre = precompute_all(pp, id_feats, id_preds)

    variants = [
        baseline,
        density_ratio,
        knn_variance,
        pca_residual,
        neighbor_agreement,
        class_contrast,
        mahalanobis,
    ]

    all_results = {}
    for var_fn in variants:
        name = var_fn.__name__
        print(f"\nTesting: {name}")
        t0 = time.time()
        results = {}
        for ood_name, data in ood_data.items():
            auroc = run_ood_eval(
                pp, id_feats, id_preds, data['feats'], data['preds'],
                var_fn, pre)
            results[ood_name] = auroc
            print(f"  {ood_name}: {100*auroc:.2f}%")
        results['mean'] = np.mean(list(results.values()))
        all_results[name] = results
        print(f"  Mean: {100*results['mean']:.2f}% ({time.time()-t0:.0f}s)")

    # Table
    print("\n" + "=" * 80)
    print("COMPARISON — Feature Distance Ideas")
    print("=" * 80)
    ds_names = list(ood_data.keys())
    header = f"{'Method':<22}"
    for ds in ds_names:
        header += f" {ds:>10}"
    header += f" {'Mean':>10}   vs Baseline"
    print(header)
    print("-" * 80)

    baseline_mean = all_results.get('baseline', {}).get('mean', 0)
    best_name, best_mean = '', 0
    for name, res in all_results.items():
        delta = res['mean'] - baseline_mean
        mark = " *" if res['mean'] > best_mean else ""
        if res['mean'] > best_mean:
            best_mean, best_name = res['mean'], name
        line = f"{name:<22}"
        for ds in ds_names:
            line += f" {100*res[ds]:>9.2f}%"
        line += f" {100*res['mean']:>9.2f}%"
        line += f"   {delta:+.2f}%"
        line += mark
        print(line)

    print(f"\nBest: {best_name} ({100*best_mean:.2f}%)")


if __name__ == '__main__':
    main()
