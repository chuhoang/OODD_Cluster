"""
Test improvement ideas for class-conditional KNN and compare.
"""
import sys, os, time
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT_DIR)

import numpy as np
import torch
from tqdm import tqdm
from sklearn import metrics
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


def compute_intra_top1(pp, batch_data, batch_preds, intra_scores):
    """Baseline: KNN to all features of predicted class (single class)."""
    for c in np.unique(batch_preds):
        c_mask = batch_preds == c
        if c not in pp.cluster_features or len(pp.cluster_features[c]) == 0:
            continue
        all_feats_c = np.concatenate(
            [f for f in pp.cluster_features[c].values()], axis=0)
        k_eff = min(pp.K1, len(all_feats_c))
        batch_intra = batched_matrix_multiply(all_feats_c, batch_data[c_mask], K=k_eff)
        intra_scores[np.where(c_mask)[0]] = batch_intra


def compute_intra_topk3(pp, batch_data, batch_softmax, intra_scores):
    """Top-3 classes weighted by softmax probability (batched, optimized)."""
    topk = np.argsort(-batch_softmax, axis=1)[:, :3]
    topk_prob = np.take_along_axis(batch_softmax, topk, axis=1)
    topk_prob /= topk_prob.sum(axis=1, keepdims=True) + 1e-10

    for c in np.unique(topk.ravel()):
        if c not in pp.cluster_features or len(pp.cluster_features[c]) == 0:
            continue
        all_feats_c = np.concatenate(
            [f for f in pp.cluster_features[c].values()], axis=0)
        k_eff = min(pp.K1, len(all_feats_c))
        c_in_topk = (topk == c)
        rows, cols = np.where(c_in_topk)
        if len(rows) == 0:
            continue
        unique_rows = np.unique(rows)
        c_data = batch_data[unique_rows]
        batch_sim = batched_matrix_multiply(all_feats_c, c_data, K=k_eff)
        for i, row in enumerate(unique_rows):
            col_indices = cols[rows == row]
            total_w = topk_prob[row, col_indices].sum()
            intra_scores[row] += total_w * batch_sim[i]


def compute_intra_zscore(pp, batch_data, batch_preds, intra_scores, class_stats):
    """Z-score normalized intra-class similarity."""
    for c in np.unique(batch_preds):
        c_mask = batch_preds == c
        if c not in pp.cluster_features or len(pp.cluster_features[c]) == 0:
            continue
        all_feats_c = np.concatenate(
            [f for f in pp.cluster_features[c].values()], axis=0)
        k_eff = min(pp.K1, len(all_feats_c))
        batch_intra = batched_matrix_multiply(all_feats_c, batch_data[c_mask], K=k_eff)
        c_indices = np.where(c_mask)[0]
        if c in class_stats:
            mu, std = class_stats[c]
            if std > 1e-8:
                batch_intra = (batch_intra - mu) / std
        intra_scores[c_indices] = batch_intra


def compute_intra_adaptive_k1(pp, batch_data, batch_preds, intra_scores):
    """Adaptive K1: K1 = max(1, min(10, N_c // 5))."""
    for c in np.unique(batch_preds):
        c_mask = batch_preds == c
        if c not in pp.cluster_features or len(pp.cluster_features[c]) == 0:
            continue
        all_feats_c = np.concatenate(
            [f for f in pp.cluster_features[c].values()], axis=0)
        k_eff = max(1, min(pp.K1, len(all_feats_c) // 5))
        k_eff = min(k_eff, len(all_feats_c))
        batch_intra = batched_matrix_multiply(all_feats_c, batch_data[c_mask], K=k_eff)
        intra_scores[np.where(c_mask)[0]] = batch_intra


def run_ood_eval(pp, id_feats, id_preds, ood_feats, ood_preds, ood_softmax,
                 variant='baseline', class_stats=None, id_softmax=None):
    """Run OOD evaluation with a specific intra-score variant."""
    queue = PriorityQueue()
    memory_bank = None

    all_data = np.concatenate([id_feats, ood_feats], axis=0)
    all_preds = np.concatenate([id_preds, ood_preds], axis=0)
    all_softmax = np.concatenate([
        id_softmax if id_softmax is not None else np.zeros((len(id_feats), 100)),
        ood_softmax
    ], axis=0)
    label_id = np.concatenate([np.ones(len(id_feats)), np.zeros(len(ood_feats))])

    np.random.seed(100)
    idx = np.random.permutation(len(all_data))
    all_data, all_preds, label_id = all_data[idx], all_preds[idx], label_id[idx]
    all_softmax = all_softmax[idx]

    batch_size = 512
    num_batches = len(all_data) // batch_size + (len(all_data) % batch_size > 0)
    scores_list = []

    for i in tqdm(range(num_batches), desc=f"  {variant}", leave=False):
        start, end = i * batch_size, min((i + 1) * batch_size, len(all_data))
        batch_data = all_data[start:end]
        batch_preds = all_preds[start:end]
        batch_sm = all_softmax[start:end]

        intra_scores = np.zeros(len(batch_data))

        if variant == 'baseline':
            compute_intra_top1(pp, batch_data, batch_preds, intra_scores)
        elif variant == 'topk3':
            compute_intra_topk3(pp, batch_data, batch_sm, intra_scores)
        elif variant == 'zscore':
            compute_intra_zscore(pp, batch_data, batch_preds, intra_scores, class_stats)
        elif variant == 'adaptive_k1':
            compute_intra_adaptive_k1(pp, batch_data, batch_preds, intra_scores)

        # Queue update
        for j in range(len(batch_data)):
            queue.put(ScoreData(-intra_scores[j], batch_data[j]))
            if queue.qsize() > pp.queue_size:
                queue.get()

        # OOD distance
        data_list = [item.data for item in list(queue.queue)]
        new_food = np.array(data_list)
        ood_batch_score = batched_matrix_multiply(new_food, batch_data, pp.K2)

        batch_score = -intra_scores - ood_batch_score
        scores_list.append(batch_score)

    scores_all = np.concatenate(scores_list)
    return compute_auroc(scores_all[label_id == 1], scores_all[label_id == 0])


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

    # Extract ID features with softmax
    print("Extracting ID features...")
    id_feats_list, id_preds_list, id_sm_list = [], [], []
    for batch in tqdm(id_loader_dict['test'], desc='ID test'):
        data = batch['data'].cuda()
        output, feature = net(data, return_feature=True)
        feat = feature.data.cpu().numpy()
        feat = feat / (np.linalg.norm(feat, axis=-1, keepdims=True) + 1e-10)
        sm = torch.softmax(output, dim=1).detach().cpu().numpy()
        pred = output.argmax(dim=1).cpu()
        id_feats_list.append(feat)
        id_preds_list.append(pred)
        id_sm_list.append(sm)
    id_feats = np.concatenate(id_feats_list, axis=0)
    id_preds = torch.cat(id_preds_list).numpy().astype(int)
    id_softmax = np.concatenate(id_sm_list, axis=0)

    # Extract near-OOD features with softmax
    print("Extracting OOD features...")
    ood_data = {}
    for ood_name, ood_dl in ood_loader_dict['nearood'].items():
        of_list, op_list, os_list = [], [], []
        for batch in tqdm(ood_dl, desc=f'  {ood_name}', leave=False):
            data = batch['data'].cuda()
            output, feature = net(data, return_feature=True)
            feat = feature.data.cpu().numpy()
            feat = feat / (np.linalg.norm(feat, axis=-1, keepdims=True) + 1e-10)
            sm = torch.softmax(output, dim=1).detach().cpu().numpy()
            pred = output.argmax(dim=1).cpu()
            of_list.append(feat)
            op_list.append(pred)
            os_list.append(sm)
        ood_data[ood_name] = {
            'feats': np.concatenate(of_list, axis=0),
            'preds': torch.cat(op_list).numpy().astype(int),
            'softmax': np.concatenate(os_list, axis=0),
        }

    # Precompute class stats for Z-score variant from ID features
    print("Precomputing Z-score stats...")
    class_stats = {}
    for c in range(100):
        c_mask = id_preds == c
        if c_mask.sum() > 1 and c in pp.cluster_features and len(pp.cluster_features[c]) > 0:
            all_feats_c = np.concatenate(
                [f for f in pp.cluster_features[c].values()], axis=0)
            k_eff = min(pp.K1, len(all_feats_c))
            c_intras = batched_matrix_multiply(all_feats_c, id_feats[c_mask], K=k_eff)
            class_stats[c] = (c_intras.mean(), c_intras.std())

    variants = ['baseline', 'topk3', 'zscore', 'adaptive_k1']
    all_results = {}

    for variant in variants:
        print(f"\nTesting: {variant}")
        t0 = time.time()
        results = {}
        for ood_name, data in ood_data.items():
            auroc = run_ood_eval(
                pp, id_feats, id_preds,
                data['feats'], data['preds'], data['softmax'],
                variant=variant, class_stats=class_stats, id_softmax=id_softmax)
            results[ood_name] = auroc
            print(f"  {ood_name}: {100*auroc:.2f}%")
        results['mean'] = np.mean(list(results.values()))
        all_results[variant] = results
        print(f"  Mean: {100*results['mean']:.2f}% ({time.time()-t0:.0f}s)")

    # Comparison table
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    ds_names = list(ood_data.keys())
    header = f"{'Variant':<16}"
    for ds in ds_names:
        header += f" {ds:>10}"
    header += f" {'Mean':>10}"
    print(header)
    print("-" * 70)
    best_mean, best_variant = 0, ''
    for variant, results in all_results.items():
        line = f"{variant:<16}"
        for ds in ds_names:
            line += f" {100*results[ds]:>9.2f}%"
        line += f" {100*results['mean']:>9.2f}%"
        if results['mean'] > best_mean:
            best_mean, best_variant = results['mean'], variant
            line += " *"
        print(line)
    print(f"\nBest: {best_variant} ({100*best_mean:.2f}%)")


if __name__ == '__main__':
    main()
