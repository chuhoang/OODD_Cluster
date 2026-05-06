# Support direct execution: python -m openood.postprocessors.weakness_removal_knn_postprocessor
# or: python openood/postprocessors/weakness_removal_knn_postprocessor.py
import sys as _sys
if __package__ is None and __name__ == '__main__':
    import os as _os
    _ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')
    _sys.path.insert(0, _ROOT)

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
from torch.utils.data import DataLoader
from queue import PriorityQueue

try:
    from .base_postprocessor import BasePostprocessor
    import openood.utils.comm as comm
except ImportError:
    from openood.postprocessors.base_postprocessor import BasePostprocessor
    import openood.utils.comm as comm

normalizer = lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10)


def kth_largest_per_column(matrix, k):
    kth_values, _ = torch.kthvalue(matrix, matrix.size(0) - k + 1, dim=0)
    return kth_values.cpu().numpy()


def batched_matrix_multiply(ftrain, second_matrix, K, batch_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ftrain_tensor = torch.tensor(ftrain, device=device).float()
    second_matrix_tensor = torch.tensor(second_matrix, device=device).float()
    p, _ = second_matrix_tensor.shape

    num_batches = p // batch_size + (p % batch_size > 0)
    res = []
    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, p)
        second_batch_tensor = second_matrix_tensor[start:end, :].float()
        batch_result = ftrain_tensor @ second_batch_tensor.T
        score = kth_largest_per_column(batch_result, K)
        res.append(score)
    return np.concatenate(res, axis=0)


class ScoreData:
    def __init__(self, score, data):
        self.score = score
        self.data = data

    def __lt__(self, other):
        return float(self.score) > float(other.score)


def xmeans_cluster(feats, k_min=1, k_max=4):
    """X-Means via pyclustering library: auto-select K in [k_min, k_max]."""
    from pyclustering.cluster.xmeans import xmeans
    from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

    N = len(feats)
    k_max = max(k_min, min(k_max, N // 2))

    if k_max <= k_min:
        kmeans = KMeans(n_clusters=k_min, random_state=42, n_init=10)
        labels = kmeans.fit_predict(feats)
        centers = kmeans.cluster_centers_
        return centers, labels, {c: feats[labels == c] for c in range(k_min)}

    # pyclustering xmeans
    initial_centers = kmeans_plusplus_initializer(feats.tolist(), k_min).initialize()
    xm = xmeans(feats.tolist(), initial_centers, kmax=k_max)
    xm.process()

    clusters = xm.get_clusters()   # list of lists of indices
    centers = np.array(xm.get_centers())

    # Convert to labels array
    labels = np.zeros(N, dtype=int)
    for c_idx, cluster in enumerate(clusters):
        for idx in cluster:
            labels[idx] = c_idx

    per_cluster = {c: feats[labels == c] for c in range(len(centers))}
    return centers, labels, per_cluster


def to_device(data):
    """Move tensor to CUDA if available, else keep on CPU."""
    if torch.cuda.is_available():
        return data.cuda()
    return data


class WeaknessRemovalKNNPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config=config)
        args = config.postprocessor.postprocessor_args
        self.K1 = int(args.get('K1', 10))
        self.K2 = int(args.get('K2', 5))
        self.weak_remove_ratio = float(args['weak_remove_ratio'])
        self.queue_size = int(args.get('queue_size', 512))
        self.k_min = int(args.get('k_min', 1))
        self.k_max = int(args.get('k_max', 4))

        self.strong_features = {}
        self.cluster_strong_features = {}  # class_id → {cluster_id: features}
        self.cluster_centroids = {}        # class_id → (K_c, D) centroids
        self.ood_memory_bank = None
        self.activation_log = None
        self.id_feature = None
        self.id_pred = None
        self.id_name = None
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if self.setup_flag:
            return

        if not os.path.exists('./cache'):
            os.makedirs('./cache')

        self.id_name = id_loader_dict["train"].dataset.name
        r = str(self.weak_remove_ratio).replace('.', '_')
        cache_name = f"cache/{self.id_name}_wr_knn_r{r}.npz"

        if not os.path.exists(cache_name):
            num_classes = id_loader_dict['train'].dataset.num_classes
            class_features = {c: [] for c in range(num_classes)}

            net.eval()
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup [Extract]: ', position=0, leave=True):
                    data = to_device(batch['data_aux'])
                    labels = batch['label']
                    data = data.float()
                    bs, ns, C, H, W = data.shape
                    data = data.view(-1, C, H, W)
                    output, feature = net(data, return_feature=True)
                    msp, _ = torch.max(torch.softmax(output, dim=1), dim=1)
                    output = output.view(bs, ns, -1)
                    feature = feature.view(bs, ns, -1)
                    msp = msp.view(bs, ns)

                    msp_best, select_idx = torch.max(msp, dim=-1)
                    feature_best = feature[torch.arange(feature.size(0)), select_idx]
                    feature_normed = normalizer(feature_best.data.cpu().numpy())

                    for c in range(num_classes):
                        mask = labels == c
                        if mask.sum() > 0:
                            class_features[c].append(feature_normed[mask.numpy()])

            # ── Phase 1: X-Means + Weakness Removal ──
            all_strong = []
            all_weak_for_ood = []
            self.strong_features = {}

            for c in range(num_classes):
                if len(class_features[c]) == 0:
                    self.strong_features[c] = np.array([])
                    continue

                feats = np.concatenate(class_features[c], axis=0)
                centers, labels, per_cluster = xmeans_cluster(
                    feats, k_min=self.k_min, k_max=self.k_max)

                strong_c = []
                for nc in range(len(centers)):
                    cluster_feats = per_cluster[nc]
                    centroid = cluster_feats.mean(axis=0)

                    # Hypersphere distance: cosine sim to centroid
                    sims = batched_matrix_multiply(
                        centroid[None, :], cluster_feats, K=1)
                    # HIGH sim = close to centroid = strong
                    # LOW sim = far from centroid = weak

                    n_remove = max(1, int(len(cluster_feats) * self.weak_remove_ratio))
                    sorted_idx = np.argsort(sims)  # ascending: weakest first
                    weak_idx = sorted_idx[:n_remove]
                    strong_idx = sorted_idx[n_remove:]

                    # 50% of 7% = weakest-of-weak → OOD bank
                    wow_n = max(1, n_remove // 2)
                    wow_idx = weak_idx[:wow_n]
                    all_weak_for_ood.append(cluster_feats[wow_idx])

                    strong_c.append(cluster_feats[strong_idx])

                self.strong_features[c] = (np.concatenate(strong_c, axis=0)
                                           if strong_c else np.array([]))
                self.cluster_strong_features[c] = {nc: strong_c[nc]
                                                    for nc in range(len(strong_c))
                                                    if len(strong_c[nc]) > 0}
                self.cluster_centroids[c] = centers
                all_strong.append(self.strong_features[c])

            self.activation_log = np.concatenate(all_strong, axis=0)
            self.ood_memory_bank = np.concatenate(all_weak_for_ood, axis=0)
            self.setup_flag = True

            np.savez(cache_name,
                     activation_log=self.activation_log,
                     ood_memory_bank=self.ood_memory_bank,
                     strong_features=np.array(self.strong_features, dtype=object))
            for c in range(num_classes):
                if len(self.strong_features[c]) > 0:
                    np.save(f"cache/{self.id_name}_wr_knn_r{r}_strong_c{c}.npy",
                            self.strong_features[c])
        else:
            data = np.load(cache_name, allow_pickle=True)
            self.activation_log = data['activation_log']
            self.ood_memory_bank = data['ood_memory_bank']
            self.strong_features = data['strong_features'].item()
            self.setup_flag = True
            for c in self.strong_features:
                fname = f"cache/{self.id_name}_wr_knn_r{r}_strong_c{c}.npy"
                if os.path.exists(fname) and len(self.strong_features.get(c, [])) == 0:
                    self.strong_features[c] = np.load(fname)

        print(f"ID dict: {self.activation_log.shape[0]} strong features")
        print(f"OOD bank: {self.ood_memory_bank.shape[0]} weak features")

    @torch.no_grad()
    def acc_postprocess(self, net: nn.Module, data: Any):
        output, feature = net(data, return_feature=True)
        feature_normed = normalizer(feature.data.cpu().numpy())
        msp, pred = torch.max(torch.softmax(output, dim=1), dim=1)
        return msp, pred, feature_normed

    @torch.no_grad()
    def conf_postprocess(self, food, ftest, preds_ood=None, preds_id=None):
        # ── Seed queue with OOD memory bank ──
        queue = PriorityQueue()
        ood_seed_scores = batched_matrix_multiply(
            self.activation_log, self.ood_memory_bank, self.K1)
        for j in range(len(self.ood_memory_bank)):
            queue.put(ScoreData(ood_seed_scores[j], self.ood_memory_bank[j]))
            if queue.qsize() > self.queue_size:
                queue.get()

        all_data = np.concatenate([ftest, food], axis=0)
        all_preds = (np.concatenate([preds_id, preds_ood], axis=0)
                     if preds_id is not None else None)
        label_id = np.concatenate([np.ones(len(ftest)), np.zeros(len(food))])

        np.random.seed(100)
        idx = np.random.permutation(len(all_data))
        all_data, label_id = all_data[idx], label_id[idx]
        if all_preds is not None:
            all_preds = all_preds[idx]

        batch_size = 512
        num_batches = (len(all_data) // batch_size +
                       (len(all_data) % batch_size > 0))
        scores_list = []

        for i in tqdm(range(num_batches), desc="WR-KNN Inference"):
            start = i * batch_size
            end = min(start + batch_size, len(all_data))
            batch_data = all_data[start:end]
            batch_preds = all_preds[start:end] if all_preds is not None else None

            # ── 1. Class-conditional ID score (all clusters of predicted class) ──
            intra_scores = np.zeros(len(batch_data))
            if batch_preds is not None:
                for c in np.unique(batch_preds):
                    c_mask = batch_preds == c
                    c_indices = np.where(c_mask)[0]
                    if (c not in self.strong_features or
                            len(self.strong_features[c]) == 0):
                        continue
                    k = min(self.K1, len(self.strong_features[c]))
                    intra_scores[c_indices] = batched_matrix_multiply(
                        self.strong_features[c], batch_data[c_mask], K=k)

            # ── 2. Update OOD queue (original KNN pattern) ──
            global_id_scores = batched_matrix_multiply(
                self.activation_log, batch_data, self.K1)
            for j in range(len(batch_data)):
                queue.put(ScoreData(global_id_scores[j], batch_data[j]))
                if queue.qsize() > self.queue_size:
                    queue.get()

            # ── 3. OOD distance from queue ──
            new_food = np.array([item.data for item in list(queue.queue)])
            ood_batch_score = batched_matrix_multiply(
                new_food, batch_data, self.K2)

            # ── 4. Final score ──
            batch_score = intra_scores - ood_batch_score
            scores_list.append(batch_score)

        scores_all = np.concatenate(scores_list)
        return (scores_all[label_id == 1],
                scores_all[label_id == 0],
                label_id)

    def inference(self, net, data_loader, dataset_name=None, progress=True):
        prefix = (f"cache/{dataset_name}_vs_{self.id_name}"
                  if dataset_name else
                  f"cache/{data_loader.dataset.name}_vs_{self.id_name}")
        cname = prefix + "_wr_knn_out.npy"
        cp = prefix + "_wr_knn_out_pred.npy"
        cl = prefix + "_wr_knn_out_label.npy"

        if not os.path.exists(cname):
            pl, ll, fl = [], [], []
            for batch in tqdm(data_loader,
                              disable=not progress or not comm.is_main_process()):
                data = to_device(batch['data'])
                label = to_device(batch['label'])
                _, pred, feat = self.acc_postprocess(net, data)
                pl.append(pred.cpu())
                ll.append(label.cpu())
                fl.append(feat)
            pl = torch.cat(pl).numpy().astype(int)
            ll = torch.cat(ll).numpy().astype(int)
            fl = np.concatenate(fl, axis=0)

            id_c, ood_c, _ = self.conf_postprocess(
                fl, self.id_feature, preds_ood=pl, preds_id=self.id_pred)
            np.save(cname, fl); np.save(cp, pl); np.save(cl, ll)
            return id_c, pl, ood_c, ll
        else:
            fl = np.load(cname); pl = np.load(cp); ll = np.load(cl)
            id_c, ood_c, _ = self.conf_postprocess(
                fl, self.id_feature, preds_ood=pl, preds_id=self.id_pred)
            return id_c, pl, ood_c, ll

    def acc_inference(self, net, data_loader, progress=True):
        cname = f"cache/{data_loader.dataset.name}_vs_{self.id_name}_wr_knn_out.npy"
        cp = f"cache/{data_loader.dataset.name}_vs_{self.id_name}_wr_knn_out_pred.npy"
        cl = f"cache/{data_loader.dataset.name}_vs_{self.id_name}_wr_knn_out_label.npy"

        if not os.path.exists(cname):
            pl, ll, fl = [], [], []
            for batch in tqdm(data_loader,
                              disable=not progress or not comm.is_main_process()):
                data = to_device(batch['data'])
                label = to_device(batch['label'])
                _, pred, feat = self.acc_postprocess(net, data)
                pl.append(pred.cpu())
                ll.append(label.cpu())
                fl.append(feat)
            pl = torch.cat(pl).numpy().astype(int)
            ll = torch.cat(ll).numpy().astype(int)
            fl = np.concatenate(fl, axis=0)

            np.save(cname, fl); np.save(cp, pl); np.save(cl, ll)
            self.id_feature = fl; self.id_pred = pl
            return pl, ll
        else:
            pl = np.load(cp); ll = np.load(cl)
            self.id_feature = np.load(cname); self.id_pred = pl
            return pl, ll

    def set_hyperparam(self, hyperparam: list):
        self.K1 = hyperparam[0]

    def get_hyperparam(self):
        return self.K1


if __name__ == '__main__':
    import argparse
    import sys
    import os
    import time
    import datetime
    import numpy as np
    from sklearn import metrics

    parser = argparse.ArgumentParser(
        description='Weakness Removal KNN OOD Detection')
    parser.add_argument('--dataset', default='wood',
                        choices=['wood', 'cifar100'],
                        help='Dataset to evaluate (default: wood)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device (default: 0, use -1 for CPU)')
    parser.add_argument('--K1', type=int, default=10,
                        help='K for intra-class ID scoring (default: 10)')
    parser.add_argument('--K2', type=int, default=5,
                        help='K for OOD scoring (default: 5)')
    parser.add_argument('--weak_remove_ratio', type=float, default=0.15,
                        help='Fraction of weak features removed (default: 0.15)')
    parser.add_argument('--queue_size', type=int, default=512,
                        help='OOD queue size (default: 512)')
    parser.add_argument('--k_min', type=int, default=1,
                        help='Min clusters for X-Means (default: 1)')
    parser.add_argument('--k_max', type=int, default=4,
                        help='Max clusters for X-Means (default: 4)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (default: 64)')
    args = parser.parse_args()

    # ── Resolve root dir & sys.path ──
    ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..')
    sys.path.insert(0, ROOT_DIR)

    if args.gpu == '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        num_gpus = 0
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        num_gpus = 1

    from openood.utils.config import Config
    from openood.datasets import get_dataloader, get_ood_dataloader
    from openood.utils import setup_logger

    # ── Dataset-specific configs ──
    if args.dataset == 'wood':
        dataset_cfg = os.path.join(ROOT_DIR, 'configs/datasets/wood/wood.yml')
        ood_cfg = os.path.join(ROOT_DIR, 'configs/datasets/wood/wood_ood.yml')
        network_cfg = os.path.join(ROOT_DIR, 'configs/networks/resnet18_224x224.yml')
        default_ckpt = None  # must be provided by user
        num_classes = 130
        from openood.networks.resnet18_224x224 import ResNet18_224x224
        net = ResNet18_224x224(num_classes=num_classes)
    else:  # cifar100
        dataset_cfg = os.path.join(ROOT_DIR, 'configs/datasets/cifar100/cifar100.yml')
        ood_cfg = os.path.join(ROOT_DIR, 'configs/datasets/cifar100/cifar100_ood.yml')
        network_cfg = os.path.join(ROOT_DIR, 'configs/networks/resnet18_32x32.yml')
        default_ckpt = os.path.join(ROOT_DIR, 'checkpoint/cifar100_res18/s0/best.ckpt')
        num_classes = 100
        from openood.networks.resnet18_32x32 import ResNet18_32x32
        net = ResNet18_32x32(num_classes=num_classes)

    checkpoint = args.checkpoint or default_ckpt
    if checkpoint is None:
        parser.error('--checkpoint is required for wood dataset '
                     '(no default checkpoint available)')

    postprocessor_cfg = os.path.join(
        ROOT_DIR, 'configs/postprocessors/weakness_removal_knn.yml')
    preprocessor_cfg = os.path.join(
        ROOT_DIR, 'configs/preprocessors/base_preprocessor.yml')
    pipeline_cfg = os.path.join(
        ROOT_DIR, 'configs/pipelines/test/test_ood.yml')

    config = Config(dataset_cfg, ood_cfg, network_cfg,
                    pipeline_cfg, preprocessor_cfg, postprocessor_cfg)
    config.parse_refs()
    config.network.pretrained = True
    config.network.num_gpus = num_gpus
    config.network.checkpoint = checkpoint
    config.postprocessor.postprocessor_args.K1 = args.K1
    config.postprocessor.postprocessor_args.K2 = args.K2
    config.postprocessor.postprocessor_args.weak_remove_ratio = \
        args.weak_remove_ratio
    config.postprocessor.postprocessor_args.queue_size = args.queue_size
    config.postprocessor.postprocessor_args.k_min = args.k_min
    config.postprocessor.postprocessor_args.k_max = args.k_max
    config.dataset.train.batch_size = args.batch_size
    config.dataset.test.batch_size = args.batch_size
    config.dataset.val.batch_size = args.batch_size

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    config.exp_name = (f"{config.dataset.name}_{config.network.name}_"
                       f"{config.pipeline.name}_ood_"
                       f"{config.postprocessor.name}_{ts}")
    config.output_dir = os.path.join(ROOT_DIR, 'results', config.exp_name)
    setup_logger(config)

    pp = WeaknessRemovalKNNPostprocessor(config)
    id_loader_dict = get_dataloader(config)
    ood_loader_dict = get_ood_dataloader(config)

    # ── Load model ──
    print(f"Loading checkpoint: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location='cpu')
    net.load_state_dict(ckpt, strict=False)
    net.eval()
    if num_gpus > 0:
        net.cuda()

    # ── Setup ──
    print("Setting up WR-KNN...")
    t0 = time.time()
    pp.setup(net, id_loader_dict, ood_loader_dict)
    print(f"Setup: {time.time() - t0:.0f}s")

    # ── ID test accuracy ──
    print("Extracting ID test features...")
    t0 = time.time()
    id_preds, id_labels = pp.acc_inference(net, id_loader_dict['test'])
    id_acc = (id_preds == id_labels).mean()
    print(f"ID accuracy: {100 * id_acc:.2f}% ({time.time() - t0:.0f}s)")

    # ── OOD evaluation ──
    def compute_auroc(id_conf, ood_conf):
        conf = np.concatenate([id_conf, ood_conf])
        label = np.concatenate([np.ones(len(id_conf)), -np.ones(len(ood_conf))])
        fpr, tpr, _ = metrics.roc_curve(label, conf)
        return max(metrics.auc(fpr, tpr), metrics.auc(tpr, fpr))

    print("\n" + "=" * 70)
    for split in ['nearood', 'farood']:
        print(f"\n--- {split.upper()} ---")
        aurocs = []
        for name, dl in ood_loader_dict[split].items():
            t0 = time.time()
            id_c, ood_p, ood_c, ood_l = pp.inference(net, dl, name)
            auroc = compute_auroc(id_c, ood_c)
            aurocs.append(auroc)
            print(f"  {name}: AUROC={100 * auroc:.2f}% "
                  f"({time.time() - t0:.0f}s)")
        print(f"  Mean AUROC: {100 * np.mean(aurocs):.2f}%")
    print("\nDone.")
