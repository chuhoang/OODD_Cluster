from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .base_postprocessor import BasePostprocessor
import os
from torch.utils.data import DataLoader
import openood.utils.comm as comm
from queue import PriorityQueue

normalizer = lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10)


def kth_largest_per_column(matrix, k):
    kth_values, _ = torch.kthvalue(matrix, matrix.size(0) - k + 1, dim=0)
    return kth_values.cpu().numpy()


def batched_matrix_multiply(ftrain, second_matrix, K, batch_size=256):
    """
    Compute dot product (cosine similarity on L2-normed features) in batches.
    ftrain: (n, d), second_matrix: (p, d), result: (p,) — K-th largest per column.
    """
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


def batched_matrix_multiply_k1(ftrain, second_matrix, batch_size=256):
    """
    Like batched_matrix_multiply but with K=1 (max similarity per column).
    Returns (p,) array of max cosine similarities.
    """
    return batched_matrix_multiply(ftrain, second_matrix, K=1, batch_size=batch_size)


class ScoreData:
    def __init__(self, score, data):
        self.score = score
        self.data = data

    def __lt__(self, other):
        return float(self.score) > float(other.score)


def xmeans_cluster(feats, k_min=1, k_max=4, random_state=42):
    """
    X-Means clustering: auto-selects K in [k_min, k_max] using inertia elbow.
    Reduces dimension to 64 via PCA before computing criteria for stability.
    Returns: cluster_centers (in original space), labels, per_cluster_features
    """
    N, D = feats.shape
    k_max = max(k_min, min(k_max, N // 2))
    if k_max <= k_min:
        kmeans = KMeans(n_clusters=k_min, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(feats)
        centers = kmeans.cluster_centers_
        pcf = {c: feats[labels == c] for c in range(k_min)}
        return centers, labels, pcf

    # Reduce dimension for stable clustering
    pca_dim = min(64, D, N - 1)
    if pca_dim < D:
        pca = PCA(n_components=pca_dim, random_state=random_state)
        feats_reduced = pca.fit_transform(feats)
    else:
        feats_reduced = feats

    inertias = []
    models = []
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(feats_reduced)
        inertias.append(kmeans.inertia_)
        models.append(kmeans)

    # Compute "acceleration" of inertia drop (second difference)
    # Best K is where inertia stops dropping significantly
    gains = []
    for i in range(1, len(inertias)):
        gain = inertias[i - 1] - inertias[i]
        gains.append(gain)

    # Pick K with best gain-to-noise ratio (simplified gap statistic approach)
    # Default to k_max unless the last gains are negligible
    if len(gains) >= 2:
        # If the last gain is < 20% of the first gain, stop earlier
        threshold = 0.15 * gains[0]
        best_idx = len(gains) - 1  # default: highest K
        for i in range(1, len(gains)):
            if gains[i] < threshold and best_idx == len(gains) - 1:
                best_idx = i  # stop at this K (i+1 clusters)
                break
    else:
        best_idx = len(gains) - 1

    best_k = best_idx + 1  # K is index+1
    best_k = max(k_min, min(best_k, k_max))

    # Re-run K-Means in original space with selected K
    kmeans = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(feats)
    centers = kmeans.cluster_centers_

    pcf = {c: feats[labels == c] for c in range(best_k)}
    return centers, labels, pcf


class ClassConditionalKNNPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config=config)
        args = config.postprocessor.postprocessor_args
        self.K1 = int(args.get('K1', 10))
        self.K2 = int(args.get('K2', 5))
        self.ALPHA = float(args.get('ALPHA', 0.5))
        self.queue_size = int(args.get('queue_size', 512))
        self.k_min = int(args.get('k_min', 1))
        self.k_max = int(args.get('k_max', 4))
        self.intra_weight = float(args.get('intra_weight', 1.0))

        self.cluster_centers = {}
        self.cluster_features = {}
        self.num_clusters_found = {}
        self.activation_log = None
        self.id_feature = None
        self.id_pred = None
        self.id_name = None
        self.aux_feature = None
        self.setup_flag = False
        self.count = 0

    def get_auxiliary_data(self, net=None, aux_data_loader=None):
        if aux_data_loader is None:
            return None
        net.eval()
        with torch.no_grad():
            aux_feature_dict = {}
            for aux_key, loader in aux_data_loader.items():
                aux_feature = []
                iter_idx = 0
                for batch in tqdm(loader, desc='get_aux_feature: ', position=0, leave=True):
                    data = batch['data'].cuda()
                    data = data[:128].float()
                    output, feature = net(data, return_feature=True)
                    aux_feature.append(normalizer(feature.data.cpu().numpy()))
                    iter_idx += 1
                    if iter_idx == 1:
                        break
                aux_feature = np.concatenate(aux_feature, axis=0)
                aux_feature_dict[aux_key] = aux_feature
            return aux_feature_dict

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if self.setup_flag:
            return

        aux_loader = None
        for aux_key, _ in ood_loader_dict.items():
            if "t_out" == aux_key:
                aux_loader = ood_loader_dict[aux_key]
                break
        if aux_loader is not None:
            self.aux_feature = self.get_auxiliary_data(net, aux_loader)

        if not os.path.exists('./cache'):
            os.makedirs('./cache')

        self.id_name = id_loader_dict["train"].dataset.name
        cache_name = f"cache/{self.id_name}_cc_knn_v2.npz"

        if not os.path.exists(cache_name):
            num_classes = id_loader_dict['train'].dataset.num_classes
            class_features = {c: [] for c in range(num_classes)}
            class_msp = {c: [] for c in range(num_classes)}

            net.eval()
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup [Extract]: ', position=0, leave=True):
                    data = batch['data_aux'].cuda()
                    labels = batch['label']
                    data = data.float()
                    batch_size, num_samples, C, H, W = data.shape
                    data = data.view(-1, C, H, W)
                    output, feature = net(data, return_feature=True)
                    msp, _ = torch.max(torch.softmax(output, dim=1), dim=1)

                    output = output.view(batch_size, num_samples, -1)
                    feature = feature.view(batch_size, num_samples, -1)
                    msp = msp.view(batch_size, num_samples)

                    msp_best, select_idx = torch.max(msp, dim=-1)
                    feature_best = feature[torch.arange(feature.size(0)), select_idx]
                    feature_normed = normalizer(feature_best.data.cpu().numpy())

                    for c in range(num_classes):
                        mask = labels == c
                        if mask.sum() > 0:
                            class_features[c].append(feature_normed[mask.numpy()])
                            class_msp[c].append(msp_best[mask])

            all_retained = []
            for c in range(num_classes):
                if len(class_features[c]) == 0:
                    self.cluster_centers[c] = np.array([])
                    self.cluster_features[c] = {}
                    self.num_clusters_found[c] = 0
                    continue

                feats = np.concatenate(class_features[c], axis=0)
                msps = torch.cat(class_msp[c])
                n_keep = max(self.k_max * 2, int(self.ALPHA * len(feats)))
                n_keep = min(n_keep, len(feats))

                sorted_idx = torch.argsort(msps, descending=True)[:n_keep].cpu().numpy()
                feats_retained = feats[sorted_idx]

                centers, labels, pcf = xmeans_cluster(
                    feats_retained, k_min=self.k_min, k_max=self.k_max)
                self.cluster_centers[c] = centers
                self.cluster_features[c] = pcf
                self.num_clusters_found[c] = len(centers)
                all_retained.append(feats_retained)

            self.activation_log = np.concatenate(all_retained, axis=0)
            self.setup_flag = True

            np.savez(cache_name,
                     cluster_centers=np.array(self.cluster_centers, dtype=object),
                     num_clusters_found=np.array(self.num_clusters_found, dtype=object),
                     activation_log=self.activation_log)
            for c in range(num_classes):
                for k, v in self.cluster_features[c].items():
                    np.save(f"cache/{self.id_name}_cc_knn_v2_c{c}_k{k}.npy", v)
        else:
            data = np.load(cache_name, allow_pickle=True)
            self.cluster_centers = data['cluster_centers'].item()
            self.num_clusters_found = data['num_clusters_found'].item()
            self.activation_log = data['activation_log']
            self.setup_flag = True
            for c in self.cluster_centers:
                self.cluster_features[c] = {}
                for k in range(self.num_clusters_found.get(c, 0)):
                    fname = f"cache/{self.id_name}_cc_knn_v2_c{c}_k{k}.npy"
                    if os.path.exists(fname):
                        self.cluster_features[c][k] = np.load(fname)

        print(f"X-Means clusters per class: {self.num_clusters_found}")

    @torch.no_grad()
    def acc_postprocess(self, net: nn.Module, data: Any):
        output, feature = net(data, return_feature=True)
        feature_normed = normalizer(feature.data.cpu().numpy())
        msp, pred = torch.max(torch.softmax(output, dim=1), dim=1)
        return msp, pred, feature_normed

    @torch.no_grad()
    def conf_postprocess(self, food, ftest, preds_ood=None, preds_id=None):
        """
        Class-conditional OOD scoring — follows the original KNN pattern:
          1. Compute intra-class similarity (class-conditional version of ID similarity)
          2. Update OOD queue: keep samples with LOW intra_sim (most OOD-like)
          3. Compute OOD distance: K2-th largest sim to OOD memory bank
          4. Score = intra_sim - ood_sim  (higher = more ID-like)
        """
        queue = PriorityQueue()

        # Seed OOD memory bank with auxiliary data (same as original KNN)
        if self.aux_feature is not None:
            keys = list(self.aux_feature.keys())
            key = keys[min(self.count, len(keys) - 1)]
            self.count += 1
            prior_food = self.aux_feature[key]
            prior_score = batched_matrix_multiply(self.activation_log, prior_food, self.K1)
            for j in range(prior_food.shape[0]):
                queue.put(ScoreData(-prior_score[j], prior_food[j]))
                if queue.qsize() > self.queue_size:
                    queue.get()
            memory_bank = prior_food[:5]
        else:
            memory_bank = None

        all_data = np.concatenate([ftest, food], axis=0)
        all_preds = np.concatenate([preds_id, preds_ood], axis=0) if preds_id is not None else None
        label_id = np.concatenate([np.ones(ftest.shape[0]), np.zeros(food.shape[0])], axis=0)

        np.random.seed(100)
        idx = np.random.permutation(all_data.shape[0])
        all_data = all_data[idx]
        label_id = label_id[idx]
        if all_preds is not None:
            all_preds = all_preds[idx]

        batch_size = 512
        num_batches = all_data.shape[0] // batch_size + (all_data.shape[0] % batch_size > 0)
        scores_list = []

        for i in tqdm(range(num_batches), desc="Class-Conditional Inference"):
            start = i * batch_size
            end = min(start + batch_size, all_data.shape[0])
            batch_data = all_data[start:end]
            batch_preds = all_preds[start:end] if all_preds is not None else None

            # --- 1. Compute class-conditional intra similarity ---
            intra_scores = np.zeros(batch_data.shape[0])
            if batch_preds is not None:
                unique_preds = np.unique(batch_preds)
                for c in unique_preds:
                    c_mask = batch_preds == c
                    c_data = batch_data[c_mask]
                    c_indices = np.where(c_mask)[0]

                    if c not in self.cluster_features or len(self.cluster_features[c]) == 0:
                        continue

                    # Concatenate ALL features from ALL clusters of class c
                    all_feats_c = np.concatenate(
                        [feats for feats in self.cluster_features[c].values()], axis=0)
                    k_eff = min(self.K1, len(all_feats_c))
                    batch_intra = batched_matrix_multiply(
                        all_feats_c, c_data, K=k_eff)
                    intra_scores[c_indices] = batch_intra

            # --- 2. Update OOD queue ---
            for j in range(batch_data.shape[0]):
                queue.put(ScoreData(-intra_scores[j], batch_data[j]))
                if queue.qsize() > self.queue_size:
                    queue.get()

            # --- 3. Compute OOD distance from updated queue ---
            data_list = [item.data for item in list(queue.queue)]
            new_food = np.array(data_list)
            if memory_bank is not None:
                new_food = np.concatenate([new_food, memory_bank], axis=0)

            ood_batch_score = batched_matrix_multiply(new_food, batch_data, self.K2)

            # --- 4. Final score ---
            batch_score = -intra_scores - ood_batch_score
            scores_list.append(batch_score)

        scores_all = np.concatenate(scores_list, axis=0)
        scores_in_final = scores_all[label_id == 1]
        scores_ood_final = scores_all[label_id == 0]

        return scores_in_final, scores_ood_final, label_id

    def inference(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  dataset_name: str = None,
                  progress: bool = True):
        if dataset_name is None:
            prefix = f"cache/{data_loader.dataset.name}_vs_{self.id_name}"
        else:
            prefix = f"cache/{dataset_name}_vs_{self.id_name}"
        cache_name = prefix + "_cc_knn_v2_out.npy"
        cache_pred = prefix + "_cc_knn_v2_out_pred.npy"
        cache_label = prefix + "_cc_knn_v2_out_label.npy"

        if not os.path.exists(cache_name):
            pred_list, label_list, feature_list = [], [], []
            for batch in tqdm(data_loader,
                              disable=not progress or not comm.is_main_process()):
                data = batch['data'].cuda()
                label = batch['label'].cuda()
                msp, pred, feature_normed = self.acc_postprocess(net, data)
                pred_list.append(pred.cpu())
                label_list.append(label.cpu())
                feature_list.append(feature_normed)

            pred_list = torch.cat(pred_list).numpy().astype(int)
            label_list = torch.cat(label_list).numpy().astype(int)
            feature_list = np.concatenate(feature_list, axis=0)

            id_conf, ood_conf, _ = self.conf_postprocess(
                feature_list, self.id_feature,
                preds_ood=pred_list, preds_id=self.id_pred)

            np.save(cache_name, feature_list)
            np.save(cache_pred, pred_list)
            np.save(cache_label, label_list)
            return id_conf, pred_list, ood_conf, label_list
        else:
            ood_feature = np.load(cache_name)
            pred_list = np.load(cache_pred)
            label_list = np.load(cache_label)
            id_conf, ood_conf, _ = self.conf_postprocess(
                ood_feature, self.id_feature,
                preds_ood=pred_list, preds_id=self.id_pred)
            return id_conf, pred_list, ood_conf, label_list

    def acc_inference(self,
                      net: nn.Module,
                      data_loader: DataLoader,
                      progress: bool = True):
        cache_name = f"cache/{data_loader.dataset.name}_vs_{self.id_name}_cc_knn_v2_out.npy"
        cache_pred = f"cache/{data_loader.dataset.name}_vs_{self.id_name}_cc_knn_v2_out_pred.npy"
        cache_label = f"cache/{data_loader.dataset.name}_vs_{self.id_name}_cc_knn_v2_out_label.npy"

        if not os.path.exists(cache_name):
            pred_list, label_list, feature_list = [], [], []
            for batch in tqdm(data_loader,
                              disable=not progress or not comm.is_main_process()):
                data = batch['data'].cuda()
                label = batch['label'].cuda()
                msp, pred, feature_normed = self.acc_postprocess(net, data)
                pred_list.append(pred.cpu())
                label_list.append(label.cpu())
                feature_list.append(feature_normed)

            pred_list = torch.cat(pred_list).numpy().astype(int)
            label_list = torch.cat(label_list).numpy().astype(int)
            feature_list = np.concatenate(feature_list, axis=0)

            np.save(cache_name, feature_list)
            np.save(cache_pred, pred_list)
            np.save(cache_label, label_list)

            self.id_feature = feature_list
            self.id_pred = pred_list
            return pred_list, label_list
        else:
            pred_list = np.load(cache_pred)
            label_list = np.load(cache_label)
            self.id_feature = np.load(cache_name)
            self.id_pred = pred_list
            return pred_list, label_list

    def set_hyperparam(self, hyperparam: list):
        self.K1 = hyperparam[0]

    def get_hyperparam(self):
        return self.K1
