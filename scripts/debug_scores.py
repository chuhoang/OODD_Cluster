"""
Quick debug: test intra_sim alone (no queue) on a sample to isolate the issue.
"""
import sys, os
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT_DIR)

import numpy as np
import torch
from sklearn import metrics

from openood.postprocessors.class_cond_knn_postprocessor import (
    ClassConditionalKNNPostprocessor, batched_matrix_multiply)
from openood.utils.config import Config
from openood.datasets import get_dataloader, get_ood_dataloader
from openood.networks import get_network

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

    # Get ID features
    print("Loading ID features...")
    id_preds, _ = pp.acc_inference(net, id_loader_dict['test'])
    id_feats = pp.id_feature

    # Use first 2000 ID + 2000 OOD for speed
    n_sample = 2000
    for ood_name, ood_dl in ood_loader_dict['nearood'].items():
        print(f"\n=== {ood_name} ===")
        ood_feats, ood_preds = [], []
        for batch in ood_dl:
            data = batch['data'].cuda()
            _, pred, feat = pp.acc_postprocess(net, data)
            ood_feats.append(feat)
            ood_preds.append(pred.cpu())
        ood_feats = np.concatenate(ood_feats, axis=0)
        ood_preds = torch.cat(ood_preds).numpy().astype(int)

        # Sample
        n_id = min(n_sample, len(id_feats))
        n_ood = min(n_sample, len(ood_feats))
        id_idx = np.random.choice(len(id_feats), n_id, replace=False)
        ood_idx = np.random.choice(len(ood_feats), n_ood, replace=False)

        all_feats = np.concatenate([id_feats[id_idx], ood_feats[ood_idx]], axis=0)
        all_preds = np.concatenate([id_preds[id_idx], ood_preds[ood_idx]], axis=0)
        labels = np.concatenate([np.ones(n_id), np.zeros(n_ood)])

        # Batch-compute intra_sim
        N = len(all_feats)
        intra_scores = np.zeros(N)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        for c in np.unique(all_preds):
            c_mask = all_preds == c
            if c not in pp.cluster_features or len(pp.cluster_features[c]) == 0:
                continue
            c_data = all_feats[c_mask]
            c_indices = np.where(c_mask)[0]
            centers = pp.cluster_centers[c]
            c_t = torch.tensor(c_data, device=device).float()
            sim_to_centers = (c_t @ torch.tensor(centers, device=device).float().T).argmax(dim=1).cpu().numpy()

            for nc in range(len(centers)):
                nc_mask = sim_to_centers == nc
                if nc_mask.sum() == 0:
                    continue
                nc_feats = pp.cluster_features[c][nc]
                k_eff = min(pp.K1, len(nc_feats))
                batch_sim = batched_matrix_multiply(nc_feats, c_data[nc_mask], K=k_eff)
                intra_scores[c_indices[nc_mask]] = batch_sim

        # Check distributions
        id_intra = intra_scores[labels == 1]
        ood_intra = intra_scores[labels == 0]
        print(f"  intra_sim: ID mean={id_intra.mean():.4f} std={id_intra.std():.4f}")
        print(f"  intra_sim: OOD mean={ood_intra.mean():.4f} std={ood_intra.std():.4f}")
        print(f"  ID-OOD gap: {id_intra.mean() - ood_intra.mean():.4f}")

        # AUROC using intra_sim only
        fpr, tpr, _ = metrics.roc_curve(labels, intra_scores)
        auroc = metrics.auc(fpr, tpr)
        print(f"  AUROC (intra only): {100*max(auroc, 1-auroc):.2f}% (raw={100*auroc:.2f}%)")

        # Also check: global ID similarity for comparison
        global_scores = batched_matrix_multiply(pp.activation_log, all_feats, K=pp.K1)
        id_global = global_scores[labels == 1]
        ood_global = global_scores[labels == 0]
        print(f"  global_sim: ID mean={id_global.mean():.4f} OOD mean={ood_global.mean():.4f}")
        fpr2, tpr2, _ = metrics.roc_curve(labels, global_scores)
        auroc2 = metrics.auc(fpr2, tpr2)
        print(f"  AUROC (global sim): {100*max(auroc2, 1-auroc2):.2f}%")


if __name__ == '__main__':
    main()
