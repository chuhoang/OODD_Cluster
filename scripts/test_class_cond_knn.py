"""
Test script for ClassConditionalKNNPostprocessor on CIFAR-100 with ResNet-18.

Usage:
    python scripts/test_class_cond_knn.py
"""
import os
import sys
import time
import numpy as np

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT_DIR)

import torch
from openood.utils.config import Config, merge_configs
from openood.datasets import get_dataloader, get_ood_dataloader
from openood.networks import get_network
from openood.postprocessors.class_cond_knn_postprocessor import ClassConditionalKNNPostprocessor
from openood.utils import setup_logger
from sklearn import metrics


def compute_auroc(id_conf, ood_conf):
    """Compute AUROC given ID and OOD confidence scores."""
    n_id = len(id_conf)
    n_ood = len(ood_conf)
    conf = np.concatenate([id_conf, ood_conf])
    label = np.concatenate([np.ones(n_id), -1 * np.ones(n_ood)])
    fpr, tpr, _ = metrics.roc_curve(label, conf)
    auroc = metrics.auc(fpr, tpr)
    return auroc


def main():
    # --- Build config ---
    config = Config(
        os.path.join(ROOT_DIR, 'configs/pipelines/test/test_ood.yml'),
        os.path.join(ROOT_DIR, 'configs/datasets/cifar100/cifar100.yml'),
        os.path.join(ROOT_DIR, 'configs/datasets/cifar100/cifar100_ood.yml'),
        os.path.join(ROOT_DIR, 'configs/networks/resnet18_32x32.yml'),
        os.path.join(ROOT_DIR, 'configs/postprocessors/class_cond_knn.yml'),
        os.path.join(ROOT_DIR, 'configs/preprocessors/base_preprocessor.yml'),
    )
    config.parse_refs()

    # Override checkpoint path and enable loading
    config.network.pretrained = True
    config.network.checkpoint = os.path.join(
        ROOT_DIR, 'checkpoint/cifar100_res18/s0/best.ckpt')
    config.output_dir = os.path.join(ROOT_DIR, 'results')
    # Use a unique exp name with timestamp to avoid interactive prompt
    import datetime
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    config.exp_name = (
        f"{config.dataset.name}_{config.network.name}_{config.pipeline.name}"
        f"_ood_{config.postprocessor.name}_{timestamp}"
    )
    config.output_dir = os.path.join(config.output_dir, config.exp_name)

    # --- Setup logger ---
    setup_logger(config)

    # --- Load data ---
    print('Loading data loaders...')
    id_loader_dict = get_dataloader(config)
    ood_loader_dict = get_ood_dataloader(config)

    # --- Load model ---
    print('Loading model...')
    net = get_network(config.network)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.eval()

    # --- Setup postprocessor ---
    print('Setting up postprocessor...')
    postprocessor = ClassConditionalKNNPostprocessor(config)
    t0 = time.time()
    postprocessor.setup(net, id_loader_dict, ood_loader_dict)
    print(f'Setup took {time.time() - t0:.1f}s')
    print(f'Clusters per class: {postprocessor.num_clusters_found}')

    # --- ID test features ---
    print('\nExtracting ID test features...')
    t0 = time.time()
    id_pred, id_labels = postprocessor.acc_inference(
        net, id_loader_dict['test'])
    id_acc = (id_pred == id_labels).mean()
    print(f'ID Accuracy: {id_acc:.4f}')
    print(f'ID test features: {postprocessor.id_feature.shape}')
    print(f'Extraction took {time.time() - t0:.1f}s')

    # --- Evaluate OOD ---
    print('\n' + '=' * 70)
    print('OOD DETECTION RESULTS')
    print('=' * 70)

    nearood_results = {}
    farood_results = {}

    for split_name in ['nearood', 'farood']:
        print(f'\n--- {split_name.upper()} ---')
        aurocs = []
        for dataset_name, ood_dl in ood_loader_dict[split_name].items():
            print(f'  Processing {dataset_name}...')
            t0 = time.time()
            id_conf, ood_pred, ood_conf, ood_gt = postprocessor.inference(
                net, ood_dl, dataset_name)
            auroc = compute_auroc(id_conf, ood_conf)
            aurocs.append(auroc)
            print(f'    {dataset_name}: AUROC = {100*auroc:.2f}%, '
                  f'time = {time.time()-t0:.1f}s')
            if split_name == 'nearood':
                nearood_results[dataset_name] = auroc
            else:
                farood_results[dataset_name] = auroc

        mean_auroc = np.mean(aurocs)
        print(f'  Mean {split_name} AUROC: {100*mean_auroc:.2f}%')

    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    for name, auroc in {**nearood_results, **farood_results}.items():
        print(f'  {name}: {100*auroc:.2f}%')
    print(f'  ID Accuracy: {100*id_acc:.2f}%')
    print(f'  Mean Near-OOD AUROC: {100*np.mean(list(nearood_results.values())):.2f}%')
    print(f'  Mean Far-OOD AUROC: {100*np.mean(list(farood_results.values())):.2f}%')


if __name__ == '__main__':
    main()
