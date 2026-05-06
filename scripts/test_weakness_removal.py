"""
Test Weakness Removal KNN postprocessor on CIFAR-100 with ResNet-18.
"""
import sys, os, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # force CPU (must be before torch import)

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT_DIR)

import numpy as np
import torch
from tqdm import tqdm
from sklearn import metrics

from openood.utils.config import Config
from openood.datasets import get_dataloader, get_ood_dataloader
from openood.networks import get_network
from openood.postprocessors.weakness_removal_knn_postprocessor import (
    WeaknessRemovalKNNPostprocessor)
from openood.utils import setup_logger


def compute_auroc(id_conf, ood_conf):
    conf = np.concatenate([id_conf, ood_conf])
    label = np.concatenate([np.ones(len(id_conf)), -np.ones(len(ood_conf))])
    fpr, tpr, _ = metrics.roc_curve(label, conf)
    return max(metrics.auc(fpr, tpr), metrics.auc(tpr, fpr))


def main():
    config = Config(
        os.path.join(ROOT_DIR, 'configs/pipelines/test/test_ood.yml'),
        os.path.join(ROOT_DIR, 'configs/datasets/cifar100/cifar100.yml'),
        os.path.join(ROOT_DIR, 'configs/datasets/cifar100/cifar100_ood.yml'),
        os.path.join(ROOT_DIR, 'configs/networks/resnet18_32x32.yml'),
        os.path.join(ROOT_DIR, 'configs/postprocessors/weakness_removal_knn.yml'),
        os.path.join(ROOT_DIR, 'configs/preprocessors/base_preprocessor.yml'),
    )
    config.parse_refs()
    config.network.pretrained = True
    config.network.num_gpus = 0  # force CPU
    config.network.checkpoint = os.path.join(
        ROOT_DIR, 'checkpoint/cifar100_res18/s0/best.ckpt')

    import datetime
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    config.exp_name = (f"{config.dataset.name}_{config.network.name}_"
                       f"{config.pipeline.name}_ood_{config.postprocessor.name}_{ts}")
    config.output_dir = os.path.join(ROOT_DIR, 'results', config.exp_name)
    setup_logger(config)

    pp = WeaknessRemovalKNNPostprocessor(config)
    id_loader_dict = get_dataloader(config)
    ood_loader_dict = get_ood_dataloader(config)
    from openood.networks.resnet18_32x32 import ResNet18_32x32
    net = ResNet18_32x32(num_classes=100)
    ckpt = torch.load(config.network.checkpoint, map_location='cpu')
    net.load_state_dict(ckpt, strict=False)
    net.eval()

    # Setup
    print("Setting up...")
    t0 = time.time()
    pp.setup(net, id_loader_dict, ood_loader_dict)
    print(f"Setup: {time.time()-t0:.0f}s")

    # ID test
    print("Extracting ID test...")
    t0 = time.time()
    id_preds, id_labels = pp.acc_inference(net, id_loader_dict['test'])
    id_acc = (id_preds == id_labels).mean()
    print(f"ID acc: {100*id_acc:.2f}% ({time.time()-t0:.0f}s)")

    # Evaluate all OOD
    print("\n" + "=" * 70)
    for split in ['nearood', 'farood']:
        print(f"\n--- {split.upper()} ---")
        aurocs = []
        for name, dl in ood_loader_dict[split].items():
            t0 = time.time()
            id_c, ood_p, ood_c, ood_l = pp.inference(net, dl, name)
            auroc = compute_auroc(id_c, ood_c)
            aurocs.append(auroc)
            print(f"  {name}: {100*auroc:.2f}% ({time.time()-t0:.0f}s)")
        print(f"  Mean: {100*np.mean(aurocs):.2f}%")


if __name__ == '__main__':
    main()
