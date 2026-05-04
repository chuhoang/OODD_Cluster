# How To Calculate AUROC For A New Dataset — Step By Step

This guide walks you through running OOD detection with **Class-Conditional KNN (CKNN)** on the **Wood** dataset and computing AUROC, using **CIFAR-100** as the base ID dataset for comparison.

---

## Wood Dataset Overview

The Wood dataset contains microscope images of wood species, organized into 3 splits:

| Split | Classes | Images | Description |
|---|---|---|---|
| `train/` | 130 | 38,492 | ID training set — wood species to train on |
| `val/` | 130 (same as train) | 9,011 | ID test set — same species, held-out images |
| `ood_val/` | 130 (disjoint from train) | 9,359 | OOD test set — **different** wood species, never seen during training |

The OOD split has **zero class overlap** with the train/val splits. This tests whether the model can recognize when it sees a wood species it was never trained on.

Source directory: `OOD_wood/` (provided by user, located at `../OOD_wood/` relative to this project).

Base ID comparison dataset: **CIFAR-100** (100 classes, 32x32 images).

---

## Prerequisites

- Python environment with `openood` installed (`pip install -e .` from the project root)
- A trained CIFAR-100 checkpoint: `checkpoint/resnet18_cifar100.ckpt`
- The Wood dataset at `../OOD_wood/` (or adjust the path in the script below)

---

## Step 1: Setup Wood Dataset (One-Shot Script)

Run this Python script from the project root (`sign_language/OODD/`). It will:
1. Copy/symlink Wood images into `data/images_classic/wood/`
2. Generate all imglist files (`train_wood.txt`, `test_wood.txt`, `test_wood_ood.txt`)
3. Create imglist entries for CIFAR-100/TIN/MNIST/SVHN as OOD datasets for Wood
4. Create the YAML config files `configs/datasets/wood/wood.yml` and `wood_ood.yml`
5. Add Wood normalization to `transform.py` (if not already present)

```python
#!/usr/bin/env python3
"""
setup_wood.py — One-shot setup for the Wood dataset in the OpenOOD project.
Run from: sign_language/OODD/
Wood source: ../OOD_wood/
"""
import os
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent          # OODD/
WOOD_SRC = BASE.parent / "OOD_wood"             # Wood dataset source
IMG_DIR = BASE / "data" / "images_classic" / "wood"
IMGLIST_DIR = BASE / "data" / "benchmark_imglist" / "wood"
CONFIG_DIR = BASE / "configs" / "datasets" / "wood"
CIFAR10_IMGLIST = BASE / "data" / "benchmark_imglist" / "cifar10"
CIFAR100_IMGLIST = BASE / "data" / "benchmark_imglist" / "cifar100"

# ---------- 1. Copy Wood images ----------
print("Copying wood images...")
if IMG_DIR.exists():
    shutil.rmtree(IMG_DIR)
shutil.copytree(WOOD_SRC, IMG_DIR)
print(f"  -> {IMG_DIR}")

# ---------- 2. Create imglist directories ----------
IMGLIST_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 3. Generate imglist for ID train ----------
print("Generating train_wood.txt ...")
train_dir = IMG_DIR / "train"
classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
class_to_idx = {cls: i for i, cls in enumerate(classes)}

with open(IMGLIST_DIR / "train_wood.txt", "w") as f:
    for cls_name, idx in class_to_idx.items():
        for img_path in (train_dir / cls_name).iterdir():
            if img_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                rel = str(img_path.relative_to(IMG_DIR))
                f.write(f"{rel} {idx}\n")
print(f"  -> {IMGLIST_DIR / 'train_wood.txt'}  ({len(class_to_idx)} classes)")

# ---------- 4. Generate imglist for ID test (val split) ----------
print("Generating test_wood.txt ...")
test_dir = IMG_DIR / "val"
with open(IMGLIST_DIR / "test_wood.txt", "w") as f:
    for cls_name, idx in class_to_idx.items():
        cls_dir = test_dir / cls_name
        if cls_dir.is_dir():
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    rel = str(img_path.relative_to(IMG_DIR))
                    f.write(f"{rel} {idx}\n")
print(f"  -> {IMGLIST_DIR / 'test_wood.txt'}")

# ---------- 5. Generate imglist for OOD test (ood_val split, label=-1) ----------
print("Generating test_wood_ood.txt ...")
ood_dir = IMG_DIR / "ood_val"
with open(IMGLIST_DIR / "test_wood_ood.txt", "w") as f:
    for cls_dir in ood_dir.iterdir():
        if cls_dir.is_dir():
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    rel = str(img_path.relative_to(IMG_DIR))
                    f.write(f"{rel} -1\n")
print(f"  -> {IMGLIST_DIR / 'test_wood_ood.txt'}")

# ---------- 6. Symlink OOD dataset imglists for Wood-as-ID evaluation ----------
# When Wood is ID, these datasets act as OOD:
for name, src in [("test_cifar100.txt", CIFAR100_IMGLIST / "test_cifar100.txt"),
                   ("test_cifar10.txt",  CIFAR10_IMGLIST  / "test_cifar10.txt"),
                   ("test_tin.txt",      CIFAR10_IMGLIST  / "test_tin.txt"),
                   ("test_mnist.txt",    CIFAR10_IMGLIST  / "test_mnist.txt"),
                   ("test_svhn.txt",     CIFAR10_IMGLIST  / "test_svhn.txt"),
                   ("test_texture.txt",  CIFAR10_IMGLIST  / "test_texture.txt"),
                   ("test_places365.txt",CIFAR10_IMGLIST  / "test_places365.txt")]:
    dst = IMGLIST_DIR / name
    if not dst.exists():
        os.symlink(os.path.relpath(src, IMGLIST_DIR), dst)
        print(f"  symlink: {dst} -> {src}")

# ---------- 7. Create wood.yml config ----------
print("Creating wood.yml ...")
wood_yml = f"""dataset:
  name: wood
  num_classes: {len(class_to_idx)}
  image_size: 224
  pre_size: 256

  interpolation: bilinear
  normalization_type: imagenet

  num_workers: '@{{num_workers}}'
  num_gpus: '@{{num_gpus}}'
  num_machines: '@{{num_machines}}'

  split_names: [train, val, test]

  train:
    dataset_class: ImglistDataset
    data_dir: ./data/images_classic/
    imglist_pth: ./data/benchmark_imglist/wood/train_wood.txt
    batch_size: 64
    shuffle: True
  val:
    dataset_class: ImglistDataset
    data_dir: ./data/images_classic/
    imglist_pth: ./data/benchmark_imglist/wood/test_wood.txt
    batch_size: 64
    shuffle: False
  test:
    dataset_class: ImglistDataset
    data_dir: ./data/images_classic/
    imglist_pth: ./data/benchmark_imglist/wood/test_wood.txt
    batch_size: 64
    shuffle: False
"""
with open(CONFIG_DIR / "wood.yml", "w") as f:
    f.write(wood_yml)
print(f"  -> {CONFIG_DIR / 'wood.yml'}")

# ---------- 8. Create wood_ood.yml config ----------
print("Creating wood_ood.yml ...")
wood_ood_yml = f"""ood_dataset:
  name: wood_ood
  num_classes: {len(class_to_idx)}

  num_workers: '@{{num_workers}}'
  num_gpus: '@{{num_gpus}}'
  num_machines: '@{{num_machines}}'

  dataset_class: ImglistDataset
  batch_size: 64
  shuffle: False

  split_names: [val, nearood, farood, t_out]

  val:
    data_dir: ./data/images_classic/
    imglist_pth: ./data/benchmark_imglist/wood/test_cifar100.txt

  nearood:
    datasets: [wood_ood, cifar100]
    wood_ood:
      data_dir: ./data/images_classic/
      imglist_pth: ./data/benchmark_imglist/wood/test_wood_ood.txt
    cifar100:
      data_dir: ./data/images_classic/
      imglist_pth: ./data/benchmark_imglist/wood/test_cifar100.txt

  farood:
    datasets: [cifar10, tin, mnist, svhn, texture, places365]
    cifar10:
      data_dir: ./data/images_classic/
      imglist_pth: ./data/benchmark_imglist/wood/test_cifar10.txt
    tin:
      data_dir: ./data/images_classic/
      imglist_pth: ./data/benchmark_imglist/wood/test_tin.txt
    mnist:
      data_dir: ./data/images_classic/
      imglist_pth: ./data/benchmark_imglist/wood/test_mnist.txt
    svhn:
      data_dir: ./data/images_classic/
      imglist_pth: ./data/benchmark_imglist/wood/test_svhn.txt
    texture:
      data_dir: ./data/images_classic/
      imglist_pth: ./data/benchmark_imglist/wood/test_texture.txt
    places365:
      data_dir: ./data/images_classic/
      imglist_pth: ./data/benchmark_imglist/wood/test_places365.txt

  t_out:
    datasets: [cifar100, tin, cifar10, mnist, svhn, texture, places365]
    cifar100:
      data_dir: ./data/images_classic/
      imglist_pth: ./data/benchmark_imglist/my_cifar100/val_cifar10.txt
    tin:
      data_dir: ./data/images_classic/
      imglist_pth: ./data/benchmark_imglist/my_cifar100/val_tin.txt
    cifar10:
      data_dir: ./data/images_classic/
      imglist_pth: ./data/benchmark_imglist/my_cifar100/val_cifar10.txt
    mnist:
      data_dir: ./data/images_classic/
      imglist_pth: ./data/benchmark_imglist/my_cifar100/val_mnist.txt
    svhn:
      data_dir: ./data/images_classic/
      imglist_pth: ./data/benchmark_imglist/my_cifar100/val_svhn.txt
    texture:
      data_dir: ./data/images_classic/
      imglist_pth: ./data/benchmark_imglist/my_cifar100/val_texture.txt
    places365:
      data_dir: ./data/images_classic/
      imglist_pth: ./data/benchmark_imglist/my_cifar100/val_places365.txt
"""
with open(CONFIG_DIR / "wood_ood.yml", "w") as f:
    f.write(wood_ood_yml)
print(f"  -> {CONFIG_DIR / 'wood_ood.yml'}")

# ---------- 9. Add Wood normalization to transform.py ----------
print("Checking normalization_dict in transform.py ...")
transform_py = BASE / "openood" / "preprocessors" / "transform.py"
with open(transform_py, "r") as f:
    content = f.read()

if "'wood'" not in content:
    new_entry = "    'wood': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],  # ImageNet stats (224x224)\n"
    content = content.replace(
        "    'cars': [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],\n}",
        f"    'cars': [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],\n{new_entry}}}"
    )
    with open(transform_py, "w") as f:
        f.write(content)
    print("  -> Added 'wood' normalization entry")

print("\nDone! Wood dataset is ready.")
print(f"  Train classes: {len(class_to_idx)}")
print(f"  Train images:  {sum(1 for _ in open(IMGLIST_DIR/'train_wood.txt'))}")
print(f"  Test images:   {sum(1 for _ in open(IMGLIST_DIR/'test_wood.txt'))}")
print(f"  OOD images:    {sum(1 for _ in open(IMGLIST_DIR/'test_wood_ood.txt'))}")
print(f"\nNext: Run evaluation with CKNN (see README Step 3).")
```

Save this as `setup_wood.py` in the OODD directory, then run:

```bash
cd sign_language/OODD
python setup_wood.py
```

---

## Step 2: Create Config YAML Files (Manual Alternative)

If you prefer to create configs manually instead of using the script above:

### 2a. `configs/datasets/wood/wood.yml` — ID dataset

```yaml
dataset:
  name: wood
  num_classes: 130
  image_size: 224
  pre_size: 256

  interpolation: bilinear
  normalization_type: imagenet

  num_workers: '@{num_workers}'
  num_gpus: '@{num_gpus}'
  num_machines: '@{num_machines}'

  split_names: [train, val, test]

  train:
    dataset_class: ImglistDataset
    data_dir: ./data/images_classic/
    imglist_pth: ./data/benchmark_imglist/wood/train_wood.txt
    batch_size: 64
    shuffle: True
  val:
    dataset_class: ImglistDataset
    data_dir: ./data/images_classic/
    imglist_pth: ./data/benchmark_imglist/wood/test_wood.txt
    batch_size: 64
    shuffle: False
  test:
    dataset_class: ImglistDataset
    data_dir: ./data/images_classic/
    imglist_pth: ./data/benchmark_imglist/wood/test_wood.txt
    batch_size: 64
    shuffle: False
```

### 2b. `configs/datasets/wood/wood_ood.yml` — OOD datasets

```yaml
ood_dataset:
  name: wood_ood
  num_classes: 130

  num_workers: '@{num_workers}'
  num_gpus: '@{num_gpus}'
  num_machines: '@{num_machines}'

  dataset_class: ImglistDataset
  batch_size: 64
  shuffle: False

  split_names: [val, nearood, farood, t_out]

  val:
    data_dir: ./data/images_classic/
    imglist_pth: ./data/benchmark_imglist/wood/test_cifar100.txt

  nearood:
    datasets: [wood_ood, cifar100]
    wood_ood:
      data_dir: ./data/images_classic/
      imglist_pth: ./data/benchmark_imglist/wood/test_wood_ood.txt
    cifar100:
      data_dir: ./data/images_classic/
      imglist_pth: ./data/benchmark_imglist/wood/test_cifar100.txt

  farood:
    datasets: [cifar10, tin, mnist, svhn, texture, places365]
    cifar10:
      data_dir: ./data/images_classic/
      imglist_pth: ./data/benchmark_imglist/wood/test_cifar10.txt
    tin:
      data_dir: ./data/images_classic/
      imglist_pth: ./data/benchmark_imglist/wood/test_tin.txt
    mnist:
      data_dir: ./data/images_classic/
      imglist_pth: ./data/benchmark_imglist/wood/test_mnist.txt
    svhn:
      data_dir: ./data/images_classic/
      imglist_pth: ./data/benchmark_imglist/wood/test_svhn.txt
    texture:
      data_dir: ./data/images_classic/
      imglist_pth: ./data/benchmark_imglist/wood/test_texture.txt
    places365:
      data_dir: ./data/images_classic/
      imglist_pth: ./data/benchmark_imglist/wood/test_places365.txt

  t_out:
    datasets: [cifar100, tin, cifar10, mnist, svhn, texture, places365]
    cifar100:
      data_dir: ./data/images_classic/
      imglist_pth: ./data/benchmark_imglist/my_cifar100/val_cifar10.txt
    tin:
      data_dir: ./data/images_classic/
      imglist_pth: ./data/benchmark_imglist/my_cifar100/val_tin.txt
    cifar10:
      data_dir: ./data/images_classic/
      imglist_pth: ./data/benchmark_imglist/my_cifar100/val_cifar10.txt
    mnist:
      data_dir: ./data/images_classic/
      imglist_pth: ./data/benchmark_imglist/my_cifar100/val_mnist.txt
    svhn:
      data_dir: ./data/images_classic/
      imglist_pth: ./data/benchmark_imglist/my_cifar100/val_svhn.txt
    texture:
      data_dir: ./data/images_classic/
      imglist_pth: ./data/benchmark_imglist/my_cifar100/val_texture.txt
    places365:
      data_dir: ./data/images_classic/
      imglist_pth: ./data/benchmark_imglist/my_cifar100/val_places365.txt
```

### 2c. Add normalization in `openood/preprocessors/transform.py`

```python
normalization_dict = {
    # ... existing ...
    'wood': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],  # ImageNet stats
}
```

---

## Step 3: Run CKNN Evaluation

### 3a. Wood as ID, CIFAR-100 / other datasets as OOD

Uses `resnet18_224x224` (ImageNet-200 pretrained backbone, fine-tune on Wood for best results):

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --config configs/datasets/wood/wood.yml \
    --config configs/datasets/wood/wood_ood.yml \
    --config configs/networks/resnet18_224x224.yml \
    --config configs/pipelines/test/test_ood.yml \
    --config configs/preprocessors/base_preprocessor.yml \
    --config configs/postprocessors/cknn.yml \
    --num_workers 8 \
    --network.checkpoint checkpoint/resnet18_imagenet200.ckpt \
    --postprocessor.postprocessor_args.K1 1 \
    --postprocessor.postprocessor_args.K2 1 \
    --postprocessor.postprocessor_args.ALPHA 0.5 \
    --postprocessor.postprocessor_args.queue_size 128 \
    --merge_option merge
```

### 3b. CIFAR-100 as ID, Wood as OOD (add wood to cifar100_ood.yml)

Add this block under `farood:` in `configs/datasets/cifar100/cifar100_ood.yml`:

```yaml
    wood:
      data_dir: ./data/images_classic/
      imglist_pth: ./data/benchmark_imglist/cifar100/test_wood_ood.txt
```

Then generate the imglist and run:

```bash
# Generate imglist (Wood's ood_val as OOD against CIFAR-100)
cd data/benchmark_imglist/cifar100/
find ../../images_classic/wood/ood_val/ -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | while read f; do echo "${f#../../images_classic/} -1"; done > test_wood_ood.txt
cd ../../..

CUDA_VISIBLE_DEVICES=0 python main.py \
    --config configs/datasets/cifar100/cifar100.yml \
    --config configs/datasets/cifar100/cifar100_ood.yml \
    --config configs/networks/resnet18_32x32.yml \
    --config configs/pipelines/test/test_ood.yml \
    --config configs/preprocessors/base_preprocessor.yml \
    --config configs/postprocessors/cknn.yml \
    --num_workers 8 \
    --network.checkpoint checkpoint/resnet18_cifar100.ckpt \
    --postprocessor.postprocessor_args.K1 1 \
    --postprocessor.postprocessor_args.K2 1 \
    --postprocessor.postprocessor_args.ALPHA 0.5 \
    --postprocessor.postprocessor_args.queue_size 128 \
    --merge_option merge
```

---

## Step 4: Understand the Output

The evaluation prints per-dataset results and saves them to `results/<dataset>/<dataset>_ood/class_conditional_knn/ood.csv`.

Example output for Wood:

```
wood_ood:   FPR@95: XX.XX, AUROC: XX.XX, AUPR_IN: XX.XX, AUPR_OUT: XX.XX
cifar100:   FPR@95: XX.XX, AUROC: XX.XX, AUPR_IN: XX.XX, AUPR_OUT: XX.XX
cifar10:    FPR@95: XX.XX, AUROC: XX.XX, AUPR_IN: XX.XX, AUPR_OUT: XX.XX
tin:        FPR@95: XX.XX, AUROC: XX.XX, AUPR_IN: XX.XX, AUPR_OUT: XX.XX
...
```

| Metric | Meaning |
|---|---|
| **FPR@95** | False positive rate at 95% TPR (lower is better) |
| **AUROC** | Area under the ROC curve (higher is better, 0–100) |
| **AUPR_IN** | AUPR with ID as positive |
| **AUPR_OUT** | AUPR with OOD as positive |

- `wood_ood` row: How well CKNN separates the in-distribution wood species from the unseen wood species.
- `cifar100`, `cifar10`, `tin`, etc.: How well it separates wood from completely different image domains.

---

## Step 5: Tune CKNN Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `K1` | 1 | K nearest neighbors from training set for ID similarity |
| `K2` | 1 | K nearest neighbors from priority queue for OOD similarity |
| `ALPHA` | 0.5 | Fraction of training features kept (top confidence) |
| `queue_size` | 128 | Max size of the FIFO priority queue |

Override on command line:

```bash
--postprocessor.postprocessor_args.K1 5
--postprocessor.postprocessor_args.K2 5
--postprocessor.postprocessor_args.ALPHA 0.5
--postprocessor.postprocessor_args.queue_size 512
```

---

## Troubleshooting

**"File not found" in imglist:**
- Verify imglist paths are relative to `data_dir` (`./data/images_classic/`)
- Run: `head -3 data/benchmark_imglist/wood/train_wood.txt` and check the paths exist

**CUDA out of memory:**
- Reduce batch sizes: `--dataset.train.batch_size 32 --dataset.test.batch_size 32`

**Wood images are 224x224 but model expects 32x32:**
- Use `resnet18_224x224.yml` for 224x224 models
- Use `resnet18_32x32.yml` for 32x32 (CIFAR-scale) — resize Wood images first if using this

**No AUROC for `wood_ood`:**
- Check `wood_ood` is listed under `nearood.datasets` in `wood_ood.yml`
- Verify `data/benchmark_imglist/wood/test_wood_ood.txt` exists
