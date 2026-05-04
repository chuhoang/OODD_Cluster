#!/usr/bin/env python3
"""
setup_wood.py — One-shot setup for the Wood dataset in the OpenOOD project.
Run from: Signlanguage/OODD/
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
print("Linking OOD imglists...")
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
print(f"\nNext: Run evaluation with CKNN (see README_class_conditional_knn.md).")
