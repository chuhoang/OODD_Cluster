# OODD: Test-time Out-of-Distribution Detection with Dynamic Dictionary
Yifeng Yang, Lin Zhu, Zewen Sun, Hengyu Liu, Qinying Gu, Nanyang Ye 
xxxxxxxxxxxxxxxxxxxx
[CVPR2025] The source code of "[OODD: Test-time Out-of-Distribution Detection with Dynamic Dictionary](https://arxiv.org/pdf/2503.10468)".
<p align="center">
  <img src="img/ov.png" width=80%/>
</p>  
Abstract: 
Out-of-distribution (OOD) detection remains challenging for deep learning models, particularly when test-time OOD samples differ significantly from training outliers. We propose \textbf{OODD}, a novel test-time OOD detection method that dynamically maintains and updates an OOD dictionary without fine-tuning. Our approach leverages a priority queue-based dictionary that accumulates representative OOD features during testing, combined with an informative inlier sampling strategy for in-distribution (ID) samples. To ensure stable performance during early testing, we propose a dual OOD stabilization mechanism that leverages strategically generated outliers derived from ID data. To our best knowledge, extensive experiments on the OpenOOD benchmark demonstrate that OODD significantly outperforms existing methods, achieving a 26.0\% improvement in FPR95 on CIFAR-100 Far OOD detection compared to the state-of-the-art approach. Furthermore, we present an optimized variant of the KNN-based OOD detection framework that achieves a 3x speedup while maintaining detection performance.

## Setup
Please refer to [OpenOOD](https://github.com/Jingkang50/OpenOOD/blob/main/README.md) for the CIFAR/ImageNet datasets and instructions on installing the OpenOOD benchmark.

## Running the code
### Download the checkpoints
The checkpoints (forked from OpenOOD) can be downloaded from this [link](https://drive.google.com/file/d/1vdwQoAfxBnIG43SDFIm2hCrDv3eA7uwv/view?usp=drive_link). Please download the checkpoints and put them in the `checkpoint` folder.
### Running 
```bash
cd OODD
bash bash.sh
```
## MCM + OODD
Please refer to `MCM+OODD_README.md` in the `MCM` folder.
## NegLabel + OODD
Please refer to `README.md` in the `NegLabel` folder.

## Weakness Removal KNN (WR-KNN) Postprocessor

Run the WR-KNN postprocessor as a standalone script for OOD detection.

### Wood Dataset

```bash
# Run WR-KNN on wood_ood dataset
python -m openood.postprocessors.weakness_removal_knn_postprocessor \
    --dataset wood \
    --checkpoint /path/to/wood_checkpoint.ckpt \
    --gpu 0 \
    --K1 10 --K2 5 \
    --weak_remove_ratio 0.15 \
    --queue_size 512 \
    --batch_size 64
```

### CIFAR-100

```bash
# Run WR-KNN on CIFAR-100 (uses default checkpoint)
python -m openood.postprocessors.weakness_removal_knn_postprocessor \
    --dataset cifar100 \
    --gpu 0 \
    --K1 10 --K2 5 \
    --weak_remove_ratio 0.15 \
    --queue_size 512
```

### Using the main.py pipeline (alternative)

```bash
# Wood dataset via main.py
python main.py \
    --config configs/datasets/wood/wood.yml \
    configs/datasets/wood/wood_ood.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/weakness_removal_knn.yml \
    --network.pretrained True \
    --network.checkpoint /path/to/wood_checkpoint.ckpt \
    --postprocessor.postprocessor_args.K1 10 \
    --postprocessor.postprocessor_args.K2 5 \
    --postprocessor.postprocessor_args.weak_remove_ratio 0.07 \
    --postprocessor.postprocessor_args.queue_size 512 \
    --merge_option merge
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `wood` | Dataset: `wood` or `cifar100` |
| `--checkpoint` | auto | Path to model checkpoint |
| `--gpu` | `0` | GPU device (`-1` for CPU) |
| `--K1` | `10` | K for intra-class ID scoring |
| `--K2` | `5` | K for OOD scoring |
| `--weak_remove_ratio` | `0.07` | Fraction of weak features removed per cluster |
| `--queue_size` | `512` | OOD memory bank queue size |
| `--k_min` | `1` | Minimum clusters for X-Means |
| `--k_max` | `4` | Maximum clusters for X-Means |
| `--batch_size` | `64` | Batch size for dataloaders |

## Acknowledgement
Our repo is developed based on [OpenOOD](https://github.com/Jingkang50/OpenOOD), [MCM](https://github.com/deeplearning-wisc/MCM), [NegLabel](https://github.com/XueJiang16/NegLabel).

## Contact us

For any additional questions, feel free to email maxwellquadyang@gmail.com .
