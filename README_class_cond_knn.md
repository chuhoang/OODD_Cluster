# Class-Conditional KNN OOD Detection

X-Means + KNN postprocessor for class-conditional out-of-distribution detection.

## Algorithm

1. **Setup**: Extract ID training features per class, retain top-α by MSP, cluster each class with X-Means (auto-selects K ∈ [1, 4] via PCA + inertia elbow)
2. **Inference**: For each test sample predicted as class c:
   - Compute K1-th nearest neighbor cosine similarity across all features of class c (intra_sim)
   - Maintain OOD memory bank via priority queue (keeps low intra_sim = most OOD-like)
   - Score = -intra_sim - ood_batch_score (higher = more ID-like)

## Quick Start

```bash
cd /home/hoangcm/Signlanguage/OODD

# Run full evaluation (CIFAR-100, ResNet-18, all OOD datasets)
python scripts/test_class_cond_knn.py
```

## Test Variants

```bash
# Compare baseline vs topk=3 vs zscore vs adaptive_k1
python scripts/test_variants.py
```

## Config

Key hyperparameters in `configs/postprocessors/class_cond_knn.yml`:

| Param | Default | Description |
|-------|---------|-------------|
| K1 | 10 | K-nearest neighbors for intra-class similarity |
| K2 | 5 | K-nearest neighbors for OOD distance |
| ALPHA | 0.5 | Fraction of ID training data retained per class |
| queue_size | 512 | OOD memory bank capacity |
| k_min | 1 | Minimum clusters per class (X-Means) |
| k_max | 4 | Maximum clusters per class (X-Means) |

## Results (CIFAR-100, ResNet-18 32x32)

| OOD Dataset | AUROC |
|------------|-------|
| **Near-OOD** | |
| TIN | 74.99% |
| CIFAR-10 | 82.07% |
| **Far-OOD** | |
| MNIST | 98.30% |
| SVHN | 90.62% |
| Texture | 67.73% |
| Places365 | 84.23% |

## Files

- `openood/postprocessors/class_cond_knn_postprocessor.py` — Postprocessor implementation
- `configs/postprocessors/class_cond_knn.yml` — Hyperparameter config
- `scripts/test_class_cond_knn.py` — Full evaluation script
- `scripts/test_variants.py` — Variant comparison
