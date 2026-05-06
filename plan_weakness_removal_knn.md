# Weakness Removal + ID/OOD Dictionary KNN OOD Detection Plan

## 1. Motivation

**Problems with the current class-conditional approach:**

1. **MSP filtering loses information**: Retaining only top-α (50%) by MSP confidence discards half the ID features. Low-confidence but correctly-labeled features carry useful structural information about the class manifold's boundary.

2. **OOD bank is unseeded**: Without auxiliary OOD data, the queue starts empty and the first batch self-contaminates (samples compare against themselves).

3. **Intra-class similarity is inverted**: Empirically, OOD samples have *higher* cosine similarity to their predicted class than ID test samples. This forces score negation, making the algorithm non-intuitive.

**Proposed solution:**

- Use **100% of ID features** (no MSP filtering), then remove only **structurally weak** features — the 7% farthest from their cluster centroid in each cluster.
- Recycle those **weak features as the initial OOD memory bank** — they represent "hard" or boundary samples that the model finds ambiguous.
- Score using the **original KNN pattern**: `id_score - ood_score` with consistent positive scores (HIGH = ID-like).

---

## 2. Algorithm Overview

### Phase 0: Why Weakness Removal?

```
Cluster centroid = mean of all features in that cluster

For each cluster:
  Feature A: 0.01 from centroid  ─┐
  Feature B: 0.02 from centroid   ├── Strong (93%) → ID Dictionary
  Feature C: 0.03 from centroid   │
  ...                              │
  Feature X: 0.15 from centroid  ─┤
  Feature Y: 0.18 from centroid   ─┼── Weak (7%) → OOD Dictionary seed
  Feature Z: 0.22 from centroid  ─┘

Intuition:
- Strong features = tight, representative core of the class
- Weak features = boundary/outlier samples → look "OOD-like" to the model
  → natural candidates for the OOD memory bank
```

### Phase 1: Setup — Clustering + Weakness Removal

```
For each class c in [0, C-1]:
  1. Collect ALL training features with ground-truth label == c
     (100% of features, no MSP filtering)

  2. Apply X-Means to determine K_c clusters per class

  3. For each cluster k in class c:
     a. Compute centroid: μ_{c,k} = mean of all features in cluster
     b. Compute distance to centroid for each feature:
        d_i = || f_i - μ_{c,k} ||_2   (Euclidean, since features are L2-normalized)
     c. Sort by d_i descending (highest = weakest)
     d. Remove top r% (r=7) → Weak Features   → go to Phase 4 (OOD bank)
     e. Keep bottom (100-r)%     → Strong Features → go to ID Dictionary

  4. Aggregate:
     - ID Dictionary (F_train): all strong features across ALL classes/clusters

### Phase 2: OOD Memory Bank Init — Weakest-of-Weak + Cutout

```
Step 1: From each cluster's 7% weakest features, select only the bottom 50%
        (= ~3.5% of original features → weakest-of-the-weak)
        These are the features with LOWEST cosine similarity to the cluster centroid.

Step 2: For each selected weak feature:
        a. Look up the source image (image index tracked during setup)
        b. Apply Cutout augmentation (1 hole, 16×16) to create a corrupted version
        c. Forward through model → extract feature (L2-normalize)
        d. Add to OOD memory bank

Step 3: No PriorityQueue pruning needed — the direct selection already picks
        the absolute weakest features. Cutout further ensures they are OOD-like.

Result:
  - ID Dictionary: ~93% strong features per class
  - OOD Memory Bank: cutout features from ~3.5% weakest-of-weak (= 50% of 7%)
```

**Why 50% of 7%:** Only the absolute weakest matter for OOD. Taking all 7% is too many
(~17K features for CIFAR-100); 50% of 7% gives ~8K features, cutout doubles to ~16K.

**Why Cutout:** A weak feature already at the cluster boundary, with cutout corruption,
becomes even less like any ID class → strong OOD signal.

**Why no queue cutoff:** The direct selection by centroid distance already picks the
most OOD-like features. Queue pruning would be redundant — the sorting IS the selection.

### Phase 3: Inference — Class-Conditional Scoring

```
For each test batch:
  1. Forward pass → (output, feature) for each sample
     Also get predicted class c_i for each sample i

  2. For each sample i (predicted as class c_i):
     a) ID distance (class-conditional):
        - Compute K1-th largest cosine similarity to strong features
          of predicted class c_i ONLY
        - id_sim[i] = K1-th largest cos_sim(f_i, F_train[c_i])

     b) OOD distance:
        - Compute K2-th largest cosine similarity to OOD memory bank
        - ood_sim[i] = K2-th largest cos_sim(f_i, OOD_bank)

     c) Score (same pattern as original KNN):
        score[i] = id_sim[i] - ood_sim[i]
        - HIGH score → close to predicted class, far from OOD → ID
        - LOW score → far from predicted class, close to OOD → OOD

  3. Update OOD queue dynamically (same pattern as original KNN):
     - Compute global ID similarity: global_sim = batched_matrix_multiply(F_train, batch_data, K1)
     - queue.put(ScoreData(global_sim[j], batch_data[j])) for each sample
     - get() removes HIGHEST → keeps LOW global_sim = OOD-like
     - Queue starts pre-seeded with weakest-of-weak cutout features (Phase 2)
     - Grows adaptively during inference with new OOD-looking test features
```

---

## 3. Detailed Design

### 3.1 New Class: `WeaknessRemovalKNNPostprocessor`

Inherits from `BasePostprocessor`. Lives in `OODD/openood/postprocessors/weakness_removal_knn_postprocessor.py`.

**`__init__` parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `K1` | int | 10 | K for intra-class ID distance |
| `K2` | int | 5 | K for OOD distance |
| `weak_remove_ratio` | float | 0.07 | Fraction of farthest features removed per cluster (7%) |
| `queue_size` | int | 512 | Size of OOD memory bank priority queue |
| `k_min` | int | 1 | Minimum clusters per class (X-Means) |
| `k_max` | int | 4 | Maximum clusters per class (< 5) |

**Internal state:**

| Field | Description |
|-------|-------------|
| `strong_features` | dict: class_id → (N_c, D) array of strong ID features (93%) |
| `weak_features` | (N_weak, D) array of all weak features (7% → OOD seed) |
| `cluster_centers` | dict: class_id → (K_c, D) centroids |
| `activation_log` | (N_strong, D) all strong features (global ID reference for queue) |
| `id_feature` / `id_pred` | ID test features and predictions |

### 3.2 Setup Phase (`setup` method)

```python
def setup(self, net, id_loader_dict, ood_loader_dict):
    # Step 1: Extract ALL features per class (100%, no MSP filtering)
    class_features = {c: [] for c in range(num_classes)}
    for batch in id_loader_dict['train']:
        features = extract_features(net, batch)  # from data_aux, best per sample
        labels = batch['label']
        for c in range(num_classes):
            mask = labels == c
            class_features[c].append(features[mask])

    # Step 2: X-Means per class + weakness removal
    all_weak = []
    all_strong = []
    self.strong_features = {}
    self.weak_features_list = []

    for c in range(num_classes):
        feats = np.concatenate(class_features[c], axis=0)

        # X-Means clustering
        centers, labels, per_cluster = xmeans_cluster(feats, self.k_min, self.k_max)

        strong_c = []
        for nc in range(len(centers)):
            cluster_feats = per_cluster[nc]
            centroid = cluster_feats.mean(axis=0)

            # Hypersphere distance: use batched_matrix_multiply (cosine sim, K=1)
            # centroid as reference (1, D), cluster_feats as query → K-th largest cos_sim
            sims = batched_matrix_multiply(centroid[None, :], cluster_feats, K=1)
            # sims[i] = cos_sim(feature_i, centroid) — all distances on hypersphere
            # HIGH sim = close to centroid = strong feature
            # LOW sim = far from centroid = weak feature

            # Sort ascending: weakest (lowest sim) first, remove bottom 7%
            n_remove = max(1, int(len(cluster_feats) * self.weak_remove_ratio))
            sorted_idx = np.argsort(sims)  # ascending: lowest sim first = weakest
            weak_idx = sorted_idx[:n_remove]      # 7% lowest sim → OOD bank
            strong_idx = sorted_idx[n_remove:]     # 93% highest sim → ID dict

            all_weak.append(cluster_feats[weak_idx])
            strong_c.append(cluster_feats[strong_idx])

        self.strong_features[c] = np.concatenate(strong_c, axis=0) if strong_c else np.array([])
        all_strong.append(self.strong_features[c])

    self.activation_log = np.concatenate(all_strong, axis=0)   # Global ID ref

    # ---- Phase 2: OOD memory bank via weakest-of-weak + Cutout ----
    # From the 7% weak features, select bottom 50% (weakest-of-weak)
    # by taking the first half after sorting by centroid distance ascending
    weakest_of_weak_feats = []
    weakest_of_weak_idxs = []
    for c in range(num_classes):
        for nc in range(len(per_cluster)):
            cluster_feats = per_cluster[nc]
            centroid = cluster_feats.mean(axis=0)
            sims = batched_matrix_multiply(centroid[None,:], cluster_feats, K=1)
            n_weak = max(1, int(len(cluster_feats) * self.weak_remove_ratio))
            sorted_idx = np.argsort(sims)  # ascending: weakest first
            weak_idx = sorted_idx[:n_weak]                      # 7% weakest
            wow_idx = weak_idx[:max(1, n_weak // 2)]            # 50% of 7% = ~3.5%
            weakest_of_weak_feats.append(cluster_feats[wow_idx])
            weakest_of_weak_idxs.append(global_img_idxs[wow_idx])  # track source images

    # Apply Cutout to source images, extract OOD features
    self.ood_memory_bank = []
    for feat, img_idx in zip(weakest_of_weak_feats, weakest_of_weak_idxs):
        img = id_train_dataset[img_idx]
        cutout_img = apply_cutout(img, n_holes=1, length=16)
        feat_cutout = normalizer(net(cutout_img).cpu().numpy())
        self.ood_memory_bank.append(feat_cutout)
    self.ood_memory_bank = np.array(self.ood_memory_bank)
```

### 3.3 Inference Phase (`conf_postprocess` method)

```python
def conf_postprocess(self, food, ftest, preds_ood=None, preds_id=None):
    # ---- Seed OOD queue with weakest-of-weak cutout features ----
    queue = PriorityQueue()
    ood_seed_scores = batched_matrix_multiply(
        self.activation_log, self.ood_memory_bank, self.K1)
    for j in range(len(self.ood_memory_bank)):
        queue.put(ScoreData(ood_seed_scores[j], self.ood_memory_bank[j]))
        if queue.qsize() > self.queue_size:
            queue.get()
    memory_bank = None  # already seeded

    all_data = np.concatenate([ftest, food], axis=0)
    all_preds = np.concatenate([preds_id, preds_ood], axis=0)
    label_id = np.concatenate([np.ones(len(ftest)), np.zeros(len(food))])

    # Shuffle
    np.random.seed(100)
    idx = np.random.permutation(len(all_data))
    all_data, all_preds, label_id = all_data[idx], all_preds[idx], label_id[idx]

    scores_list = []
    for i in range(num_batches):
        batch_data = all_data[start:end]
        batch_preds = all_preds[start:end]

        # ---- 1. Class-conditional ID similarity ----
        intra_scores = np.zeros(len(batch_data))
        for c in np.unique(batch_preds):
            c_mask = batch_preds == c
            if c in self.strong_features and len(self.strong_features[c]) > 0:
                k = min(self.K1, len(self.strong_features[c]))
                intra_scores[np.where(c_mask)[0]] = batched_matrix_multiply(
                    self.strong_features[c], batch_data[c_mask], K=k)

        # ---- 2. Update OOD queue (same pattern as original KNN) ----
        global_id_scores = batched_matrix_multiply(
            self.activation_log, batch_data, self.K1)
        for j in range(len(batch_data)):
            queue.put(ScoreData(global_id_scores[j], batch_data[j]))
            if queue.qsize() > self.queue_size:
                queue.get()

        # ---- 3. OOD distance from updated queue ----
        new_food = np.array([item.data for item in list(queue.queue)])
        ood_batch_score = batched_matrix_multiply(new_food, batch_data, self.K2)

        # ---- 4. Final score ----
        batch_score = intra_scores - ood_batch_score
        scores_list.append(batch_score)

    scores_all = np.concatenate(scores_list)
    return scores_all[label_id == 1], scores_all[label_id == 0], label_id
```

---

## 4. Score Direction Analysis

Unlike the previous class-conditional approach where OOD had *higher* intra-class similarity (requiring negation), this method follows the original KNN pattern:

| Component | Direction | Meaning |
|-----------|-----------|---------|
| `global_id_scores` | HIGH = ID-like | Original KNN proven pattern |
| Queue keeps | LOW global sim | OOD-like features |
| `ood_batch_score` | HIGH = OOD-like | Similar to OOD bank |
| `intra_scores` (class-cond) | HIGH = ID-like | Close to predicted class's strong features |
| **Final score** | **HIGH = ID-like** | `intra - ood` |

The removal of weak features makes the ID dictionary more representative of the class core, which should make ID test samples have higher similarity to their class than OOD samples do. This corrects the previous inversion.

---

## 5. Key Design Decisions — Complete Summary

### 5.1 ID Dictionary Initialization (Phase 1)

```
For each class c (0..C-1):
  1. Collect ALL training features with ground-truth label == c (100%, no MSP filter)
  2. X-Means clustering: auto-select K_c ∈ [1, 4] (k_min=1, k_max=4, K_c < 5)
  3. For each cluster in class c:
     a. centroid = mean(cluster_features)
     b. sims = batched_matrix_multiply(centroid[None,:], cluster_features, K=1)
        → cosine similarity to centroid on L2-normalized hypersphere
        → LARGER sim = CLOSER to centroid = STRONGER feature
        → SMALLER sim = FARTHER from centroid = WEAKER feature
     c. Sort ascending (weakest first), split:
        - Top 7% lowest sim → WEAK features (go to Phase 2)
        - Bottom 93% highest sim → STRONG features (ID Dictionary)
  4. ID Dictionary F_train[c] = all strong features of class c
     Global ID ref: activation_log = concat(F_train[c] for all c)
```

### 5.2 OOD Memory Bank Initialization (Phase 2)

```
1. From the 7% weak features per cluster, select bottom 50% (weakest-of-weak)
   → ~3.5% of original features per cluster
   → These have the LOWEST cosine similarity to their cluster centroid

2. For each weakest-of-weak feature:
   a. Look up the source image (image index tracked during setup)
   b. Apply Cutout augmentation: mask 1 hole, 16×16 pixels
   c. Forward through model → extract feature → L2-normalize
   d. Add to OOD memory bank

3. Seed the queue with OOD memory bank features:
   ood_scores = batched_matrix_multiply(activation_log, ood_memory_bank, K1)
   for each feature:
       queue.put(ScoreData(ood_score, feature))
       if queue.qsize() > queue_size: queue.get()
   → Queue starts with most OOD-like cutout features (no cold start)
```

### 5.3 OOD Queue Update During Inference (Phase 3)

```
For each test batch:
  1. Forward pass → features + predicted classes

  2. Compute class-conditional ID score:
     For each sample i predicted as class c_i:
       intra_scores[i] = batched_matrix_multiply(
           F_train[c_i], feature[i], K=K1)
     → K1-th largest cosine similarity to predicted class's STRONG features
     → HIGH = close to predicted class → ID-like

  3. Update OOD queue (same pattern as original KNN):
     global_scores = batched_matrix_multiply(activation_log, batch_data, K1)
     for each sample j:
         queue.put(ScoreData(global_scores[j], batch_data[j]))
         if queue.qsize() > queue_size: queue.get()
     → get() removes HIGHEST global_score (most ID-like)
     → Queue KEEPS samples with LOW global_score (farthest from ID dict = OOD-like)
     → Queue ADAPTS: test OOD features replace weaker seed features over time

  4. Compute OOD score from updated queue:
     ood_features = all features currently in queue
     ood_scores = batched_matrix_multiply(ood_features, batch_data, K=K2)
     → K2-th largest cosine similarity to OOD queue features
     → HIGH = close to OOD bank → OOD-like
     → LOW = far from OOD bank → ID-like

  5. Final score:
     score[i] = intra_scores[i] - ood_scores[i]
     → ID sample:  HIGH intra (close to class) - LOW ood (far from OOD)  = HIGH ✓
     → OOD sample: LOW intra (far from class)  - HIGH ood (close to OOD) = LOW  ✓
     → HIGHER score = MORE ID-like (follows original KNN convention)
```

### 5.4 Distance Convention Summary

| Step | Function | K | Reference | Query | Meaning of HIGH value |
|------|----------|---|-----------|-------|----------------------|
| Weakness detection | `batched_matrix_multiply` | 1 | centroid (1,D) | cluster features | Close to centroid = STRONG |
| ID scoring | `batched_matrix_multiply` | K1 | F_train[c] | test feature | Close to predicted class = ID-like |
| Queue update | `batched_matrix_multiply` | K1 | activation_log | batch data | Close to ID dict → get() REMOVES |
| OOD scoring | `batched_matrix_multiply` | K2 | queue features | batch data | Close to OOD bank = OOD-like |

All distances use `batched_matrix_multiply` → cosine similarity on L2-normalized hypersphere.
**LARGER value = SMALLER angle = CLOSER. SMALLER value = LARGER angle = FARTHER.**

### 5.5 Score Direction: `id - ood` (NOT reverse)

```
score = intra_scores - ood_batch_score

  ID sample:  HIGH intra (close to class) - LOW ood (far from OOD)  = HIGH ✓
  OOD sample: LOW intra (far from class)  - HIGH ood (close to OOD) = LOW  ✓

No negation needed — follows the natural convention where HIGHER = more ID-like.
This assumes that with weak features removed, ID test samples have higher similarity
to their class's strong features than OOD samples do. Verify empirically.

---

## 6. Implementation Steps

1. Create `openood/postprocessors/weakness_removal_knn_postprocessor.py`
2. Add config `configs/postprocessors/weakness_removal_knn.yml`
3. Register in `utils.py` and `evaluation_api/postprocessor.py`
4. Create test script `scripts/test_weakness_removal.py`
5. Implement **cutout augmentation** on weak feature images (use `torchvision.transforms.RandomErasing` or manual cutout with `n_holes=1, length=16`)
6. Run on CIFAR-100 ResNet-18 and compare against baselines

---

## 7. Expected Improvements vs Current Approach

| Aspect | Current Class-Cond KNN | Proposed Weakness Removal |
|--------|----------------------|--------------------------|
| ID feature selection | Top-α by MSP (50%) | 100% minus 7% structural outliers |
| Score direction | Inverted (needs negation) | Natural (HIGH = ID) |
| OOD bank seeding | Empty (or aux only) | Pre-seeded with weak features |
| Queue update metric | intra_scores (inconsistent) | global_id_scores (consistent) |
| Cold start | Yes (first batch self-contaminates) | No (pre-seeded) |

