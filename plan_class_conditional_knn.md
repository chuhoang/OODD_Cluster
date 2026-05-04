# Class-Conditional X-Means + KNN OOD Detection Plan

## 1. Motivation

**Problem with current KNN postprocessor:**

The current algorithm treats all ID (in-distribution) training features as one flat pool — it selects top-α highest-MSP samples globally and interleaves across classes. Then for any test input, it computes distance to the same global ID set regardless of which class the model predicts.

This is suboptimal because:
- A test image predicted as "dog" should be compared primarily against the "dog" class manifold, not against all classes equally.
- OOD detection should be **class-conditional**: is this sample OOD *with respect to the class it's predicted as*?
- A sample that is far from class A but close to class B may still be ID (just misclassified), while a sample far from its predicted class is genuinely OOD.

**Proposed solution:**

Cluster each ID class independently (via X-Means — automatically determines optimal K per class), then for a test input predicted as class `c`, compute:
- **Intra-class distance**: distance from the test feature to the nearest cluster(s) of class `c`
- **Inter/OOD distance**: distance to the OOD memory bank (retained from current approach)
- **Score**: `score = inter_distance - intra_distance` (or similar formulation)

---

## 2. Algorithm Overview

### Phase 1: Setup — Per-Class X-Means Clustering

```
For each class c in [0, C-1]:
  1. Collect all training features whose ground-truth label == c
  2. Apply X-Means to automatically determine K_c clusters:
     - Starts with K=2, iteratively splits clusters
     - Uses BIC (Bayesian Information Criterion) to decide accept/reject each split
     - Stops when no split improves BIC or K_max is reached
  3. Store:
     - cluster_centers[c]: (K_c, D) array of cluster centroids
     - cluster_assignments[c]: cluster assignment for each sample
     - (optional) cluster_radii[c]: max distance from center to assigned points
```

**Why X-Means over K-Means:**
- No need to guess `K` per class — classes with tight manifolds naturally get 1-2 clusters, diverse classes get more
- BIC naturally penalizes over-splitting, so you don't overfit to noise
- Handles the tiny-class edge case gracefully: if `N_c` is small, BIC keeps K low (or even K=1)
- The cluster count per class becomes a useful signal itself: classes that need many clusters may be harder to detect OOD for

**X-Means parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k_min` | 1 | Minimum clusters per class |
| `k_max` | `min(4, floor(N_c / 2))` | Maximum clusters per class, capped at 4 (< 5) and bounded by data |
| `splitting_method` | BIC | Criterion for accepting a split (BIC or AIC) |


### Phase 2: OOD Memory Bank (kept from current approach)

- If auxiliary OOD data is available (e.g., T-out), compute features and store in a priority queue keyed by distance-to-ID, same as current `conf_postprocess`.
- This queue is dynamically updated during inference.
- `memory_bank`: top-N most ID-like OOD samples (high recall against ID).

### Phase 3: Inference — Class-Conditional Scoring

```
For each test batch:
  1. Forward pass → (predicted_class, feature)
  2. For each sample i with predicted class c_i:

     a) Intra-class distance (to predicted class):
        - Find nearest cluster center in class c_i
        - dist_intra[i] = min_{k in clusters(c_i)} || feature[i] - center_k ||

     b) Inter/OOD distance:
        - Compute K2 nearest neighbors from OOD memory bank
        - dist_ood[i] = mean distance to K2 nearest OOD samples
        (OR use the current batched_matrix_multiply K-th largest approach)

     c) Score:
        - score[i] = dist_ood[i] - α * dist_intra[i]
        - High score → likely ID (close to its predicted class, far from OOD)
        - Low score → likely OOD (far from its predicted class, or close to OOD)
```

---

## 3. Detailed Design

### 3.1 New Class: `ClassConditionalKNNPostprocessor`

Inherits from `BasePostprocessor`. Lives in `OODD/openood/postprocessors/class_cond_knn_postprocessor.py`.

**`__init__` parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `K1` | int | 5 | K for intra-class KNN (nearest ID samples within predicted class) |
| `K2` | int | 10 | K for OOD distance computation |
| `ALPHA` | float | 0.5 | Fraction of ID data to retain (by MSP confidence) |
| `queue_size` | int | 200 | Size of OOD memory bank priority queue |
| `k_min` | int | 1 | Minimum clusters per class (X-Means starts here) |
| `k_max` | int | 4 | Maximum clusters per class (capped at 4, i.e. < 5) |
| `intra_weight` | float | 1.0 | Weight α on intra-class distance in score formula |
| `use_centroid_only` | bool | True | If True, use cluster centroids only. If False, use all retained ID samples within predicted class |

**Internal state:**

| Field | Description |
|-------|-------------|
| `cluster_centers` | dict: class_id → (K_c, D) numpy array of centroids (K_c varies per class) |
| `cluster_assignments` | dict: class_id → numpy array of cluster indices per sample |
| `class_features` | dict: class_id → (N_c, D) numpy array of retained features |
| `activation_log` | (N_retained, D) all retained ID features (for backward compat / fallback) |
| `ood_memory_bank` | PriorityQueue of OOD features (same as current) |
| `aux_feature` | dict of auxiliary OOD features (same as current) |
| `num_clusters_found` | dict: class_id → int (K_c determined by X-Means; informative/debugging) |

### 3.2 Setup Phase (`setup` method)

```python
def setup(self, net, id_loader_dict, ood_loader_dict):
    # Step 1: Extract features per class from ID training data
    class_features = {c: [] for c in range(num_classes)}
    class_msp = {c: [] for c in range(num_classes)}

    for batch in id_loader_dict['train']:
        features = extract_features(net, batch)  # (B, D)
        labels = batch['label']
        msps = compute_msp(net, batch)           # (B,)

        for c in range(num_classes):
            mask = labels == c
            class_features[c].append(features[mask])
            class_msp[c].append(msps[mask])

    # Step 2: For each class, sort by MSP, retain top-α, cluster with X-Means
    for c in range(num_classes):
        feats = concat(class_features[c])        # (N_c, D)
        msps  = concat(class_msp[c])             # (N_c,)
        n_keep = max(self.k_min, int(ALPHA * len(feats)))

        # Keep top-α highest confidence samples
        idx = argsort(msps, descending)[:n_keep]
        feats_retained = feats[idx]

        # X-Means clustering (auto-selects optimal K ≤ k_max)
        k_max_effective = min(self.k_max, n_keep // 2)
        xmeans = XMeans(
            k_min=self.k_min,
            k_max=k_max_effective,
            splitting_criterion='BIC',
            random_state=42
        )
        xmeans.fit(feats_retained)

        self.cluster_centers[c] = xmeans.cluster_centers_
        self.class_features[c] = feats_retained
        self.cluster_assignments[c] = xmeans.labels_
        self.num_clusters_found[c] = len(xmeans.cluster_centers_)  # dynamic K_c per class

    # Step 3: Initialize OOD memory bank (same as current)
    if aux_loader available:
        self.aux_feature = self.get_auxiliary_data(net, aux_loader)
```

### 3.3 Inference Phase (`conf_postprocess` method)

```python
def conf_postprocess(self, food, ftest, preds_test):
    """
    food: OOD features (N_ood, D)
    ftest: test features (N_test, D)
    preds_test: predicted class for each test sample (N_test,)
    """
    # Step 1: Update OOD memory bank (same queue mechanism as current)
    queue = PriorityQueue()
    # ... process food through queue ...

    # Step 2: For each test sample, compute class-conditional score
    scores = []
    for i, (feat, pred_c) in enumerate(zip(ftest, preds_test)):
        # Intra-class: distance to nearest centroid of predicted class
        centers = self.cluster_centers[pred_c]       # (K_c, D) — K_c varies per class
        dist_intra = min(||feat - center|| for center in centers)

        # OOD distance: K-th nearest neighbor in OOD memory bank
        ood_features = get_all_from_queue(queue)
        dist_ood = kth_nearest_distance(feat, ood_features, K2)

        # Score
        score = dist_ood - intra_weight * dist_intra
        scores.append(score)

    return scores_id, scores_ood
```

### 3.4 Score Formulation Options

| Variant | Formula | Intuition |
|---------|---------|-----------|
| **V1 (simple)** | `dist_ood - α * dist_intra` | Linear combination |
| **V2 (ratio)** | `dist_ood / (dist_intra + ε)` | Scale-invariant |
| **V3 (multi-cluster)** | `dist_ood - α * mean(dist_to_top_m_clusters)` | Smooth over multiple nearby clusters |
| **V4 (density)** | `dist_ood / max(dist_intra, r_c)` where `r_c` = cluster radius | Accounts for class spread |

**Recommendation**: Start with V1, benchmark against V2.

---

## 4. Implementation Steps

### Step 1: Create new postprocessor file
- `OODD/openood/postprocessors/class_cond_knn_postprocessor.py`
- Class `ClassConditionalKNNPostprocessor(BasePostprocessor)`
- Dependency: `pyclustering` library for X-Means (`pip install pyclustering`)

### Step 2: Implement `setup()`
- Extract per-class features from `id_loader_dict['train']`
- Sort by MSP, retain top-α per class
- Run X-Means per class, store centroids (dynamic K_c per class)
- Cache results to `./cache/{dataset}_class_clusters.npz`

### Step 3: Implement `conf_postprocess()`
- OOD memory bank logic (ported from current `KNNPostprocessor`)
- Class-conditional distance computation
- Vectorized batch implementation (avoid per-sample loop where possible)

### Step 4: Implement `inference()` and `acc_inference()`
- Same caching pattern as current KNN postprocessor
- Pass predictions through to `conf_postprocess` for class-conditional scoring

### Step 5: Register in postprocessor factory
- Update `__init__.py` if there's a postprocessor registry

### Step 6: Add config
- Add default hyperparameters to the relevant config YAML

---

## 5. Key Design Decisions (to review)

1. **Cluster on ground-truth labels or predicted labels?**
   - Setup: cluster on **ground-truth** (we have labels for ID training data)
   - Inference: use **predicted** class to look up which centroids to compare against

2. **What if a class has too few samples for clustering?**
   - X-Means handles this: `k_max_effective = min(k_max, n_keep // 2)` ensures at least 2 samples per cluster
   - If `N_c < 2*k_min`, X-Means will stay at `k_min=1` — each sample compared directly to the single centroid
   - This is equivalent to "average feature" representation for tiny classes, which is reasonable

3. **Centroid-only vs. full retained samples for intra-class distance?**
   - Centroid-only: fast (O(K) per sample), coarser
   - Full samples: slower (O(N_c) per sample), finer grain
   - **Hybrid**: use centroid first, if `dist_intra > threshold`, fall back to K1 nearest retained samples

4. **Cache invalidation?**
   - Same pattern as current: cache per `(id_dataset, model_checkpoint)` key
   - Store centroids + class features in `.npz` files

---

## 6. Expected Improvements

| Aspect | Current KNN | Proposed |
|--------|-------------|----------|
| ID representation | Global flat pool, interleaved by class | Per-class clusters |
| OOD criterion | Global distance gap | Class-conditional distance gap |
| Sensitivity to class imbalance | Implicit via interleaving | Explicit per-class retention |
| Compute at inference | KNN × batch to global ID + queue | Centroid distance + KNN to queue |
| Interpretability | "Far from ID in general" | "Far from predicted class specifically" |

---

## 7. Open Questions

1. Should we merge intra-class distance from the **top-M nearest classes** (not just predicted class) to handle uncertainty? E.g., soft-class-conditional: weight by softmax probabilities.

2. Should the OOD memory bank also be organized per-class? (i.e., separate queue per class, populated by OOD samples that the model predicts as that class)

3. `k_max=4` caps clusters per class at < 5. Is this sufficient for all class manifolds, or should classes with very high intra-class variance be allowed more?
