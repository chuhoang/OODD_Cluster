"""HoroPCA visualization: CIFAR-100 (ID) + MNIST (OOD) features from ResNet-HYPER.

Visualizes coarse (h_c) and fine (h_f) Lorentz embeddings on the Poincaré disk
using HoroPCA — PCA in the tangent space at the Fréchet mean.

Usage:
    python OODD/visualize_horopca_resnet.py

Output:
    OODD/results/horopca_resnet_cifar100_mnist.png
"""

import sys, os
from pathlib import Path

_THIS = Path(__file__).resolve()
_SL  = _THIS.parent.parent
sys.path.insert(0, str(_SL))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T, datasets as tv_datasets
from sklearn.decomposition import PCA
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from HDHM.models.resnet_hyper import ResNetHyper
from HDHM.losses.hyperbolic_loss import HyperbolicClassifier
from cifar100_hierarchy import COARSE_LABELS, FINE_TO_COARSE, FINE_LABELS

# ─── Config ──────────────────────────────────────────────────────────────────
CKPT_PATH   = "training/runs/resnet_hyper_r18_final/best.pth"
DATA_ROOT   = "sign_language/deep-hybrid-models/data"
OODD_ROOT   = "OODD/data"
OUT_DIR     = Path("OODD/results")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_MNIST     = 500           # MNIST OOD samples
N_CIFAR     = 1000          # CIFAR-100 ID samples (stratified across coarse)
COARSE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]
MNIST_COLOR  = "#333333"
MNIST_LABEL  = -1

os.makedirs(OUT_DIR, exist_ok=True)

# ─── Poincaré ball operations (c=1) ─────────────────────────────────────────
def _mob_add(x, y):
    x2 = (x * x).sum(-1, keepdims=True)
    y2 = (y * y).sum(-1, keepdims=True)
    xy = (x * y).sum(-1, keepdims=True)
    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    den = np.clip(1 + 2 * xy + x2 * y2, 1e-8, None)
    return num / den

def _exp0(v):
    nv = np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-8)
    return np.tanh(nv) * v / nv

def _log0(p):
    np_ = np.linalg.norm(p, axis=-1, keepdims=True).clip(1e-8)
    return np.arctanh(np_.clip(0, 1 - 1e-5)) * p / np_

def _lam(x):
    x2 = (x * x).sum(-1, keepdims=True)
    return 2.0 / np.clip(1 - x2, 1e-8, None)

def _exp_x(x, v):
    lam = _lam(x)
    nv = np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-8)
    y = np.tanh(lam * nv / 2) * v / nv
    return _mob_add(x, y)

def _log_x(x, y):
    add = _mob_add(-x, y)
    n_add = np.linalg.norm(add, axis=-1, keepdims=True).clip(1e-8)
    lam = _lam(x)
    return (2 / lam) * np.arctanh(n_add.clip(0, 1 - 1e-5)) * add / n_add

def frechet_mean(pts, n_iter=60, lr=0.05, tol=1e-6):
    mu = pts.mean(0, keepdims=True)
    mn = np.linalg.norm(mu)
    if mn >= 1.0:
        mu = mu / mn * 0.9
    for _ in range(n_iter):
        grads = _log_x(mu, pts)
        step = lr * grads.mean(0, keepdims=True)
        mu_new = _exp_x(mu, step)
        mn = np.linalg.norm(mu_new)
        if mn >= 1.0:
            mu_new = mu_new / mn * (1 - 1e-5)
        if np.linalg.norm(mu_new - mu) < tol:
            return mu_new.squeeze(0)
        mu = mu_new
    return mu.squeeze(0)

def lorentz_to_poincare(h):
    """Stereographic projection: Lorentz L^d → Poincaré ball B^d.

    h: [N, d+1] on Lorentz hyperboloid (⟨h,h⟩_L = -1, h₀ > 0)
    returns p: [N, d] with ||p|| < 1
    """
    return h[:, 1:] / (1.0 + h[:, 0, np.newaxis])

def horopca(poincare_pts, n_components=2):
    print(f"  Fréchet mean over {len(poincare_pts)} pts …")
    mu = frechet_mean(poincare_pts)
    print(f"  μ radius = {np.linalg.norm(mu):.4f}")
    logs = _log_x(mu[np.newaxis], poincare_pts)
    pca = PCA(n_components=n_components, whiten=False)
    coords = pca.fit_transform(logs)
    print(f"  PC1={pca.explained_variance_ratio_[0]:.1%}  "
          f"PC2={pca.explained_variance_ratio_[1]:.1%}")
    return coords, mu, pca

def coords_to_disk(coords, scale):
    return _exp0(coords / scale)

# ─── Load model ─────────────────────────────────────────────────────────────
def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    embed_dim = args.get("embed_dim", 512)

    model = ResNetHyper(arch=args.get("arch", "resnet18"), pretrained=False,
                        embed_dim=embed_dim)
    hf = HyperbolicClassifier(embed_dim, n_classes=100)
    hc = HyperbolicClassifier(embed_dim, n_classes=20)
    model.set_hyp_heads(hf, hc)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    print(f"Loaded {args.get('arch','resnet18')} epoch {ckpt['epoch']}")
    return model, ckpt

# ─── Feature extraction ─────────────────────────────────────────────────────
val_tf = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@torch.no_grad()
def extract_cifar(model, device, n_samples):
    """Extract features from CIFAR-100 test set, stratified across coarse classes."""
    ds = tv_datasets.CIFAR100(root=DATA_ROOT, train=False, download=False)
    zf_all, zc_all, hf_all, hc_all = [], [], [], []
    fine_labels, coarse_labels = [], []

    # Load coarse labels
    import pickle
    raw_path = Path(DATA_ROOT) / "cifar-100-python" / "test"
    with open(raw_path, "rb") as f:
        raw = pickle.load(f, encoding="bytes")
    coarse_map = list(raw[b"coarse_labels"])

    # Subsample per coarse class
    samples_per_coarse = max(1, n_samples // 20)

    B = 256
    for coarse_c in range(20):
        # Find indices belonging to this coarse class
        idx = [i for i, cl in enumerate(coarse_map) if cl == coarse_c]
        idx = idx[:samples_per_coarse]
        if not idx:
            continue

        for start in range(0, len(idx), B):
            end = min(start + B, len(idx))
            batch_idx = idx[start:end]
            images = torch.stack([
                val_tf(Image.fromarray(ds.data[i])) for i in batch_idx
            ]).to(device)
            y_fine = torch.tensor([ds.targets[i] for i in batch_idx])
            y_coarse = torch.tensor([coarse_map[i] for i in batch_idx])

            out = model(images)
            zf_all.append(out["z_f"].cpu().float().numpy())
            zc_all.append(out["z_c"].cpu().float().numpy())
            hf_all.append(out["h_f"].cpu().float().numpy())
            hc_all.append(out["h_c"].cpu().float().numpy())
            fine_labels.append(y_fine.numpy())
            coarse_labels.append(y_coarse.numpy())

    return (np.concatenate(zf_all), np.concatenate(zc_all),
            np.concatenate(hf_all), np.concatenate(hc_all),
            np.concatenate(fine_labels), np.concatenate(coarse_labels))


@torch.no_grad()
def extract_mnist(model, device, n_samples):
    """Extract features from MNIST test set (OOD)."""
    mnist_dir = Path(OODD_ROOT) / "images_classic" / "mnist" / "test"
    files = sorted(mnist_dir.glob("*.jpg"))[:n_samples]

    zf_all, zc_all, hf_all, hc_all = [], [], [], []

    B = 256
    for start in range(0, len(files), B):
        end = min(start + B, len(files))
        batch_files = files[start:end]

        images = []
        for f in batch_files:
            img = Image.open(f).convert("RGB")
            images.append(val_tf(img))
        images = torch.stack(images).to(device)

        out = model(images)
        zf_all.append(out["z_f"].cpu().float().numpy())
        zc_all.append(out["z_c"].cpu().float().numpy())
        hf_all.append(out["h_f"].cpu().float().numpy())
        hc_all.append(out["h_c"].cpu().float().numpy())

    return (np.concatenate(zf_all), np.concatenate(zc_all),
            np.concatenate(hf_all), np.concatenate(hc_all))


# ─── Drawing ────────────────────────────────────────────────────────────────
def draw_disk(ax, title=""):
    ax.add_patch(plt.Circle((0, 0), 1.0, color="#cccccc",
                            fill=False, lw=1.5, ls="--", zorder=0))
    for r in [0.33, 0.66]:
        ax.add_patch(plt.Circle((0, 0), r, color="#e8e8e8",
                                fill=False, lw=0.5, zorder=0))
    ax.axhline(0, color="#e8e8e8", lw=0.3, zorder=0)
    ax.axvline(0, color="#e8e8e8", lw=0.3, zorder=0)
    ax.scatter([0], [0], c="black", s=25, marker="+", zorder=5)
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("HoroPCA ResNet-HYPER: CIFAR-100 vs MNIST OOD")
    print(f"  checkpoint: {CKPT_PATH}")
    print(f"  device:     {DEVICE}")
    print(f"  ID samples: {N_CIFAR}   OOD samples: {N_MNIST}")
    print("=" * 60)

    # ── Load model ──────────────────────────────────────────────────────────
    print("\nLoading model …")
    model, ckpt = load_model(CKPT_PATH, DEVICE)
    ckpt_epoch = ckpt['epoch']

    # ── Extract CIFAR-100 features ──────────────────────────────────────────
    print(f"\nExtracting CIFAR-100 test features ({N_CIFAR} samples) …")
    (zf_id, zc_id, hf_id, hc_id, fine_id, coarse_id) = extract_cifar(
        model, DEVICE, N_CIFAR
    )
    print(f"  z_f: {zf_id.shape}  z_c: {zc_id.shape}")
    print(f"  h_f: {hf_id.shape}  h_c: {hc_id.shape}")

    # ── Extract MNIST features ──────────────────────────────────────────────
    print(f"\nExtracting MNIST OOD features ({N_MNIST} samples) …")
    zf_ood, zc_ood, hf_ood, hc_ood = extract_mnist(model, DEVICE, N_MNIST)
    print(f"  z_f: {zf_ood.shape}  z_c: {zc_ood.shape}")

    # ── Lorentz → Poincaré (proper stereographic projection) ────────────────
    print("\nLorentz → Poincaré (stereographic projection) …")
    pf_id = lorentz_to_poincare(hf_id);    pc_id = lorentz_to_poincare(hc_id)
    pf_ood = lorentz_to_poincare(hf_ood);  pc_ood = lorentz_to_poincare(hc_ood)

    print(f"  ID   ‖h_f‖={np.linalg.norm(pf_id,axis=-1).mean():.3f}  "
          f"‖h_c‖={np.linalg.norm(pc_id,axis=-1).mean():.3f}")
    print(f"  OOD  ‖h_f‖={np.linalg.norm(pf_ood,axis=-1).mean():.3f}  "
          f"‖h_c‖={np.linalg.norm(pc_ood,axis=-1).mean():.3f}")

    # ── HoroPCA: joint basis (h_f + h_c, Lorentz→Poincaré) ─────────────────
    all_pts = np.concatenate([pf_id, pc_id, pf_ood, pc_ood], axis=0)
    print(f"\nHoroPCA on {len(all_pts)} points (h_f + h_c, Lorentz→Poincaré) …")
    coords_all, mu, pca_obj = horopca(all_pts, n_components=2)

    n = len(pf_id)
    hf_id_c  = coords_all[:n]           # CIFAR-100 h_f
    hc_id_c  = coords_all[n:2*n]        # CIFAR-100 h_c
    hf_ood_c = coords_all[2*n:2*n+N_MNIST]        # MNIST h_f
    hc_ood_c = coords_all[2*n+N_MNIST:2*n+2*N_MNIST]  # MNIST h_c

    # Scale to Poincaré disk
    all_t = np.concatenate([hf_id_c, hc_id_c, hf_ood_c, hc_ood_c])
    disk_scale = np.percentile(np.linalg.norm(all_t, axis=-1), 95) * 1.15

    def to_disk(t):
        return coords_to_disk(t, disk_scale)

    df_id  = to_disk(hf_id_c);   dc_id  = to_disk(hc_id_c)
    df_ood = to_disk(hf_ood_c);  dc_ood = to_disk(hc_ood_c)

    # ── Single HoroPCA Disk: h_f (•) + h_c (★) ────────────────────────────
    ev = pca_obj.explained_variance_ratio_
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    fig.suptitle(
        f"HoroPCA — ResNet-HYPER (ResNet18, epoch {ckpt_epoch})\n"
        f"$h_f$ (•) + $h_c$ (★)  on shared Poincaré disk  |  "
        f"CIFAR-100 ID vs MNIST OOD\n"
        f"Lorentz→Poincaré via stereographic projection  |  "
        f"PC1={ev[0]:.1%}  PC2={ev[1]:.1%}",
        fontsize=12, fontweight="bold", y=0.97,
    )
    draw_disk(ax, "")

    # ── CIFAR-100 h_f (• small circles) ────────────────────────────────────
    for c in range(20):
        m = coarse_id == c
        if m.sum() == 0:
            continue
        ax.scatter(df_id[m, 0], df_id[m, 1],
                   c=COARSE_COLORS[c], s=25, marker="o",
                   alpha=0.75, edgecolors="white", linewidths=0.3, zorder=3,
                   label=f"{COARSE_LABELS[c]} $h_f$" if m.sum() > 0 else "")

    # ── MNIST h_f (× crosses) ──────────────────────────────────────────────
    ax.scatter(df_ood[:, 0], df_ood[:, 1],
               c=MNIST_COLOR, s=30, marker="x",
               alpha=0.7, linewidths=1.0, zorder=4, label="MNIST $h_f$")

    # ── CIFAR-100 h_c (★ stars, larger) ────────────────────────────────────
    for c in range(20):
        m = coarse_id == c
        if m.sum() == 0:
            continue
        ax.scatter(dc_id[m, 0], dc_id[m, 1],
                   c=COARSE_COLORS[c], s=100, marker="*",
                   alpha=0.90, edgecolors="white", linewidths=0.5, zorder=5)

    # ── MNIST h_c (★ stars) ────────────────────────────────────────────────
    ax.scatter(dc_ood[:, 0], dc_ood[:, 1],
               c=MNIST_COLOR, s=70, marker="*",
               alpha=0.7, edgecolors="none", zorder=5, label="MNIST $h_c$")

    # ── Fine→coarse connecting lines (thin, 10% of ID samples) ────────────
    n_conn = min(100, n)
    step = max(1, n // n_conn)
    for i in range(0, n, step):
        ax.plot([df_id[i, 0], dc_id[i, 0]],
                [df_id[i, 1], dc_id[i, 1]],
                c=COARSE_COLORS[coarse_id[i]], lw=0.5, alpha=0.15, zorder=1)

    # ── Radius reference rings ──────────────────────────────────────────────
    r_f = np.linalg.norm(df_id, axis=-1).mean()
    r_c = np.linalg.norm(dc_id, axis=-1).mean()
    r_ood = np.linalg.norm(df_ood, axis=-1).mean()
    for r, ls, lbl in [(r_f, "--", f"mean $h_f$ ID  r={r_f:.2f}"),
                        (r_c, "-",  f"mean $h_c$ ID  r={r_c:.2f}"),
                        (r_ood, ":", f"mean $h_f$ OOD r={r_ood:.2f}")]:
        ax.add_patch(plt.Circle((0, 0), r, color="#777777",
                                fill=False, lw=1.2, ls=ls, alpha=0.4, zorder=1))
        ax.text(0, r + 0.02, lbl, ha="center", va="bottom",
                fontsize=7, color="#555555")

    # ── Legend ──────────────────────────────────────────────────────────────
    coarse_handles = [mpatches.Patch(color=COARSE_COLORS[i], label=COARSE_LABELS[i])
                      for i in range(20) if (coarse_id == i).sum() > 0]
    ood_handle = mpatches.Patch(color=MNIST_COLOR, label="MNIST (OOD)")
    all_handles = coarse_handles + [ood_handle]
    from matplotlib.lines import Line2D
    marker_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#555555',
               markersize=8, label='$h_f$ (fine)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#555555',
               markersize=12, label='$h_c$ (coarse)'),
        Line2D([0], [0], marker='x', color='#555555', markerfacecolor='#555555',
               markersize=8, label='OOD (×)'),
    ]
    fig.legend(handles=all_handles + marker_handles,
               loc="lower center", ncol=7, fontsize=6.5, frameon=True,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.07, 1, 0.94])
    out_path = OUT_DIR / "horopca_resnet_lorentz.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_path}")

    # ── Fine-grained (100-class) HoroPCA ────────────────────────────────────
    # Generate 100-color palette from HSV
    import matplotlib.colors as mcolors
    fine_colors = [mcolors.hsv_to_rgb((i / 100, 0.75, 0.9)) for i in range(100)]

    fig2, ax2 = plt.subplots(1, 1, figsize=(14, 14))
    fig2.suptitle(
        f"HoroPCA — ResNet-HYPER (ResNet18, epoch {ckpt_epoch})  |  "
        f"$h_f$ only — 100 Fine-Grained Classes\n"
        f"Lorentz→Poincaré stereographic projection  |  "
        f"CIFAR-100 ID vs MNIST OOD  |  "
        f"PC1={ev[0]:.1%}  PC2={ev[1]:.1%}",
        fontsize=11, fontweight="bold", y=0.97,
    )
    draw_disk(ax2, "")

    # CIFAR-100: each fine class gets a unique color
    for f in range(100):
        m = fine_id == f
        if m.sum() == 0:
            continue
        c = coarse_id[m][0]  # coarse class determines base hue
        ax2.scatter(df_id[m, 0], df_id[m, 1],
                    c=[fine_colors[f]], s=20, marker="o",
                    alpha=0.75, edgecolors="white", linewidths=0.2, zorder=3,
                    label=FINE_LABELS[f] if m.sum() >= 3 else "")

    # MNIST OOD
    ax2.scatter(df_ood[:, 0], df_ood[:, 1],
                c=MNIST_COLOR, s=28, marker="x",
                alpha=0.7, linewidths=1.0, zorder=4, label="MNIST (OOD)")

    # Highlight a few example classes with annotations
    highlight_classes = [0, 5, 10, 25, 42, 60, 80, 95]  # diverse fine classes
    for f in highlight_classes:
        m = fine_id == f
        if m.sum() == 0:
            continue
        pts = df_id[m]
        centroid = pts.mean(axis=0)
        ax2.annotate(FINE_LABELS[f], (centroid[0], centroid[1]),
                     fontsize=6, fontweight="bold",
                     color=fine_colors[f], alpha=0.9,
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=fine_colors[f], alpha=0.7))

    # Reference rings
    r_f_mean = np.linalg.norm(df_id, axis=-1).mean()
    r_ood_mean = np.linalg.norm(df_ood, axis=-1).mean()
    for r, ls, lbl in [(r_f_mean, "-", f"mean ID r={r_f_mean:.2f}"),
                        (r_ood_mean, ":", f"mean OOD r={r_ood_mean:.2f}")]:
        ax2.add_patch(plt.Circle((0, 0), r, color="#666666",
                                 fill=False, lw=1.0, ls=ls, alpha=0.35, zorder=1))
        ax2.text(0, r + 0.015, lbl, ha="center", va="bottom",
                 fontsize=7, color="#555555")

    # Compact legend: show only highlighted + coarse grouping labels
    legend_patches = [mpatches.Patch(color=COARSE_COLORS[c], label=COARSE_LABELS[c])
                      for c in range(20)]
    legend_patches.append(mpatches.Patch(color=MNIST_COLOR, label="MNIST (OOD)"))
    fig2.legend(handles=legend_patches, loc="lower center", ncol=5,
                fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    out_path2 = OUT_DIR / "horopca_resnet_100class.png"
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path2}")
    plt.close(fig2)

    # ── OOD separation stats ────────────────────────────────────────────────
    print("\n── OOD Separation (Lorentz→Poincaré radius before PCA) ──")
    for label, pf, pc in [("ID", pf_id, pc_id), ("OOD", pf_ood, pc_ood)]:
        rf = np.linalg.norm(pf, axis=-1)  # Poincaré norm of h_f
        rc = np.linalg.norm(pc, axis=-1)  # Poincaré norm of h_c
        print(f"  {label:4s}  ‖h_f‖={rf.mean():.3f}±{rf.std():.3f}  "
              f"‖h_c‖={rc.mean():.3f}±{rc.std():.3f}  gap={rc.mean()-rf.mean():+.3f}")

    # Per-coarse fine→coarse gap
    print("\n── Per-coarse h_f→h_c Poincaré gap ──")
    for c in range(20):
        m = coarse_id == c
        if m.sum() == 0:
            continue
        rf = np.linalg.norm(pf_id[m], axis=-1).mean()
        rc = np.linalg.norm(pc_id[m], axis=-1).mean()
        print(f"  {COARSE_LABELS[c]:35s}  gap={rc-rf:+.4f}  "
              f"(‖h_f‖={rf:.3f}  ‖h_c‖={rc:.3f})")


if __name__ == "__main__":
    main()
