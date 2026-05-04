"""HoroPCA visualization showing L_pop cone containment.

Projects h_f (fine) and h_c (coarse) together on Poincaré disk.
Draws fine→coarse connecting lines and computes containment stats.

Usage: python OODD/visualize_pop_cone.py
"""

import sys, os
from pathlib import Path
_THIS = Path(__file__).resolve(); _SL = _THIS.parent.parent
sys.path.insert(0, str(_SL))

import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from PIL import Image
from torchvision import transforms as T, datasets as tv_datasets
from sklearn.decomposition import PCA
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from HDHM.models.resnet_hyper import ResNetHyper
from HDHM.losses.hyperbolic_loss import (
    HyperbolicClassifier, lorentz_to_euclidean,
    exterior_angle as compute_cone_angle,
    half_aperture as compute_half_aperture,
)

CKPT = "training/runs/resnet_hyper_r18_v3/best.pth"
DATA_ROOT = "sign_language/deep-hybrid-models/data"
OUT_DIR = Path("OODD/results"); os.makedirs(OUT_DIR, exist_ok=True)
N_SAMPLES = 800  # stratified across 20 coarse classes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pick 4 diverse coarse classes to show cones for
SHOW_CONES = [4, 8, 13, 17]  # fruits_veg, large_carnivores, non-insect_inv, vehicles_1
COARSE_COLORS = plt.cm.tab20(np.linspace(0, 1, 20))
from cifar100_hierarchy import COARSE_LABELS

# ─── Poincaré geometry ─────────────────────────────────────────────────────
def _mob_add(x, y):
    x2 = (x*x).sum(-1,keepdims=True); y2 = (y*y).sum(-1,keepdims=True)
    xy = (x*y).sum(-1,keepdims=True)
    return ((1+2*xy+y2)*x + (1-x2)*y) / np.clip(1+2*xy+x2*y2, 1e-8, None)

def _exp0(v):
    nv = np.linalg.norm(v,axis=-1,keepdims=True).clip(1e-8)
    return np.tanh(nv) * v/nv

def _lam(x):
    return 2.0 / np.clip(1-(x*x).sum(-1,keepdims=True), 1e-8, None)

def _exp_x(x, v):
    lam = _lam(x); nv = np.linalg.norm(v,axis=-1,keepdims=True).clip(1e-8)
    return _mob_add(x, np.tanh(lam*nv/2)*v/nv)

def _log_x(x, y):
    add = _mob_add(-x, y); n_add = np.linalg.norm(add,axis=-1,keepdims=True).clip(1e-8)
    return (2/_lam(x))*np.arctanh(n_add.clip(0,1-1e-5))*add/n_add

def frechet_mean(pts, n_iter=60, lr=0.05, tol=1e-6):
    mu = pts.mean(0,keepdims=True)
    if np.linalg.norm(mu) >= 1.0: mu = mu/np.linalg.norm(mu)*0.9
    for _ in range(n_iter):
        step = lr*_log_x(mu, pts).mean(0,keepdims=True)
        mu_new = _exp_x(mu, step)
        if np.linalg.norm(mu_new) >= 1.0: mu_new = mu_new/np.linalg.norm(mu_new)*(1-1e-5)
        if np.linalg.norm(mu_new-mu) < tol: return mu_new.squeeze(0)
        mu = mu_new
    return mu.squeeze(0)

def lorentz_to_poincare(h):
    """Stereographic: Lorentz L^d → Poincaré B^d."""
    return h[:, 1:] / (1.0 + h[:, 0, np.newaxis])

def draw_disk(ax):
    for r in [0.33, 0.66]:
        ax.add_patch(plt.Circle((0,0), r, color="#e8e8e8", fill=False, lw=0.4, zorder=0))
    ax.axhline(0, color="#e8e8e8", lw=0.3, zorder=0)
    ax.axvline(0, color="#e8e8e8", lw=0.3, zorder=0)
    ax.add_patch(plt.Circle((0,0), 1.0, color="#cccccc", fill=False, lw=1.2, ls="--", zorder=0))
    ax.scatter([0],[0], c="black", s=20, marker="+", zorder=5)
    ax.set_xlim(-1.15,1.15); ax.set_ylim(-1.15,1.15); ax.set_aspect("equal"); ax.axis("off")

# ─── Model ──────────────────────────────────────────────────────────────────
print(f"Loading model from {CKPT} ...")
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
args = ckpt.get("args", {})
ed = args.get("embed_dim", 512)
model = ResNetHyper(arch=args.get("arch","resnet18"), pretrained=False, embed_dim=ed)
model.load_state_dict({k:v for k,v in ckpt["model_state_dict"].items()
                       if not k.startswith("hyp_head")}, strict=False)
model = model.to(DEVICE).eval()
print(f"  epoch {ckpt['epoch']}")

val_tf = T.Compose([
    T.Resize(224, T.InterpolationMode.BICUBIC), T.CenterCrop(224),
    T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ─── Extract features ──────────────────────────────────────────────────────
print(f"Extracting {N_SAMPLES} CIFAR-100 test samples ...")
ds = tv_datasets.CIFAR100(root=DATA_ROOT, train=False, download=False)
import pickle
with open(Path(DATA_ROOT)/"cifar-100-python"/"test","rb") as f:
    coarse_labels = list(pickle.load(f,encoding="bytes")[b"coarse_labels"])

# Stratified sampling
idx = []
samples_per = max(1, N_SAMPLES//20)
for c in range(20):
    ci = [i for i,cl in enumerate(coarse_labels) if cl==c][:samples_per]
    idx.extend(ci)
idx = idx[:N_SAMPLES]

hf_all, hc_all = [], []
coarse_all, fine_all = [], []
B = 256
for start in tqdm(range(0, len(idx), B), desc="Extracting"):
    end = min(start+B, len(idx))
    bi = idx[start:end]
    imgs = torch.stack([val_tf(Image.fromarray(ds.data[i])) for i in bi]).to(DEVICE)
    with torch.no_grad():
        out = model(imgs)
    hf_all.append(out["h_f"].cpu().numpy())
    hc_all.append(out["h_c"].cpu().numpy())
    coarse_all.extend([coarse_labels[i] for i in bi])
    fine_all.extend([ds.targets[i] for i in bi])

hf = np.concatenate(hf_all); hc = np.concatenate(hc_all)
coarse_id = np.array(coarse_all); fine_id = np.array(fine_all)
n = len(hf)

# ─── Lorentz → Poincaré ────────────────────────────────────────────────────
pf = lorentz_to_poincare(hf)   # fine
pc = lorentz_to_poincare(hc)   # coarse

print(f"  ‖h_f‖_poin={np.linalg.norm(pf,axis=-1).mean():.3f}  "
      f"‖h_c‖_poin={np.linalg.norm(pc,axis=-1).mean():.3f}")

# ─── Cone containment stats (before HoroPCA) ──────────────────────────────
print("\n── L_pop Cone Containment Check ──")
hf_t = torch.from_numpy(hf); hc_t = torch.from_numpy(hc)
for c in range(20):
    m = coarse_id == c
    if m.sum() < 3: continue
    ang = compute_cone_angle(hc_t[m], hf_t[m])
    ap = compute_half_aperture(hc_t[m], margin=0.1)
    contained = (ang <= ap).float().mean().item()*100
    violated = (ang > ap).float().mean().item()*100
    print(f"  {COARSE_LABELS[c]:35s}  contained={contained:5.1f}%  violated={violated:5.1f}%  ‖h_c‖={np.linalg.norm(pc[m],axis=-1).mean():.3f}")

# ─── HoroPCA ────────────────────────────────────────────────────────────────
all_pts = np.concatenate([pf, pc], axis=0)
print(f"\nHoroPCA on {len(all_pts)} points ...")
mu = frechet_mean(all_pts)
print(f"  μ radius = {np.linalg.norm(mu):.4f}")
logs = _log_x(mu[np.newaxis], all_pts)
pca = PCA(n_components=2); coords = pca.fit_transform(logs)
print(f"  PC1={pca.explained_variance_ratio_[0]:.1%}  PC2={pca.explained_variance_ratio_[1]:.1%}")

cf = coords[:n]; cc = coords[n:]
all_t = np.concatenate([cf, cc])
disk_s = np.percentile(np.linalg.norm(all_t,axis=-1), 95)*1.2
df = _exp0(cf/disk_s); dc = _exp0(cc/disk_s)

# ─── Plot ───────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ev = pca.explained_variance_ratio_

# ── Panel 1: All coarse classes colored ────────────────────────────────────
fig.suptitle(f"L_pop Cone Containment — ResNet-HYPER v3 (epoch {ckpt['epoch']})  |  "
             f"$h_f$(•) → $h_c$(★)  |  PC1={ev[0]:.1%}  PC2={ev[1]:.1%}",
             fontsize=13, fontweight="bold", y=0.98)
draw_disk(ax1)

for c in range(20):
    m = coarse_id == c
    if m.sum() == 0: continue
    # Fine (small circles)
    ax1.scatter(df[m,0], df[m,1], c=[COARSE_COLORS[c]], s=15, marker="o",
                alpha=0.6, edgecolors="none", zorder=3)
    # Coarse (large stars)
    ax1.scatter(dc[m,0], dc[m,1], c=[COARSE_COLORS[c]], s=80, marker="*",
                alpha=0.9, edgecolors="white", linewidths=0.4, zorder=5)

# Draw fine→coarse connecting lines for a few samples
for c in SHOW_CONES:
    m = np.where(coarse_id == c)[0][:8]
    for i in m:
        ax1.plot([df[i,0], dc[i,0]], [df[i,1], dc[i,1]],
                 c=COARSE_COLORS[c], lw=0.6, alpha=0.2, zorder=1)

ax1.set_title("All 20 coarse classes", fontsize=11)

# ── Panel 2: Zoom on 4 classes with containment status ─────────────────────
draw_disk(ax2)
ax2.set_title("Zoom: 4 classes + cone containment check", fontsize=11)

for c_idx, c in enumerate(SHOW_CONES):
    m = coarse_id == c
    if m.sum() == 0: continue

    # Compute containment per sample
    hc_c = hc_t[m]; hf_c = hf_t[m]
    ang = compute_cone_angle(hc_c, hf_c)
    ap = compute_half_aperture(hc_c, margin=0.1)
    contained_mask = (ang <= ap).numpy()

    # Fine: green edge = contained, red edge = violated
    for j, ii in enumerate(np.where(m)[0]):
        ec = "#2ca02c" if contained_mask[j] else "#d62728"
        ax2.scatter([df[ii,0]], [df[ii,1]], c=[COARSE_COLORS[c]], s=22, marker="o",
                    alpha=0.7, edgecolors=ec, linewidths=1.0, zorder=3)

    # Coarse
    ax2.scatter(dc[m,0], dc[m,1], c=[COARSE_COLORS[c]], s=100, marker="*",
                alpha=0.95, edgecolors="white", linewidths=0.5, zorder=5,
                label=f"{COARSE_LABELS[c]} ({contained_mask.mean()*100:.0f}% in cone)")

    # Connecting lines
    for j, ii in enumerate(np.where(m)[0][:8]):
        ec = "#2ca02c" if contained_mask[j] else "#d62728"
        ax2.plot([df[ii,0], dc[ii,0]], [df[ii,1], dc[ii,1]],
                 c=ec, lw=0.8, alpha=0.3, zorder=1)

ax2.legend(loc="lower center", fontsize=8, ncol=2, frameon=True,
           bbox_to_anchor=(0.5, -0.12))

# ── Legend markers ──────────────────────────────────────────────────────────
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
marker_h = [
    Line2D([0],[0], marker='*', color='w', markerfacecolor='#555', markersize=14, label='$h_c$ (coarse)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#555', markersize=8, label='$h_f$ (fine)'),
    Line2D([0],[0], color='#2ca02c', lw=1.5, label='contained in cone ✅'),
    Line2D([0],[0], color='#d62728', lw=1.5, label='violated cone ❌'),
]
fig.legend(handles=marker_h, loc="lower center", ncol=4, fontsize=9,
           frameon=True, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0,0.05,1,0.94])
out = OUT_DIR / "pop_cone_containment.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out}")

# ─── Global containment summary ─────────────────────────────────────────────
ang_all = compute_cone_angle(hc_t, hf_t).numpy()
ap_all = compute_half_aperture(hc_t, margin=0.1).numpy()
contained = (ang_all <= ap_all).mean() * 100
print(f"\n═══ Global Cone Containment: {contained:.1f}% ═══")
print(f"  angle:     {ang_all.mean():.4f} ± {ang_all.std():.4f}")
print(f"  aperture:  {ap_all.mean():.4f} ± {ap_all.std():.4f}")
print(f"  violation: {100-contained:.1f}%")
