import numpy as np
import os 
from .results_helper import setup_publication_style
ROOT = "cluster_examples"
# if __name__ == "__main__":
#     file = os.path.join(ROOT, "cluster_results.npz")
#     data = np.load(file)
#     print("Keys in file:", data.files)

#     clusters = data["clusters"]
#     print("Unique clusters:", np.unique(clusters), "Total:", clusters.shape)
#     # for k in data.files:
#     #     arr = data[k]
#     #     print(f"\n=== {k} ===")
#     #     print("shape:", arr.shape, "dtype:", arr.dtype)
#     #     print(arr)    # careful if large, can be huge

CLASS_INFO = {
    "cifar10": {
        "index": [5, 3, 8],
        "names": ["cat", "dog", "ship"]
    }
}


import os, math, numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ---------- helper: make clusters 1D ----------
# --- helpers you already had ---

def ensure_1d_clusters(clusters):
    clusters = np.asarray(clusters)
    if clusters.ndim == 2: clusters = clusters[0]
    return clusters.astype(np.int64)

def simplex3_to_xy(P):
    P = np.asarray(P, float)
    P = P / np.clip(P.sum(1, keepdims=True), 1e-12, None)
    x = P[:,1] + 0.5*P[:,2]
    y = (math.sqrt(3)/2.0)*P[:,2]
    return x, y

def make_cmap(K):
    base = plt.get_cmap("tab20").colors
    if K <= 20: colors = base[:K]
    else:       colors = [plt.cm.hsv(i / K) for i in range(K)]
    return ListedColormap(colors)

# --- vertical palette strip (discrete, no ticks) ---
def add_palette_strip_vertical(ax, cmap, K, label="clusters"):
    iax = inset_axes(ax, width="3%", height="50%",  # narrow, tall
                     loc="center right", bbox_to_anchor=(0,0,1,1),
                     bbox_transform=ax.transAxes, borderpad=1.2)
    strip = np.arange(K)[:, None]   # (K,1)
    iax.imshow(strip, aspect="auto", cmap=cmap, interpolation="nearest",
               extent=[0, 1, 0, K], origin="lower")
    iax.set_xticks([]); iax.set_yticks([])
    for sp in iax.spines.values(): sp.set_visible(False)
    # label vertically
    iax.text(0.5, 1.04, label, rotation=0, 
             ha="center", va="center",
             transform=iax.transAxes,
            #  clip_on=False  
             )

# def plot_simplex_clusters(
#     embs, clusters, class_labels=("class 0","class 1","class 2"),
#     out_path="simplex_clusters.png", s=6, alpha=0.6,
#     bar_mode="strip"  # "strip" | "colorbar"
# ):
#     C = ensure_1d_clusters(clusters)
#     E = np.asarray(embs, float)
#     x, y = simplex3_to_xy(E)

#     v0, v1, v2 = (0.0, 0.0), (1.0, 0.0), (0.5, math.sqrt(3)/2.0)
#     K = int(C.max()) + 1
#     cmap = make_cmap(K)

#     fig, ax = plt.subplots(figsize=(7, 6), dpi=150)

#     # triangle
#     ax.plot([v0[0], v1[0], v2[0], v0[0]], [v0[1], v1[1], v2[1], v0[1]], lw=1.5, color="black")
#     for t in np.linspace(0.1, 0.9, 5):
#         ax.plot([0.5*t, 1-0.5*t], [t*math.sqrt(3)/2]*2, lw=0.5, color="#ccc", alpha=0.6)

#     # scatter: single call, color by cluster id
#     sc = ax.scatter(x, y, c=C, cmap=cmap, s=s, alpha=alpha, edgecolors="none",
#                     vmin=0, vmax=K-1)

#     # vertical bar labeled "clusters"
#     if bar_mode == "strip":
#         add_palette_strip_vertical(ax, cmap, K, label="clusters")
#     else:
#         sm = ScalarMappable(cmap=cmap); sm.set_array([])
#         cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.02, fraction=0.05)
#         cbar.set_ticks([])               # hide ticks
#         cbar.set_label("clusters", rotation=0)

def make_error_aware_cluster_colors(K, cmap, cluster_error_means,
                                    red_color=(0.85, 0.12, 0.12),
                                    blend_strength=1.0,
                                    vmin_mu=None, vmax_mu=None):
    """
    Returns colors_k: shape (K,4), one color per cluster id (0..K-1),
    blending each cluster's base color toward `red_color` according to its mean error.
    """
    if cmap is None:
        # assumes you already have make_cmap(K)
        cmap = make_cmap(K)

    base_palette = np.asarray([cmap(i) for i in range(K)])  # (K,4)
    mu = np.asarray(cluster_error_means, float)

    mu_min = float(np.min(mu)) if vmin_mu is None else float(vmin_mu)
    mu_max = float(np.max(mu)) if vmax_mu is None else float(vmax_mu)
    denom = (mu_max - mu_min) if mu_max > mu_min else 1.0
    w = np.clip(((mu - mu_min) / denom) * blend_strength, 0.0, 1.0)  # (K,)

    colors_k = np.array([_lerp_color(base_palette[i], red_color, w[i]) for i in range(K)])
    return colors_k

def plot_simplex_clusters(
    embs, clusters, class_labels=("class 0","class 1","class 2"),
    out_path="simplex_clusters.png", s=6, alpha=0.6,
    bar_mode="strip",               # "strip" | "colorbar" | "none"
    cmap=None,
    triangle_scale=0.82,            # <── shrink the simplex toward its centroid
    label_fs=16,                    # <── bigger labels
    label_pad=0.04, 
    colors=None                                      # extra offset for labels (in simplex units)
):
    C = ensure_1d_clusters(clusters)
    E = np.asarray(embs, float)
    x_raw, y_raw = simplex3_to_xy(E)

    # original vertices (equilateral triangle)
    v0, v1, v2 = (0.0, 0.0), (1.0, 0.0), (0.5, math.sqrt(3)/2.0)
    # centroid of the simplex triangle
    cx, cy = 0.5, (math.sqrt(3)/6.0)

    def shrink_xy(x, y, scale, cx=cx, cy=cy):
        """Affinely shrink (x,y) toward centroid (cx,cy) by 'scale'."""
        return cx + scale*(x - cx), cy + scale*(y - cy)

    # shrink vertices and data
    v0s = shrink_xy(*v0, triangle_scale)
    v1s = shrink_xy(*v1, triangle_scale)
    v2s = shrink_xy(*v2, triangle_scale)
    x, y = shrink_xy(x_raw, y_raw, triangle_scale)

    K = int(C.max()) + 1
    if colors is None:
        cmap = make_cmap(K)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)

    # triangle border
    ax.plot([v0s[0], v1s[0], v2s[0], v0s[0]],
            [v0s[1], v1s[1], v2s[1], v0s[1]],
            lw=1.5, color="black")

    # inner grid lines (also shrunk)
    for t in np.linspace(0.1, 0.9, 5):
        # unscaled endpoints at height y = t*sqrt(3)/2, then shrink
        x1, y1 = 0.5*t, t*math.sqrt(3)/2
        x2, y2 = 1 - 0.5*t, t*math.sqrt(3)/2
        X1, Y1 = shrink_xy(x1, y1, triangle_scale)
        X2, Y2 = shrink_xy(x2, y2, triangle_scale)
        ax.plot([X1, X2], [Y1, Y2], lw=0.5, color="#ccc", alpha=0.6)

    # scatter
    # sc = ax.scatter(x, y, c=C, cmap=cmap, s=s, alpha=alpha, edgecolors="none",
    #                 vmin=0, vmax=K-1)
    sc = ax.scatter(x, y, c=colors_k[C], s=s, alpha=alpha, edgecolors="none")


    # legend handling
    if bar_mode == "strip":
        add_palette_strip_vertical(ax, cmap, K, label="clusters")
    elif bar_mode == "colorbar":
        sm = ScalarMappable(cmap=cmap); sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.02, fraction=0.05)
        cbar.set_ticks([]); cbar.set_label("clusters", rotation=0)
    # elif "none": no legend in-figure

    # corner labels (push slightly outward from the shrunken vertices)
    def push_out(v, pad):
        # move from centroid a bit farther to keep labels outside the triangle
        return (v[0] + (v[0]-cx)*pad/triangle_scale,
                v[1] + (v[1]-cy)*pad/triangle_scale)

    lv0 = push_out(v0s, label_pad)
    lv1 = push_out(v1s, label_pad)
    lv2 = push_out(v2s, label_pad)

    ax.text(lv0[0], lv0[1], class_labels[0], ha="right", va="top",
            fontsize=label_fs, fontweight="medium", clip_on=False)
    ax.text(lv1[0], lv1[1], class_labels[1], ha="left",  va="top",
            fontsize=label_fs, fontweight="medium", clip_on=False)
    ax.text(lv2[0], lv2[1], class_labels[2], ha="center", va="bottom",
            fontsize=label_fs, fontweight="medium", clip_on=False)

    # cosmetics
    ax.set_aspect("equal")
    # keep generous margins so the larger labels don't get clipped
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, v2[1]+0.05)
    ax.set_xticks([]); ax.set_yticks([])

    # remove spines
    for side in ("left", "bottom", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.tick_params(left=False, bottom=False)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

# def plot_simplex_clusters(
#     embs, clusters, class_labels=("class 0","class 1","class 2"),
#     out_path="simplex_clusters.png", s=6, alpha=0.6,
#     bar_mode="strip",  # "strip" | "colorbar"
#     cmap=None          # <── add
# ):
#     C = ensure_1d_clusters(clusters)
#     E = np.asarray(embs, float)
#     x, y = simplex3_to_xy(E)

#     K = int(C.max()) + 1
#     if cmap is None:
#         cmap = make_cmap(K)

#     fig, ax = plt.subplots(figsize=(7, 6), dpi=150)


#     v0, v1, v2 = (0.0, 0.0), (1.0, 0.0), (0.5, math.sqrt(3)/2.0)
#         # triangle
#     ax.plot([v0[0], v1[0], v2[0], v0[0]], [v0[1], v1[1], v2[1], v0[1]], lw=1.5, color="black")
#     for t in np.linspace(0.1, 0.9, 5):
#         ax.plot([0.5*t, 1-0.5*t], [t*math.sqrt(3)/2]*2, lw=0.5, color="#ccc", alpha=0.6)


#     # ... triangle grid as before ...

#     sc = ax.scatter(x, y, c=C, cmap=cmap, s=s, alpha=alpha, edgecolors="none",
#                     vmin=0, vmax=K-1)

#     if bar_mode == "strip":
#         add_palette_strip_vertical(ax, cmap, K, label="clusters")
#     elif bar_mode == "colorbar":
#         sm = ScalarMappable(cmap=cmap); sm.set_array([])
#         cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.02, fraction=0.05)
#         cbar.set_ticks([])
#         cbar.set_label("clusters", rotation=0)

#     # corner labels
#     ax.text(v0[0]-0.03, v0[1]-0.03, class_labels[0], ha="right", va="top")
#     ax.text(v1[0]+0.03, v1[1]-0.03, class_labels[1], ha="left",  va="top")
#     ax.text(v2[0],      v2[1]+0.03, class_labels[2], ha="center", va="bottom")


#     ax.set_aspect("equal")
#     ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, v2[1]+0.05)
#     ax.set_xticks([]); ax.set_yticks([])

#     # hide the frame/spines so no bars at bottom/left corners
#     for side in ("left", "bottom", "right", "top"):
#         ax.spines[side].set_visible(False)
#     ax.tick_params(left=False, bottom=False)

#     plt.tight_layout()
#     plt.savefig(out_path, bbox_inches="tight")
#     plt.close(fig)
#     print(f"Saved: {out_path}")



import numpy as np
import matplotlib.pyplot as plt

# def plot_cluster_intervals(
#         cluster_error_means,
#         cluster_error_vars,
#         cluster_intervals,
#         save_path=None,
#         cmap=None):                      # <── add

#     mu  = np.asarray(cluster_error_means, float)
#     sd  = np.sqrt(np.asarray(cluster_error_vars, float))
#     cis = np.asarray(cluster_intervals, float)
#     K = mu.size
#     if cis.shape != (K, 2):
#         raise ValueError(f"cluster_intervals must be (K,2), got {cis.shape}")

#     # shared palette
#     if cmap is None:
#         cmap = make_cmap(K)
#     palette = np.asarray([cmap(i) for i in range(K)])   # indexable by cluster ID

#     # sort by decreasing mean, but keep original cluster IDs
#     order = np.argsort(mu)[::-1]
#     mu_sorted  = mu[order]
#     sd_sorted  = sd[order]
#     cis_sorted = cis[order]
#     orig_ids_sorted = order           # original cluster IDs in plotted order

#     x = np.arange(K)                  # ranks (enforces visual sorting)

#     plt.figure(figsize=(11, 4.5))

#     # draw CI + mean, using the color of the ORIGINAL cluster id
#     for xi, (lo, hi), m, cid in zip(x, cis_sorted, mu_sorted, orig_ids_sorted):
#         col = palette[cid]
#         plt.vlines(xi, lo, hi, linewidth=2, color=col)
#         plt.plot([xi], [m], 'o', color=col)
#         # optional: add std bar in same color
#         # plt.errorbar(xi, m, yerr=sd_sorted[xi], fmt='o', capsize=3, linewidth=1.5, color=col)

#     # plt.xlabel(r"Level sets $\mathcal{X}_z$, with $z \in \mathcal{Z}$")
#     plt.ylabel(r"$\eta_{r}(z)$")
#     plt.ylim(0, 1)
#     plt.grid(True, axis='y', linestyle='--', alpha=0.5)
#     plt.xticks([])

#     plt.tight_layout()
#     if save_path is not None:
#         plt.savefig(save_path, dpi=200, bbox_inches='tight')
#     plt.show()


def _lerp_color(c_base, c_target, w):
    # c_* are RGBA or RGB in [0,1]; return same length as base
    L = len(c_base)
    c_base = np.array(c_base[:3])           # ignore alpha for blend
    c_target = np.array(c_target[:3])
    out_rgb = (1.0 - w) * c_base + w * c_target
    if L == 4:
        return (*out_rgb, 1.0)
    return tuple(out_rgb)

def plot_cluster_intervals(
        cluster_error_means,
        cluster_error_vars,
        cluster_intervals,
        save_path=None,
        cmap=None,                  # <-- pass the SAME cmap used in the simplex plot
        *,
        red_color=(0.85, 0.12, 0.12),
        blend_strength=1.0,         # 0 = keep base colors; 1 = fully use red mapping
        figsize=(6, 2.3),
        gap=0.55,
        vline_width=3.0,
        marker_size=8.0,
        ylabel_size=14
    ):
    mu  = np.asarray(cluster_error_means, float)
    sd  = np.sqrt(np.asarray(cluster_error_vars, float))
    cis = np.asarray(cluster_intervals, float)
    K = mu.size
    if cis.shape != (K, 2):
        raise ValueError(f"cluster_intervals must be (K,2), got {cis.shape}")

    # base palette (same one used for simplex)
    # packed positions
    x = np.arange(K) * gap

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    for xi, (lo, hi), m, cid, w in zip(x, cis_s, mu_s, orig_ids_s, weights):
        ax.vlines(xi, lo, hi, linewidth=vline_width, color=col)
        # mean marker: fill with blended color, edge in base color to reinforce identity
        ax.plot([xi], [m], 'o', ms=marker_size, color=col,
                markeredgecolor=base_col, markeredgewidth=0.8)

    ax.set_ylabel(r"$\eta_{r}(z)$", fontsize=ylabel_size)
    ax.set_ylim(0, 1)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.set_xticks([])
    ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
    plt.margins(x=0.01)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# def plot_cluster_intervals(
#         cluster_error_means,
#         cluster_error_vars,
#         cluster_intervals,
#         save_path=None,
#         cmap=None,
#         *,
#         figsize=(6, 2.3),   # smaller overall figure
#         gap=0.65,           # < 1.0 packs bars closer (0.5 = very tight)
#         vline_width=3.0,    # thicker CI lines
#         marker_size=7.5,    # bigger mean markers
#         ylabel_size=14      # bigger y-label
#     ):
#     import numpy as np
#     import matplotlib.pyplot as plt

#     mu  = np.asarray(cluster_error_means, float)
#     sd  = np.sqrt(np.asarray(cluster_error_vars, float))
#     cis = np.asarray(cluster_intervals, float)
#     K = mu.size
#     if cis.shape != (K, 2):
#         raise ValueError(f"cluster_intervals must be (K,2), got {cis.shape}")

#     # shared palette
#     if cmap is None:
#         cmap = make_cmap(K)
#     palette = np.asarray([cmap(i) for i in range(K)])   # indexable by cluster ID

#     # sort by decreasing mean, keep original cluster IDs
#     order = np.argsort(mu)[::-1]
#     mu_s, sd_s, cis_s = mu[order], sd[order], cis[order]
#     orig_ids_s = order

#     # packed x-positions
#     x = np.arange(K) * gap

#     fig, ax = plt.subplots(figsize=figsize, dpi=150)

#     # draw CI + mean with original-color
#     for xi, (lo, hi), m, cid in zip(x, cis_s, mu_s, orig_ids_s):
#         col = palette[cid]
#         ax.vlines(xi, lo, hi, linewidth=vline_width, color=col)
#         ax.plot([xi], [m], 'o', ms=marker_size, color=col)

#     ax.set_ylabel(r"$\eta_{r}(z)$", fontsize=ylabel_size)
#     ax.set_ylim(0, 1)
#     ax.grid(True, axis='y', linestyle='--', alpha=0.4)
#     ax.set_xticks([])

#     # tighten horizontal margins so bars appear closer on the page
#     ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
#     plt.margins(x=0.01)          # minimal extra x padding
#     plt.tight_layout()

#     if save_path is not None:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.show()
 
def _lerp_color(c_base, c_target, w):
    a = np.array(c_base[:3]); b = np.array(c_target[:3])
    rgb = (1.0 - w) * a + w * b
    return (*rgb, 1.0)

def make_error_aware_cluster_colors(K, cmap, cluster_error_means,
                                    red_color=(0.85, 0.12, 0.12),
                                    blend_strength=1.0,
                                    vmin_mu=None, vmax_mu=None):
    """
    Returns colors_k: shape (K,4), one color per cluster id (0..K-1),
    blending each cluster's base color toward `red_color` according to its mean error.
    """
    if cmap is None:
        # assumes you already have make_cmap(K)
        cmap = make_cmap(K)

    base_palette = np.asarray([cmap(i) for i in range(K)])  # (K,4)
    mu = np.asarray(cluster_error_means, float)

    mu_min = float(np.min(mu)) if vmin_mu is None else float(vmin_mu)
    mu_max = float(np.max(mu)) if vmax_mu is None else float(vmax_mu)
    denom = (mu_max - mu_min) if mu_max > mu_min else 1.0
    w = np.clip(((mu - mu_min) / denom) * blend_strength, 0.0, 1.0)  # (K,)

    colors_k = np.array([_lerp_color(base_palette[i], red_color, w[i]) for i in range(K)])
    return colors_k   

import numpy as np

import numpy as np
import matplotlib as mpl

def make_cluster_colors(cluster_error_means, cmap_name="YlOrRd"):
    mu = np.asarray(cluster_error_means, float)
    norm = mpl.colors.Normalize(vmin=float(mu.min()), vmax=float(mu.max()))
    cmap = mpl.cm.get_cmap(cmap_name)
    colors_k = np.array([cmap(norm(m)) for m in mu])  # shape (K, 4)
    return colors_k, norm, cmap  # you can use norm+cmap for a colorbar if needed


def print_shortest_20_intervals(cluster_error_means, cluster_intervals, cluster_counts=None, batch=None, num=50):
    """
    Sort clusters by CI length and print the first 20 with mean and CI.
    Shapes supported:
      - means: (k,)              intervals: (k, 2)               [and optional counts: (k,)]
      - means: (bs, k)           intervals: (bs, k, 2)           [and optional counts: (bs, k)]
    Args:
      batch: which batch to use if inputs are batched; default 0.
    """
    means = np.asarray(cluster_error_means)
    intervals = np.asarray(cluster_intervals)
    counts = None if cluster_counts is None else np.asarray(cluster_counts)

    # Select batch if needed
    if intervals.ndim == 3:
        b = 0 if batch is None else batch
        inter = intervals[b]          # (k, 2)
        mu = means[b]                 # (k,)
        cnt = None if counts is None else counts[b]  # (k,)
    elif intervals.ndim == 2:
        inter = intervals
        mu = means
        cnt = counts
    else:
        raise ValueError("cluster_intervals must be (k,2) or (bs,k,2)")

    # CI length
    lengths = inter[:, 1] - inter[:, 0]

    # Valid mask: finite; if counts provided, require > 0
    mask = np.isfinite(lengths)
    if cnt is not None:
        mask &= (cnt > 0)

    # Indices of the first 20 by increasing length
    kept = np.flatnonzero(mask)
    if kept.size == 0:
        print("No valid clusters to display.")
        return
    order = np.argsort(lengths[kept])[:num]
    top = kept[order]

    # Pretty print
    for rank, c in enumerate(top, 1):
        lo, up = inter[c]
        print(f"{rank:2d}. cluster {c:4d} | len={lengths[c]:.6f} | mean={mu[c]:.6f} | CI=[{lo:.6f}, {up:.6f}]"
              + ("" if cnt is None else f" | n={int(cnt[c])}"))

def save_cluster_legend_vertical(cmap, K, out_path="clusters_legend.pdf", 
                                 label=r"Level sets $\mathcal{X}_z$"):
    # import numpy as np
    # import matplotlib.pyplot as plt
    

    fig, ax = plt.subplots(figsize=(0.45, 2.8), dpi=200)
    strip = np.arange(K)[:, None]   # shape (K, 1)
    ax.imshow(strip, aspect="auto", cmap=cmap, interpolation="nearest",
              extent=[0, 1, 0, K], origin="lower")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    # optional label:
    ax.text(0.5, 1.05*K, label, ha="center", va="bottom", rotation=0, transform=ax.transData)
    plt.tight_layout(pad=0.1)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

# ---------- example usage ----------
if __name__ == "__main__":
    setup_publication_style()
    dataset = "imagenet_7_fit_cluster_False" #imagenet3
    ROOT = f"cluster_examples_{dataset}"
    data = np.load(os.path.join(ROOT, "cluster_results.npz"))
    embs = data["embs"]         # shape (N, 3)
    clusters = data["clusters"] # shape (N,) or (bs, N)
    print("embs", embs[:2])
        
    cluster_error_means = np.squeeze(data["cluster_error_means"])
    cluster_error_vars = np.squeeze(data["cluster_error_vars"])
    cluster_intervals = np.squeeze(data["cluster_intervals"])     # or None

#     K = int(ensure_1d_clusters(clusters).max()) + 1
#     blend_strength = 0.9
#     red_color = (0.85, 0.12, 0.12)
#     cmap = make_cmap(K)  # deterministic palette
#     # base palette indexed by cluster id
#     base_palette = np.asarray([cmap(i) for i in range(K)])

#     # error-aware colors_k: one RGBA per cluster
#     if cluster_error_means is not None:
#         mu = np.asarray(cluster_error_means, float)
#         mu_min, mu_max = float(mu.min()), float(mu.max())
#         denom = (mu_max - mu_min) if mu_max > mu_min else 1.0
#         w = np.clip(((mu - mu_min) / denom) * blend_strength, 0.0, 1.0)  # (K,)

#         def _lerp_rgb(a, b, t):
#             a = np.array(a[:3]); b = np.array(b[:3])
#             rgb = (1.0 - t) * a + t * b
#             return (*rgb, 1.0)

#         colors_k = np.array([_lerp_rgb(base_palette[i], red_color, w[i]) for i in range(K)])
#     else:
#         colors_k = base_palette
#     names = CLASS_INFO["cifar10"]["names"]
#     if "cifar10" in dataset:

#         plot_simplex_clusters(
#             embs=embs,
#             clusters=clusters,
#             class_labels=(names[0], names[1], names[2]),
#             out_path=os.path.join(ROOT, "simplex_clusters.pdf"),
#             s=5, alpha=0.6,
#             bar_mode="none",       # if you’re using a shared legend in LaTeX
#             cmap=cmap,
#             triangle_scale=0.80,   # smaller triangle
#             label_fs=20,           # bigger corner labels
#             label_pad=0.01,         # push labels a bit farther out if needed
#             cluster_error_means=cluster_error_means,          # <── NEW
#             red_color=red_color,      # <── NEW (match CI fn default)
#             blend_strength=blend_strength                 # <── NEW
#         )

        
#         # plot_simplex_clusters(
#         #     embs=embs,
#         #     clusters=clusters,
#         #     class_labels=(names[0], names[1], names[2]),  # rename as you like
#         #     out_path=os.path.join(ROOT, "simplex_clusters.png"),
#         #     s=6,
#         #     alpha=0.6,
#         #     # legend_max=15
#         # )

#     plot_cluster_intervals(
#         cluster_error_means=cluster_error_means,
#         cluster_error_vars=cluster_error_vars,
#         cluster_intervals=cluster_intervals,
#         save_path=os.path.join(ROOT, "conf_intervals.png"),
#         cmap=cmap,
#         figsize=(4.2, 3),   # smaller figure
#         blend_strength=0.9,
#         gap=0.55,           # bars closer together (↓ => tighter)
#         vline_width=3.2,    # thicker CIs
#         marker_size=8.0,    # larger mean dots
#         ylabel_size=14      # bigger y-axis label
#     )
#     save_cluster_legend_vertical(cmap, K, out_path=os.path.join(ROOT, "clusters_legend.pdf"))
    
# #     plot_cluster_intervals(
# #         cluster_error_means=cluster_error_means,
# #         cluster_error_vars=cluster_error_vars,
# #         cluster_intervals=cluster_intervals,     # or None

# #     save_path=os.path.join(ROOT, "conf_intervals.png"),
# # )   
#     # print(data["cluster_intervals"])
    print_shortest_20_intervals(
        cluster_error_means=cluster_error_means,
        cluster_intervals=cluster_intervals,
    # optionally:
    # cluster_counts=cluster_counts,
    # batch=0,  # if your arrays are batched
    )
    # print("cluster_intervals", cluster_intervals[40])

    
