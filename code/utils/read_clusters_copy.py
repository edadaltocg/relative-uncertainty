import os
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Optional (keep your publication style if available)
try:
    from .results_helper import setup_publication_style
except Exception:
    def setup_publication_style():
        pass

# ----------------------------- Basic helpers -----------------------------

def ensure_1d_clusters(clusters):
    clusters = np.asarray(clusters)
    if clusters.ndim == 2:
        clusters = clusters[0]
    return clusters.astype(np.int64)

def simplex3_to_xy(P):
    """Map 3-simplex barycentric coords (N,3) to 2D equilateral-triangle coords."""
    P = np.asarray(P, float)
    P = P / np.clip(P.sum(1, keepdims=True), 1e-12, None)
    x = P[:, 1] + 0.5 * P[:, 2]
    y = (math.sqrt(3) / 2.0) * P[:, 2]
    return x, y

def make_cmap(K):
    """Deterministic discrete palette for K clusters."""
    base = plt.get_cmap("tab20").colors
    if K <= 20:
        colors = base[:K]
    else:
        colors = [plt.cm.hsv(i / K) for i in range(K)]
    return ListedColormap(colors)

def _lerp_color(c_base, c_target, w):
    """Linear blend RGB of base -> target with weight w in [0,1]."""
    a = np.array(c_base[:3]); b = np.array(c_target[:3])
    rgb = (1.0 - w) * a + w * b
    return (*rgb, 1.0)

def make_error_aware_cluster_colors(
    K, cmap, cluster_error_means,
    red_color=(0.85, 0.12, 0.12),
    blend_strength=1.0,
    vmin_mu=None, vmax_mu=None
):
    """
    One RGBA per cluster id (0..K-1), blending each base color toward `red_color`
    according to the cluster mean error.

    Returns:
        colors_k : (K,4) ndarray of RGBA colors.
    """
    if cmap is None:
        cmap = make_cmap(K)

    base_palette = np.asarray([cmap(i) for i in range(K)])  # (K,4)
    mu = np.asarray(cluster_error_means, float)

    mu_min = float(np.min(mu)) if vmin_mu is None else float(vmin_mu)
    mu_max = float(np.max(mu)) if vmax_mu is None else float(vmax_mu)
    denom = (mu_max - mu_min) if mu_max > mu_min else 1.0
    w = np.clip(((mu - mu_min) / denom) * float(blend_strength), 0.0, 1.0)  # (K,)

    colors_k = np.array([_lerp_color(base_palette[i], red_color, w[i]) for i in range(K)])
    return colors_k

# ----------------------------- Legends/strips -----------------------------

def add_palette_strip_vertical(ax, cmap_or_colors, K, label="clusters"):
    """
    Vertical discrete strip. Accepts either a colormap or an explicit (K,4) color array.
    """
    if isinstance(cmap_or_colors, np.ndarray):
        cmap = ListedColormap(cmap_or_colors)
    else:
        cmap = cmap_or_colors

    iax = inset_axes(ax, width="3%", height="50%",
                     loc="center right", bbox_to_anchor=(0, 0, 1, 1),
                     bbox_transform=ax.transAxes, borderpad=1.2)
    strip = np.arange(K)[:, None]   # (K,1)
    iax.imshow(strip, aspect="auto", cmap=cmap, interpolation="nearest",
               extent=[0, 1, 0, K], origin="lower")
    iax.set_xticks([]); iax.set_yticks([])
    for sp in iax.spines.values():
        sp.set_visible(False)
    iax.text(0.5, 1.04, label, rotation=0, ha="center", va="center",
             transform=iax.transAxes)

def save_cluster_legend_vertical(cmap_or_colors, K, out_path="clusters_legend.pdf",
                                 label=r"Level sets $\mathcal{X}_z$"):
    """Standalone vertical legend strip saved to disk."""
    if isinstance(cmap_or_colors, np.ndarray):
        cmap = ListedColormap(cmap_or_colors)
    else:
        cmap = cmap_or_colors

    fig, ax = plt.subplots(figsize=(0.45, 2.8), dpi=200)
    strip = np.arange(K)[:, None]
    ax.imshow(strip, aspect="auto", cmap=cmap, interpolation="nearest",
              extent=[0, 1, 0, K], origin="lower")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(0.5, 1.05 * K, label, ha="center", va="bottom", rotation=0,
            transform=ax.transData)
    plt.tight_layout(pad=0.1)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

# ----------------------------- Main plotting -----------------------------

def plot_simplex_clusters(
    embs, clusters, class_labels=("class 0", "class 1", "class 2"),
    out_path="simplex_clusters.png", s=6, alpha=0.6,
    bar_mode="strip",               # "strip" | "colorbar" | "none"
    cmap=None,
    triangle_scale=0.82,
    label_fs=16,
    label_pad=0.04,
    colors=None

):
    """
    If cluster_error_means is provided (or colors_k is provided), the scatter uses
    error-aware colors consistent with the CI plot.
    """
    C = ensure_1d_clusters(clusters)
    E = np.asarray(embs, float)
    x_raw, y_raw = simplex3_to_xy(E)

    # Equilateral triangle vertices and centroid
    v0, v1, v2 = (0.0, 0.0), (1.0, 0.0), (0.5, math.sqrt(3) / 2.0)
    cx, cy = 0.5, (math.sqrt(3) / 6.0)

    def shrink_xy(x, y, scale, cx=cx, cy=cy):
        return cx + scale * (x - cx), cy + scale * (y - cy)

    # shrink vertices and data
    v0s = shrink_xy(*v0, triangle_scale)
    v1s = shrink_xy(*v1, triangle_scale)
    v2s = shrink_xy(*v2, triangle_scale)
    x, y = shrink_xy(x_raw, y_raw, triangle_scale)

    K = int(C.max()) + 1

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)

    # triangle border
    ax.plot([v0s[0], v1s[0], v2s[0], v0s[0]],
            [v0s[1], v1s[1], v2s[1], v0s[1]],
            lw=1.5, color="black")

    # inner grid lines (also shrunk)
    for t in np.linspace(0.1, 0.9, 5):
        x1, y1 = 0.5 * t, t * math.sqrt(3) / 2
        x2, y2 = 1 - 0.5 * t, t * math.sqrt(3) / 2
        X1, Y1 = shrink_xy(x1, y1, triangle_scale)
        X2, Y2 = shrink_xy(x2, y2, triangle_scale)
        ax.plot([X1, X2], [Y1, Y2], lw=0.5, color="#ccc", alpha=0.6)

    # scatter (error-aware or base colors)
    ax.scatter(x, y, c=colors[C], s=s, alpha=alpha, edgecolors="none")

    # legend handling
    if bar_mode == "strip":
        add_palette_strip_vertical(ax, colors, K, label="clusters")
    elif bar_mode == "colorbar":
        # map a ScalarMappable over cluster ids purely for a colorbar (discrete)
        sm = ScalarMappable(cmap=ListedColormap(colors))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.02, fraction=0.05)
        cbar.set_ticks([])
        cbar.set_label("clusters", rotation=0)

    # corner labels (push slightly outward from the shrunken vertices)
    def push_out(v, pad):
        return (v[0] + (v[0] - cx) * pad / triangle_scale,
                v[1] + (v[1] - cy) * pad / triangle_scale)

    lv0 = push_out(v0s, label_pad)
    lv1 = push_out(v1s, label_pad)
    lv2 = push_out(v2s, label_pad)

    ax.text(lv0[0], lv0[1], class_labels[0], ha="right", va="top",
            fontsize=label_fs, fontweight="medium", clip_on=False)
    ax.text(lv1[0], lv1[1], class_labels[1], ha="left", va="top",
            fontsize=label_fs, fontweight="medium", clip_on=False)
    ax.text(lv2[0], lv2[1], class_labels[2], ha="center", va="bottom",
            fontsize=label_fs, fontweight="medium", clip_on=False)

    # cosmetics
    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, v2[1] + 0.05)
    ax.set_xticks([]); ax.set_yticks([])
    for side in ("left", "bottom", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.tick_params(left=False, bottom=False)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_cluster_intervals(
    cluster_error_means,
    cluster_error_vars,
    cluster_intervals,
    save_path=None,
    cmap=None,
    *,
    red_color=(0.85, 0.12, 0.12),
    blend_strength=1.0,
    vmin_mu=None, vmax_mu=None,
    figsize=(6, 2.3),
    gap=0.55,
    vline_width=3.0,
    marker_size=8.0,
    ylabel_size=14,
    colors 
):
    """
    CI sticks + mean markers; colors consistent with simplex via make_error_aware_cluster_colors.
    """
    mu = np.asarray(cluster_error_means, float)
    sd = np.sqrt(np.asarray(cluster_error_vars, float))
    cis = np.asarray(cluster_intervals, float)
    K = mu.size
    if cis.shape != (K, 2):
        raise ValueError(f"cluster_intervals must be (K,2), got {cis.shape}")

    # sort by decreasing mean (but keep original IDs for color)
    order = np.argsort(mu)[::-1]
    mu_s, sd_s, cis_s = mu[order], sd[order], cis[order]
    orig_ids_s = order

    # packed x positions
    x = np.arange(K) * gap

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    for xi, (lo, hi), m, cid in zip(x, cis_s, mu_s, orig_ids_s):
        col = colors[cid]
        ax.vlines(xi, lo, hi, linewidth=vline_width, color=col)
        ax.plot([xi], [m], 'o', ms=marker_size, color=col)

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

# ----------------------------- Example usage -----------------------------

CLASS_INFO = {
    "cifar10": {
        "index": [5, 3, 8],
        "names": ["cat", "dog", "ship"],
    }
}

# 7 distinct, colorblind-safe colors
COLORS7 = [
    "#009E73",  # blue
    "#E64B35",  # vermillion
    "#F1C40F",  # green
    "#66CCEE",  # reddish purple   009E73
    "#2E86C1",  # orange
    "#E67E22",  # sky blue #7F3C8D  CC79A7  E67E22
    "#7F3C8D",  # yellow   7F3C8D
]

ORDER_COLOR = [1, 6, 5, 2, 3, 4, 0]
ORDER_COLOR = np.array(ORDER_COLOR)
ORDER_COLORS7 = np.array(COLORS7)[ORDER_COLOR][::-1]

COLORS_WARM_8 = [
    "#7D6E6C",  # strong red
    "#F39C12",  # orange
    "#E67E22",  # orange-ochre
    "#F1C40F",  # golden yellow
    "#C0392B",  # dark red
    "#D35400",  # pumpkin
    "#16A085",  # teal (contrast)
    "#2E86C1",  # blue (contrast)
]




if __name__ == "__main__":
    setup_publication_style()

    dataset = "cifar10_3"  # or "imagenet3"
    ROOT = f"cluster_examples_{dataset}"
    os.makedirs(ROOT, exist_ok=True)

    data = np.load(os.path.join(ROOT, "cluster_results.npz"))
    embs = data["embs"]                     # (N,3) barycentric
    clusters = data["clusters"]             # (N,) or (bs,N)
    cluster_error_means = np.squeeze(data["cluster_error_means"])
    cluster_error_vars  = np.squeeze(data["cluster_error_vars"])
    cluster_intervals   = np.squeeze(data["cluster_intervals"])  # (K,2)

    C1d = ensure_1d_clusters(clusters)
    K = int(C1d.max()) + 1

    cmap = ListedColormap(COLORS7[:K])     # exact one-to-one
    COLORS7 = np.array([cmap(i) for i in range(K)])  # per-cluster color



    # Names for the 3 vertices (adapt per dataset)
    if "cifar10" in dataset:
        names = CLASS_INFO["cifar10"]["names"]
    else:
        names = ("class 0", "class 1", "class 2")

    # Simplex plot (uses error-aware colors_k)
    plot_simplex_clusters(
        embs=embs,
        clusters=C1d,
        class_labels=(names[0], names[1], names[2]),
        out_path=os.path.join(ROOT, "simplex_clusters.pdf"),
        s=5, alpha=0.8,
        bar_mode="none",        # using a shared legend in LaTeX
        triangle_scale=0.80,
        label_fs=20,
        label_pad=0.01,
        colors=COLORS7,      # <- direct pass so it matches CI plot
    )

    # CI plot (same colors)
    plot_cluster_intervals(
        cluster_error_means=cluster_error_means,
        cluster_error_vars=cluster_error_vars,
        cluster_intervals=cluster_intervals,
        save_path=os.path.join(ROOT, "conf_intervals.png"),
        colors=COLORS7,  
        figsize=(4.2, 3),
        gap=0.55,
        vline_width=3.2,
        marker_size=8.0,
        ylabel_size=14
    )

    # Standalone vertical legend strip that matches both plots
    save_cluster_legend_vertical(ORDER_COLORS7, K, out_path=os.path.join(ROOT, "clusters_legend.pdf"))

    # Optional utility: print shortest intervals
    def print_shortest_20_intervals(cluster_error_means, cluster_intervals, cluster_counts=None, batch=None, num=20):
        means = np.asarray(cluster_error_means)
        intervals = np.asarray(cluster_intervals)
        counts = None if cluster_counts is None else np.asarray(cluster_counts)

        if intervals.ndim == 3:
            b = 0 if batch is None else batch
            inter = intervals[b]
            mu = means[b]
            cnt = None if counts is None else counts[b]
        elif intervals.ndim == 2:
            inter = intervals
            mu = means
            cnt = counts
        else:
            raise ValueError("cluster_intervals must be (k,2) or (bs,k,2)")

        lengths = inter[:, 1] - inter[:, 0]
        mask = np.isfinite(lengths)
        if cnt is not None:
            mask &= (cnt > 0)

        kept = np.flatnonzero(mask)
        if kept.size == 0:
            print("No valid clusters to display.")
            return
        order = np.argsort(lengths[kept])[:num]
        top = kept[order]
        for rank, c in enumerate(top, 1):
            lo, up = inter[c]
            print(f"{rank:2d}. cluster {c:4d} | len={lengths[c]:.6f} | mean={mu[c]:.6f} | "
                  f"CI=[{lo:.6f}, {up:.6f}]"
                  + ("" if cnt is None else f" | n={int(cnt[c])}"))

    print_shortest_20_intervals(cluster_error_means, cluster_intervals, num=20)
