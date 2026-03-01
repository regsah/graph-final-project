"""
Generate all graph analysis plots and save them as images for the LaTeX report.
Run from the project root: python customwiki-exploration/generate_plots.py
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr
from collections import Counter

# ── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(SCRIPT_DIR, "img")
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "WikiCS", "custom-wiki")
os.makedirs(IMG_DIR, exist_ok=True)

# ── load data ──────────────────────────────────────────────────────────────
print("Loading data-embeddings.json …")
with open(os.path.join(DATA_DIR, "data-embeddings.json"), "r") as f:
    data = json.load(f)
print(f"  nodes: {len(data['nodes'])}")

print("Loading master_embeddings.parquet …")
df_emb = pd.read_parquet(os.path.join(DATA_DIR, "master_embeddings.parquet"))
print(f"  embeddings: {len(df_emb)}")

# ── build graph ────────────────────────────────────────────────────────────
node_ids = {node["id"] for node in data["nodes"]}

G = nx.DiGraph()
for node in data["nodes"]:
    G.add_node(node["id"], title=node["title"], label=node.get("label", "Unknown"))
for node in data["nodes"]:
    src = node["id"]
    for tgt in node.get("outlinks", []):
        if tgt in node_ids:
            G.add_edge(src, tgt)

print(f"  Graph – nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")

# ── embedding dict ─────────────────────────────────────────────────────────
emb_dict = {}
for _, row in df_emb.iterrows():
    emb_dict[str(row["id"])] = np.array(row["embedding"])
print(f"  Embedding dim: {len(list(emb_dict.values())[0])}")

# ═══════════════════════════════════════════════════════════════════════════
# 1. Degree Distribution
# ═══════════════════════════════════════════════════════════════════════════
degrees = [G.degree(n) for n in G.nodes()]
in_degrees = [G.in_degree(n) for n in G.nodes()]
out_degrees = [G.out_degree(n) for n in G.nodes()]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, vals, color, title in zip(
    axes,
    [degrees, in_degrees, out_degrees],
    ['#2196F3', '#4CAF50', '#FF9800'],
    ['Total Degree Distribution', 'In-Degree Distribution', 'Out-Degree Distribution'],
):
    ax.hist(vals, bins=50, color=color, edgecolor='black', alpha=0.85)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Degree', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.axvline(np.mean(vals), color='red', linestyle='--',
               label=f'Mean: {np.mean(vals):.2f}')
    ax.legend(fontsize=10)
plt.suptitle('Degree Distribution Histograms', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "degree_distribution.png"), dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ degree_distribution.png")

# ── log-log ────────────────────────────────────────────────────────────────
degree_count = Counter(degrees)
deg, cnt = zip(*sorted(degree_count.items()))
plt.figure(figsize=(8, 6))
plt.loglog(deg, cnt, 'o', markersize=5, color='#E91E63', alpha=0.7)
plt.title('Degree Distribution (Log-Log Scale)', fontsize=14, fontweight='bold')
plt.xlabel('Degree (log)', fontsize=12)
plt.ylabel('Frequency (log)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "degree_distribution_loglog.png"), dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ degree_distribution_loglog.png")

# degree stats string (for the latex)
deg_stats = (
    f"Total  – Min: {min(degrees)}, Max: {max(degrees)}, "
    f"Mean: {np.mean(degrees):.2f}, Median: {np.median(degrees):.0f}\n"
    f"In     – Min: {min(in_degrees)}, Max: {max(in_degrees)}, "
    f"Mean: {np.mean(in_degrees):.2f}, Median: {np.median(in_degrees):.0f}\n"
    f"Out    – Min: {min(out_degrees)}, Max: {max(out_degrees)}, "
    f"Mean: {np.mean(out_degrees):.2f}, Median: {np.median(out_degrees):.0f}"
)
print(deg_stats)

# ═══════════════════════════════════════════════════════════════════════════
# 2. Degree Assortativity
# ═══════════════════════════════════════════════════════════════════════════
r_assort = nx.degree_assortativity_coefficient(G)
assort_label = "ASSORTATIVE" if r_assort > 0 else ("DISASSORTATIVE" if r_assort < 0 else "NEUTRAL")
print(f"  Degree Assortativity Coefficient: {r_assort:.6f}  → {assort_label}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. Topic-Link Structure Assortativity  (embeddings)
# ═══════════════════════════════════════════════════════════════════════════
cosine_sims, pearson_corrs, euclidean_dists = [], [], []
skipped = 0
for u, v in G.edges():
    u_s, v_s = str(u), str(v)
    if u_s in emb_dict and v_s in emb_dict:
        eu, ev = emb_dict[u_s], emb_dict[v_s]
        cosine_sims.append(1 - cosine(eu, ev))
        pearson_corrs.append(pearsonr(eu, ev)[0])
        euclidean_dists.append(euclidean(eu, ev))
    else:
        skipped += 1
print(f"  Edges computed: {len(cosine_sims)}, skipped: {skipped}")

# random baseline
np.random.seed(42)
node_list = [n for n in G.nodes() if str(n) in emb_dict]
n_random = min(len(cosine_sims), 50000)
random_cosine, random_pearson, random_euclid = [], [], []
for _ in range(n_random):
    i, j = np.random.choice(len(node_list), 2, replace=False)
    eu, ev = emb_dict[str(node_list[i])], emb_dict[str(node_list[j])]
    random_cosine.append(1 - cosine(eu, ev))
    random_pearson.append(pearsonr(eu, ev)[0])
    random_euclid.append(euclidean(eu, ev))
print(f"  Random pairs: {n_random}")

# helper to draw comparison histograms
def draw_comparison(connected, random_vals, metric_name, color, fname):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(connected, bins=60, alpha=0.7, color=color, edgecolor='black',
            label=f'Connected (mean: {np.mean(connected):.4f})', density=True)
    ax.hist(random_vals, bins=60, alpha=0.5, color='#FF5722', edgecolor='black',
            label=f'Random (mean: {np.mean(random_vals):.4f})', density=True)
    ax.set_title(f'{metric_name}: Connected vs Random Node Pairs',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel(metric_name, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, fname), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {fname}")

draw_comparison(cosine_sims, random_cosine, 'Cosine Similarity', '#2196F3',
                'topic_link_cosine.png')
draw_comparison(pearson_corrs, random_pearson, 'Pearson Correlation', '#4CAF50',
                'topic_link_pearson.png')
draw_comparison(euclidean_dists, random_euclid, 'Euclidean Distance', '#9C27B0',
                'topic_link_euclidean.png')

# summary table data (printed for convenience)
print("\n=== Topic-Link Assortativity Summary ===")
summary = pd.DataFrame({
    'Metric': ['Cosine Similarity', 'Pearson Correlation', 'Euclidean Distance'],
    'Connected Mean': [np.mean(cosine_sims), np.mean(pearson_corrs), np.mean(euclidean_dists)],
    'Connected Std': [np.std(cosine_sims), np.std(pearson_corrs), np.std(euclidean_dists)],
    'Random Mean': [np.mean(random_cosine), np.mean(random_pearson), np.mean(random_euclid)],
    'Random Std': [np.std(random_cosine), np.std(random_pearson), np.std(random_euclid)],
    'Diff': [
        np.mean(cosine_sims) - np.mean(random_cosine),
        np.mean(pearson_corrs) - np.mean(random_pearson),
        np.mean(euclidean_dists) - np.mean(random_euclid),
    ],
}).round(4)
print(summary.to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════
# 4. Local Transitivity (Clustering Coefficient)
# ═══════════════════════════════════════════════════════════════════════════
G_und = G.to_undirected()
cc = nx.clustering(G_und)
cc_vals = list(cc.values())
cc_nonzero = [c for c in cc_vals if c > 0]
global_trans = nx.transitivity(G_und)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].hist(cc_vals, bins=50, color='#00BCD4', edgecolor='black', alpha=0.85)
axes[0].set_title('Local Transitivity Distribution', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Clustering Coefficient', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].axvline(np.mean(cc_vals), color='red', linestyle='--',
                label=f'Mean: {np.mean(cc_vals):.4f}')
axes[0].legend(fontsize=10)

axes[1].hist(cc_nonzero, bins=50, color='#FF9800', edgecolor='black', alpha=0.85)
axes[1].set_title('Local Transitivity (CC > 0)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Clustering Coefficient', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].axvline(np.mean(cc_nonzero), color='red', linestyle='--',
                label=f'Mean (>0): {np.mean(cc_nonzero):.4f}')
axes[1].legend(fontsize=10)

plt.suptitle('Local Transitivity Distribution Histograms',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "local_transitivity.png"), dpi=200, bbox_inches='tight')
plt.close()
print("  ✓ local_transitivity.png")

print(f"\n  CC stats – Mean: {np.mean(cc_vals):.4f}, Median: {np.median(cc_vals):.4f}, "
      f"Min: {min(cc_vals):.4f}, Max: {max(cc_vals):.4f}, Std: {np.std(cc_vals):.4f}")
print(f"  CC==0: {sum(1 for c in cc_vals if c==0)}, CC>0: {len(cc_nonzero)}")
print(f"  Global Transitivity: {global_trans:.4f}")

# ── write numerical results to a small text file (handy for latex) ─────────
with open(os.path.join(SCRIPT_DIR, "results.txt"), "w", encoding="utf-8") as f:
    f.write("=== Degree Statistics ===\n")
    f.write(deg_stats + "\n\n")
    f.write(f"=== Degree Assortativity ===\n")
    f.write(f"Coefficient: {r_assort:.6f}  ({assort_label})\n\n")
    f.write("=== Topic-Link Assortativity Summary ===\n")
    f.write(summary.to_string(index=False) + "\n\n")
    f.write("=== Local Transitivity ===\n")
    f.write(f"Mean: {np.mean(cc_vals):.4f}\n")
    f.write(f"Median: {np.median(cc_vals):.4f}\n")
    f.write(f"Min: {min(cc_vals):.4f}, Max: {max(cc_vals):.4f}\n")
    f.write(f"Std: {np.std(cc_vals):.4f}\n")
    f.write(f"CC==0: {sum(1 for c in cc_vals if c==0)}, CC>0: {len(cc_nonzero)}\n")
    f.write(f"Global Transitivity: {global_trans:.4f}\n")

print("\n✅  All plots saved to", IMG_DIR)
print("✅  Numerical results saved to results.txt")
