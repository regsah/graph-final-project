"""
Community Similarity T-Test Analysis - Fast Version using cached edge data
"""

import json, os, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

CACHE_DIR = "../../cache/graph-analysis-4"
IMAGE_DIR = "./graph-analysis-4-img"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

print("=" * 60)
print("COMMUNITY SIMILARITY T-TEST ANALYSIS (Fast Version)")
print("=" * 60)

# Load graph structure
with open("../../data/data-embeddings.json", "r") as f:
    raw_data = json.load(f)

node_ids = {n["id"] for n in raw_data["nodes"]}

# Build cleaned graph
G_full = nx.DiGraph()
for n in raw_data["nodes"]:
    G_full.add_node(n["id"], title=n["title"], label=n.get("label", "Unknown"))
for n in raw_data["nodes"]:
    src = n["id"]
    for tgt in n.get("outlinks", []):
        if tgt in node_ids:
            G_full.add_edge(src, tgt)

isolated_nodes = list(nx.isolates(G_full))
G = G_full.copy()
G.remove_nodes_from(isolated_nodes)
G_und = G.to_undirected()

print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Load cached partitions
print("\n[1] Loading cached community partitions...")
with open(os.path.join(CACHE_DIR, "unified_louvain_partition.pkl"), "rb") as f:
    louvain_result = pickle.load(f)
louvain_partition = (
    louvain_result[0] if isinstance(louvain_result, tuple) else louvain_result
)
print(f"  Louvain: {len(set(louvain_partition.values()))} communities")

with open(os.path.join(CACHE_DIR, "unified_infomap_partition.pkl"), "rb") as f:
    infomap_result = pickle.load(f)
infomap_partition = (
    infomap_result[0] if isinstance(infomap_result, tuple) else infomap_result
)
print(f"  Infomap: {len(set(infomap_partition.values()))} communities")

# Load embeddings
print("\n[2] Loading embeddings...")
df_embeddings = pd.read_parquet(
    "../../cap-embeddings/BAAI_bge-m3/master_embeddings.parquet"
)
embedding_dict = dict(zip(df_embeddings["id"].astype(int), df_embeddings["embedding"]))
print(f"  Loaded {len(embedding_dict)} embeddings")

# Build edge-to-similarity lookup for undirected edges (MUCH FASTER with dict)
print("\n[3] Building edge-to-similarity lookup...")
edge_sim_lookup = {}
for u, v in G_und.edges():
    if u in embedding_dict and v in embedding_dict:
        sim = cosine_similarity(
            embedding_dict[u].reshape(1, -1), embedding_dict[v].reshape(1, -1)
        )[0][0]
        edge_sim_lookup[(min(u, v), max(u, v))] = sim
print(f"  Built lookup with {len(edge_sim_lookup)} edges")

# ============================================================
# T-TEST FOR LOUVAIN
# ============================================================
print("\n" + "=" * 60)
print("LOUVAIN T-TEST RESULTS")
print("=" * 60)

intra_sims_louvain = []
inter_sims_louvain = []

for (u, v), sim in edge_sim_lookup.items():
    if louvain_partition.get(u) == louvain_partition.get(v):
        intra_sims_louvain.append(sim)
    else:
        inter_sims_louvain.append(sim)

intra_arr_louvain = np.array(intra_sims_louvain)
inter_arr_louvain = np.array(inter_sims_louvain)

print(f"\nIntra-community edges:  {len(intra_arr_louvain)}")
print(f"Inter-community edges:  {len(inter_arr_louvain)}")
print(
    f"\nIntra-community similarity: mean={np.mean(intra_arr_louvain):.4f}, std={np.std(intra_arr_louvain):.4f}"
)
print(
    f"Inter-community similarity: mean={np.mean(inter_arr_louvain):.4f}, std={np.std(inter_arr_louvain):.4f}"
)

t_stat_louvain, p_value_louvain = stats.ttest_ind(intra_arr_louvain, inter_arr_louvain)

print(f"\n--- Louvain T-Test ---")
print(f"t-statistic: {t_stat_louvain:.6f}")
print(f"p-value:     {p_value_louvain:.6e}")
print(f"Significant (p < 0.05): {'YES' if p_value_louvain < 0.05 else 'NO'}")

pooled_std_louvain = np.sqrt(
    (np.var(intra_arr_louvain) + np.var(inter_arr_louvain)) / 2
)
cohens_d_louvain = (
    np.mean(intra_arr_louvain) - np.mean(inter_arr_louvain)
) / pooled_std_louvain
print(f"Cohen's d:  {cohens_d_louvain:.6f}")

# ============================================================
# T-TEST FOR INFOMAP
# ============================================================
print("\n" + "=" * 60)
print("INFOMAP T-TEST RESULTS")
print("=" * 60)

intra_sims_infomap = []
inter_sims_infomap = []

for (u, v), sim in edge_sim_lookup.items():
    if infomap_partition.get(u) == infomap_partition.get(v):
        intra_sims_infomap.append(sim)
    else:
        inter_sims_infomap.append(sim)

intra_arr_infomap = np.array(intra_sims_infomap)
inter_arr_infomap = np.array(inter_sims_infomap)

print(f"\nIntra-community edges:  {len(intra_arr_infomap)}")
print(f"Inter-community edges:  {len(inter_arr_infomap)}")
print(
    f"\nIntra-community similarity: mean={np.mean(intra_arr_infomap):.4f}, std={np.std(intra_arr_infomap):.4f}"
)
print(
    f"Inter-community similarity: mean={np.mean(inter_arr_infomap):.4f}, std={np.std(inter_arr_infomap):.4f}"
)

t_stat_infomap, p_value_infomap = stats.ttest_ind(intra_arr_infomap, inter_arr_infomap)

print(f"\n--- Infomap T-Test ---")
print(f"t-statistic: {t_stat_infomap:.6f}")
print(f"p-value:     {p_value_infomap:.6e}")
print(f"Significant (p < 0.05): {'YES' if p_value_infomap < 0.05 else 'NO'}")

pooled_std_infomap = np.sqrt(
    (np.var(intra_arr_infomap) + np.var(inter_arr_infomap)) / 2
)
cohens_d_infomap = (
    np.mean(intra_arr_infomap) - np.mean(inter_arr_infomap)
) / pooled_std_infomap
print(f"Cohen's d:  {cohens_d_infomap:.6f}")

# ============================================================
# SUMMARY TABLE
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)

summary_data = {
    "Algorithm": ["Louvain", "Infomap"],
    "Intra N": [len(intra_arr_louvain), len(intra_arr_infomap)],
    "Inter N": [len(inter_arr_louvain), len(inter_arr_infomap)],
    "Intra Mean": [np.mean(intra_arr_louvain), np.mean(intra_arr_infomap)],
    "Inter Mean": [np.mean(inter_arr_louvain), np.mean(inter_arr_infomap)],
    "t-statistic": [t_stat_louvain, t_stat_infomap],
    "p-value": [p_value_louvain, p_value_infomap],
    "Significant": [
        "YES" if p_value_louvain < 0.05 else "NO",
        "YES" if p_value_infomap < 0.05 else "NO",
    ],
    "Cohen's d": [cohens_d_louvain, cohens_d_infomap],
}
summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# ============================================================
# BOX PLOT WITH NODE COUNTS (Degree vs Similarity)
# ============================================================
print("\n" + "=" * 60)
print("DEGREE VS SIMILARITY BOX PLOT WITH NODE COUNTS")
print("=" * 60)

node_similarities = []
for u in G_und.nodes():
    neighbors = list(G_und.neighbors(u))
    if not neighbors or u not in embedding_dict:
        continue

    sims = []
    u_emb = embedding_dict[u].reshape(1, -1)
    for v in neighbors:
        if v in embedding_dict:
            sims.append(
                cosine_similarity(u_emb, embedding_dict[v].reshape(1, -1))[0][0]
            )

    if sims:
        node_similarities.append(
            {"node": u, "degree": G_und.degree(u), "avg_sim": np.mean(sims)}
        )

sim_df = pd.DataFrame(node_similarities)

bins = [0, 3, 10, 30, 100, 300, 500, 1000, 1500, 2000, 2500, 3000, 3500]
labels = [
    "0-3",
    "3-10",
    "10-30",
    "30-100",
    "100-300",
    "300-500",
    "500-1000",
    "1000-1500",
    "1500-2000",
    "2000-2500",
    "2500-3000",
    "3000-3500",
]
sim_df["degree_bin"] = pd.cut(sim_df["degree"], bins=bins, labels=labels)

bin_counts = sim_df["degree_bin"].value_counts().sort_index()
print("\nNode counts per degree bin:")
for label in labels:
    count = bin_counts.get(label, 0)
    print(f"  {label}: {count} nodes")

plt.figure(figsize=(16, 8))
box_data = []
for label in labels:
    data = sim_df[sim_df["degree_bin"] == label]["avg_sim"].values
    box_data.append(data)

box = plt.boxplot(box_data, tick_labels=labels, patch_artist=True)

colors = plt.cm.Spectral(np.linspace(0, 1, len(labels)))
for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

plt.title(
    "Node Degree vs. Average Neighbor Similarity\n(with node counts per bin)",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Degree Range", fontsize=12)
plt.ylabel("Average Cosine Similarity", fontsize=12)
plt.xticks(rotation=45)

ax = plt.gca()
xticks = ax.get_xticks()
for i, label in enumerate(labels):
    count = bin_counts.get(label, 0)
    y_max = ax.get_ylim()[1]
    plt.text(
        xticks[i],
        y_max - 0.015,
        f"n={count}",
        ha="center",
        va="top",
        fontsize=9,
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig(
    f"{IMAGE_DIR}/degree-vs-similarity-annotated.png", dpi=150, bbox_inches="tight"
)
plt.show()
print(f"\nSaved annotated plot to: {IMAGE_DIR}/degree-vs-similarity-annotated.png")

# ============================================================
# SAVE RESULTS
# ============================================================
results = {
    "louvain": {
        "intra_n": len(intra_arr_louvain),
        "inter_n": len(inter_arr_louvain),
        "intra_mean": float(np.mean(intra_arr_louvain)),
        "inter_mean": float(np.mean(inter_arr_louvain)),
        "t_statistic": float(t_stat_louvain),
        "p_value": float(p_value_louvain),
        "cohens_d": float(cohens_d_louvain),
        "significant": bool(p_value_louvain < 0.05),
    },
    "infomap": {
        "intra_n": len(intra_arr_infomap),
        "inter_n": len(inter_arr_infomap),
        "intra_mean": float(np.mean(intra_arr_infomap)),
        "inter_mean": float(np.mean(inter_arr_infomap)),
        "t_statistic": float(t_stat_infomap),
        "p_value": float(p_value_infomap),
        "cohens_d": float(cohens_d_infomap),
        "significant": bool(p_value_infomap < 0.05),
    },
    "degree_bins": {label: int(bin_counts.get(label, 0)) for label in labels},
}

with open(os.path.join(CACHE_DIR, "community_ttest_results.pkl"), "wb") as f:
    pickle.dump(results, f)

print("\n" + "=" * 60)
print("CONCLUSIONS")
print("=" * 60)
print(
    f"\nLOUVAIN:  t={t_stat_louvain:.4f}, p={p_value_louvain:.2e}, significant={p_value_louvain < 0.05}"
)
print(
    f"INFOMAP:  t={t_stat_infomap:.4f}, p={p_value_infomap:.2e}, significant={p_value_infomap < 0.05}"
)

if p_value_louvain < 0.05 and p_value_infomap < 0.05:
    print("\nBoth Louvain and Infomap show SIGNIFICANT difference between")
    print("intra-community and inter-community similarity (p < 0.05)")
elif p_value_louvain < 0.05:
    print("\nOnly Louvain shows SIGNIFICANT difference")
elif p_value_infomap < 0.05:
    print("\nOnly Infomap shows SIGNIFICANT difference")
else:
    print("\nNEITHER algorithm shows significant difference between")
    print("intra-community and inter-community similarity")
    print("(This suggests community detection does NOT successfully group")
    print("semantically related nodes based on embedding similarity)")

print("\nResults saved to:", os.path.join(CACHE_DIR, "community_ttest_results.pkl"))
