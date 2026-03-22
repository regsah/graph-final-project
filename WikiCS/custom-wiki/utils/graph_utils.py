import json
import os
import pickle
import random
import numpy as np
import pandas as pd
import networkx as nx
from difflib import SequenceMatcher
from scipy import sparse
from collections import deque


def load_graph_data(data_path="../../data/data-embeddings.json"):
    with open(data_path, "r") as f:
        data = json.load(f)
    nodes = data["nodes"]
    id_to_title = {n["id"]: n["title"] for n in nodes}
    title_to_id = {n["title"]: n["id"] for n in nodes}

    node_ids_set = {n["id"] for n in nodes}
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n["id"], title=n["title"])
    for n in nodes:
        src = n["id"]
        for tgt in n.get("outlinks", []):
            if tgt in node_ids_set:
                G.add_edge(src, tgt)

    return nodes, id_to_title, title_to_id, G


def fuzzy_search(query, title_to_id, threshold=0.5):
    """
    Return all titles in `title_to_id` whose fuzzy similarity to `query`
    exceeds `threshold`, sorted by score descending.
    Each entry is (title, score).
    """
    query_lower = query.lower()
    matches = []
    for title in title_to_id:
        score = SequenceMatcher(None, query_lower, title.lower()).ratio()
        if score >= threshold:
            matches.append((title, score))
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches


def resolve_title(query, title_to_id, threshold=0.6):
    """
    Return the best matching title for `query` using fuzzy matching.
    If `query` is an exact title, returns it directly.
    Otherwise returns the fuzzy match with the highest similarity score above `threshold`.
    Returns None if no match exceeds the threshold.
    """
    if query in title_to_id:
        return query

    matches = fuzzy_search(query, title_to_id, threshold=threshold)
    if not matches:
        return None

    return matches[0][0]


def load_embeddings(bge_path, n2v_path):
    """Load and align BGE-M3 and Node2Vec embeddings by node ID."""
    df_bge = pd.read_parquet(bge_path).sort_values("id").reset_index(drop=True)
    df_n2v = pd.read_parquet(n2v_path).sort_values("id").reset_index(drop=True)
    assert (df_bge["id"] == df_n2v["id"]).all(), "ID mismatch between embedding files"
    return df_bge, df_n2v


def build_normalized_embeddings(df_bge, df_n2v, G, alpha):
    """
    Build degree-normalised combined embeddings.

    Each node's embedding = concat(BGE-M3, Node2Vec / log(degree+1)^alpha)
    BGE-M3 is unchanged; Node2Vec is penalised by node popularity.

    Returns a DataFrame with columns [id, embedding].
    """
    deg_dict = {n: max(G.degree(n), 1) for n in df_bge["id"]}

    rows = []
    for _, row in df_bge.iterrows():
        nid = row["id"]
        e_bge = row["embedding"]
        e_n2v = df_n2v.loc[df_n2v["id"] == nid, "embedding"].iloc[0]
        d = deg_dict[nid]
        norm = (np.log(d + 1)) ** alpha
        rows.append({
            "id": nid,
            "embedding": np.concatenate([e_bge, e_n2v / norm]),
        })

    return pd.DataFrame(rows)


# ── Pure-numpy Node2Vec (no C-extensions required) ────────────────────────────
def _alias_setup(probs):
    """Build alias table for efficient sampling from a discrete distribution."""
    J = np.zeros(len(probs), dtype=np.int32)
    q = np.zeros(len(probs), dtype=np.float64)
    norm_const = float(np.sum(probs))
    probs = np.array(probs) / norm_const
    smaller = []
    larger = []
    for i, p in enumerate(probs):
        if p < 1.0:
            smaller.append(i)
        else:
            larger.append(i)
    while smaller and larger:
        s = smaller.pop()
        l = larger.pop()
        q[s] = probs[s] * len(probs)
        J[s] = l
        probs[l] = probs[l] + probs[s] - 1.0
        if probs[l] < 1.0:
            smaller.append(l)
        else:
            larger.append(l)
    for i in range(len(probs)):
        q[i] = probs[i] * len(probs)
    return J, q


def _alias_draw(J, q):
    """Draw a sample from the alias table."""
    i = int(np.random.randint(0, len(J)))
    u = np.random.random()
    if u < q[i]:
        return i
    return J[i]


def _compute_transition_probs(G, p, q):
    """
    Compute transition probabilities for Node2Vec random walks.
    Returns adjacency dict with alias tables for each node.
    """
    adj = {n: list(G.neighbors(n)) for n in G.nodes()}
    transition = {}
    for src in G.nodes():
        neighbors = adj[src]
        if not neighbors:
            transition[src] = None
            continue
        probs = []
        for dst in neighbors:
            w = 1.0
            if dst == src:
                w = 1.0 / p
            elif G.has_edge(dst, src):
                w = 1.0
            else:
                w = 1.0 / q
            probs.append(w)
        J, q_table = _alias_setup(probs)
        transition[src] = (J, q_table, adj[src])
    return transition


def node2vec_walks(G, p=1.0, q=1.0, walk_length=20, walks_per_node=10, seed=42):
    """
    Generate Node2Vec random walks using pure numpy/scipy (no C extensions).
    Matches torch_geometric.nn.Node2Vec with p, q parameters.

    Parameters:
        G: NetworkX graph (undirected or directed)
        p: Return parameter (1 = BFS-like, low = backtrack frequently)
        q: In-out parameter (1 = BFS-like, low = DFS-like)
        walk_length: Length of each random walk
        walks_per_node: Number of walks per source node
        seed: Random seed

    Returns:
        List of walks, each walk is a list of node IDs
    """
    np.random.seed(seed)
    random.seed(seed)
    transition = _compute_transition_probs(G, p, q)
    walks = []
    nodes = list(G.nodes())
    for _ in range(walks_per_node):
        random.shuffle(nodes)
        for node in nodes:
            if transition[node] is None:
                continue
            walk = [node]
            J, q_table, neighbors = transition[node]
            for _ in range(walk_length - 1):
                next_node = neighbors[_alias_draw(J, q_table)]
                walk.append(next_node)
            walks.append(walk)
    return walks


def node2vec_svd(G, num_walks=10, walk_length=80, embedding_dim=128, seed=42):
    """
    Node2Vec-style embeddings using Matrix Factorization (SVD on PPI-style matrix).

    Instead of Word2Vec on random walks, this builds a positive PMI-style matrix
    from random walks and factorises it with TruncatedSVD — pure numpy/scipy,
    no C extensions.

    Parameters:
        G: NetworkX graph (undirected)
        num_walks: Number of random walks per node
        walk_length: Length of each walk
        embedding_dim: Output embedding dimension
        seed: Random seed

    Returns:
        dict {node_id: numpy array (embedding_dim,)}
    """
    from sklearn.decomposition import TruncatedSVD
    np.random.seed(seed)
    random.seed(seed)

    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    walks = node2vec_walks(G, p=1.0, q=1.0, walk_length=walk_length,
                            walks_per_node=num_walks, seed=seed)

    window = 5
    count_mat = np.zeros((n, n), dtype=np.float64)
    for walk in walks:
        for i, src in enumerate(walk):
            s_idx = node_to_idx[src]
            start = max(0, i - window)
            end = min(len(walk), i + window + 1)
            for j in range(start, end):
                if i == j:
                    continue
                dst = walk[j]
                d_idx = node_to_idx[dst]
                count_mat[s_idx, d_idx] += 1.0

    pmi_mat = count_mat.copy()
    row_sum = count_mat.sum(axis=1)
    col_sum = count_mat.sum(axis=0)
    total = count_mat.sum()
    row_sum[row_sum == 0] = 1.0
    col_sum[col_sum == 0] = 1.0
    pmi_mat = np.log((count_mat * total) / (row_sum[:, None] * col_sum[None, :]) + 1e-10)
    pmi_mat = np.maximum(pmi_mat, 0)

    svd = TruncatedSVD(n_components=embedding_dim, random_state=seed)
    embeddings = svd.fit_transform(pmi_mat)

    return {nodes[i]: embeddings[i].astype(np.float32) for i in range(n)}
