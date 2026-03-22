import json
import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from difflib import SequenceMatcher


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
