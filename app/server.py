from __future__ import annotations

import json
import mimetypes
import os
import pickle
import sys
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import pandas as pd


APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
CUSTOM_WIKI_DIR = REPO_ROOT / "WikiCS" / "custom-wiki"

sys.path.insert(0, str(CUSTOM_WIKI_DIR))

from utils.graph_utils import fuzzy_search, load_graph_data, resolve_title  # noqa: E402


DATA_PATH = CUSTOM_WIKI_DIR / "data" / "data-embeddings.json"
BGE_PATH = CUSTOM_WIKI_DIR / "cap-embeddings" / "BAAI_bge-m3" / "master_embeddings.parquet"
N2V_PATH = CUSTOM_WIKI_DIR / "cap-embeddings" / "node2vec" / "master_embeddings.parquet"
ALPHA_PATH = CUSTOM_WIKI_DIR / "cache" / "recommendation-1" / "alpha.pkl"
GCN_PATH = CUSTOM_WIKI_DIR / "cache" / "gcn" / "gcn_only_results.pkl"
SAGE_PATH = CUSTOM_WIKI_DIR / "cache" / "graphSAGE" / "graphsage_results.pkl"
OLLAMA_BASE_URL = os.environ.get("WIKIPATH_OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("WIKIPATH_OLLAMA_MODEL", "llama3.2:3b")


MODEL_CATALOG = [
    {
        "id": "bge-only",
        "name": "BGE-M3 Only",
        "description": "Semantic-only retrieval using article content similarity."
    },
    {
        "id": "n2v-uncorrected",
        "name": "Node2Vec (uncorrected)",
        "description": "Pure graph structure without degree damping."
    },
    {
        "id": "n2v-corrected",
        "name": "Node2Vec (corrected)",
        "description": "Pure structure with popularity damping from the power-law correction."
    },
    {
        "id": "hybrid-corrected",
        "name": "BGE-M3 + Node2Vec (corrected)",
        "description": "Hybrid recommendation model blending semantics with corrected structural signal."
    },
    {
        "id": "gcn-only",
        "name": "GCN Only",
        "description": "Graph convolutional node embeddings used directly for structural recommendation."
    },
    {
        "id": "gcn-bge-fusion",
        "name": "GCN + BGE-M3 (rank fusion)",
        "description": "Ranking-level fusion between graph-learned and text-based similarity."
    },
    {
        "id": "hybrid-uncorrected",
        "name": "BGE-M3 + Node2Vec (uncorrected)",
        "description": "Direct semantic + structural concatenation without popularity correction."
    },
    {
        "id": "sage-only",
        "name": "GraphSAGE Only",
        "description": "GraphSAGE node embeddings used directly for structural recommendation."
    },
]


STARTER_TITLES = [
    "Natural_language_processing",
    "Operating_system",
    "Programming_language",
    "Database",
    "Cloud_computing",
    "World_Wide_Web",
    "Compiler",
    "Machine_translation",
]

LEARNING_PATH_SECTIONS = [
    {"id": "foundations", "title": "Foundations", "summary": "Articles to understand first."},
    {"id": "core", "title": "Core Concepts", "summary": "Articles central to the target topic."},
    {"id": "advanced", "title": "Advanced Topics", "summary": "Articles that deepen or specialize the topic."},
    {"id": "adjacent", "title": "Adjacent Topics", "summary": "Useful supporting or neighboring concepts."},
]


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return matrix / norms


def shorten_text(text: str, limit: int = 320) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rsplit(" ", 1)[0] + "..."


def prettify_title(title: str) -> str:
    return title.replace("_", " ")


def normalize_title_query(text: str) -> str:
    return " ".join(text.replace("_", " ").split()).strip().lower()


def linked_pattern_text(item: dict[str, object]) -> str:
    if item.get("linked_from_source") and item.get("linked_to_source"):
        return "bidirectional"
    if item.get("linked_from_source"):
        return "source_to_candidate"
    if item.get("linked_to_source"):
        return "candidate_to_source"
    return "no_direct_edge"


def canonicalize_title_for_match(title: str) -> str:
    return normalize_title_query(title)


@dataclass
class ArticleRecord:
    node_id: int
    title: str
    label: str
    excerpt: str
    in_degree: int
    out_degree: int
    total_degree: int


class RepoRecommender:
    def __init__(self) -> None:
        self.nodes, self.id_to_title, self.title_to_id, self.graph = load_graph_data(str(DATA_PATH))
        self.graph_undirected = self.graph.to_undirected()
        self.article_by_id: dict[int, ArticleRecord] = {}
        self.article_by_title: dict[str, ArticleRecord] = {}
        self.searchable_titles: list[str] = []
        self._build_articles()
        self._load_model_data()

    def _build_articles(self) -> None:
        for node in self.nodes:
            node_id = int(node["id"])
            record = ArticleRecord(
                node_id=node_id,
                title=node["title"],
                label=node.get("label", "Wikipedia article"),
                excerpt=shorten_text(node.get("raw_text", "")),
                in_degree=int(self.graph.in_degree(node_id)),
                out_degree=int(self.graph.out_degree(node_id)),
                total_degree=int(self.graph_undirected.degree(node_id)),
            )
            self.article_by_id[node_id] = record
            self.article_by_title[record.title] = record
        self.searchable_titles = sorted(self.article_by_title.keys(), key=str.lower)

    def _load_model_data(self) -> None:
        with open(ALPHA_PATH, "rb") as f:
            alpha_data = pickle.load(f)
        self.alpha = float(alpha_data["alpha"])

        df_bge = pd.read_parquet(BGE_PATH).sort_values("id").reset_index(drop=True)
        df_n2v = pd.read_parquet(N2V_PATH).sort_values("id").reset_index(drop=True)
        if not (df_bge["id"].astype(int).values == df_n2v["id"].astype(int).values).all():
            raise ValueError("BGE and Node2Vec embeddings are not aligned by id.")

        self.ids = df_bge["id"].astype(int).to_numpy()
        self.id_to_idx = {node_id: idx for idx, node_id in enumerate(self.ids)}

        self.bge_norm = normalize_rows(np.stack(df_bge["embedding"].values))
        self.n2v_raw = np.stack(df_n2v["embedding"].values).astype(np.float32)
        self.n2v_norm = normalize_rows(self.n2v_raw)

        degrees = dict(self.graph_undirected.degree())
        self.degree_penalty = np.array(
            [(np.log(max(degrees.get(int(node_id), 1), 1) + 1) ** self.alpha) for node_id in self.ids],
            dtype=np.float32,
        )

        n2v_scaled = self.n2v_raw / self.degree_penalty[:, np.newaxis]
        combined_uncorrected = np.concatenate([self.bge_norm, self.n2v_raw], axis=1)
        self.hybrid_uncorrected_norm = normalize_rows(combined_uncorrected)

        combined_corrected = np.concatenate([self.bge_norm, n2v_scaled], axis=1)
        self.hybrid_corrected_norm = normalize_rows(combined_corrected)

        with open(GCN_PATH, "rb") as f:
            gcn_data = pickle.load(f)
        self.gcn_node_to_idx = {int(k): int(v) for k, v in gcn_data["node_to_idx"].items()}
        self.gcn_idx_to_node = {int(k): int(v) for k, v in gcn_data["idx_to_node"].items()}
        self.gcn_norm = normalize_rows(np.asarray(gcn_data["final_embeddings"], dtype=np.float32))

        with open(SAGE_PATH, "rb") as f:
            sage_data = pickle.load(f)
        self.sage_node_to_idx = {int(k): int(v) for k, v in sage_data["node_to_idx"].items()}
        self.sage_idx_to_node = {int(k): int(v) for k, v in sage_data["idx_to_node"].items()}
        self.sage_norm = normalize_rows(np.asarray(sage_data["final_embeddings"], dtype=np.float32))

        common_ids = sorted(set(self.id_to_idx.keys()) & set(self.gcn_node_to_idx.keys()))
        self.common_ids = np.array(common_ids, dtype=np.int64)
        self.common_pos = {node_id: idx for idx, node_id in enumerate(common_ids)}
        self.common_bge_indices = np.array([self.id_to_idx[node_id] for node_id in common_ids], dtype=np.int32)
        self.common_gcn_indices = np.array([self.gcn_node_to_idx[node_id] for node_id in common_ids], dtype=np.int32)

    def list_models(self) -> list[dict[str, str]]:
        return MODEL_CATALOG

    def get_article_payload(self, title: str) -> dict[str, object]:
        record = self.article_by_title[title]
        return {
            "id": record.node_id,
            "title": record.title,
            "display_title": prettify_title(record.title),
            "label": record.label,
            "excerpt": record.excerpt,
            "canonical_title": record.title,
            "in_degree": record.in_degree,
            "out_degree": record.out_degree,
            "total_degree": record.total_degree,
        }

    def search(self, query: str, limit: int = 8) -> list[dict[str, object]]:
        query = query.strip()
        if not query:
            starter_records = [title for title in STARTER_TITLES if title in self.article_by_title]
            if len(starter_records) < limit:
                missing = [title for title in self.searchable_titles if title not in starter_records]
                starter_records.extend(missing[: max(0, limit - len(starter_records))])
            return [self.get_article_payload(title) for title in starter_records[:limit]]

        normalized_query = normalize_title_query(query)
        prefix_matches = []
        substring_matches = []
        for title in self.searchable_titles:
            normalized_title = normalize_title_query(title)
            if normalized_title.startswith(normalized_query):
                prefix_matches.append(title)
            elif normalized_query in normalized_title:
                substring_matches.append(title)

        matches = prefix_matches + substring_matches
        if not matches:
            fuzzy = fuzzy_search(query, self.title_to_id, threshold=0.45)
            matches = [title for title, _ in fuzzy]

        return [self.get_article_payload(title) for title in matches[:limit]]

    def resolve_query(self, query: str) -> str | None:
        if query in self.title_to_id:
            return query

        normalized_query = normalize_title_query(query)
        for title in self.searchable_titles:
            if normalize_title_query(title) == normalized_query:
                return title

        return resolve_title(query, self.title_to_id, threshold=0.6)

    def _top_results_from_scores(self, source_id: int, scores: np.ndarray, top_k: int) -> list[dict[str, object]]:
        scores = scores.astype(np.float32, copy=True)
        source_idx = self.id_to_idx.get(source_id)
        if source_idx is not None and source_idx < len(scores):
            scores[source_idx] = -np.inf

        ranked_indices = np.argsort(-scores)[:top_k]
        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            node_id = int(self.ids[idx])
            article = self.article_by_id[node_id]
            results.append(
                {
                    "rank": rank,
                    "id": node_id,
                    "title": article.title,
                    "display_title": prettify_title(article.title),
                    "label": article.label,
                    "excerpt": article.excerpt,
                    "canonical_title": article.title,
                    "in_degree": article.in_degree,
                    "out_degree": article.out_degree,
                    "total_degree": article.total_degree,
                    "score": float(scores[idx]),
                }
            )
        return results

    def _top_results_from_scores_with_ids(
        self,
        source_id: int,
        candidate_ids: np.ndarray,
        scores: np.ndarray,
        top_k: int,
    ) -> list[dict[str, object]]:
        scores = scores.astype(np.float32, copy=True)
        source_mask = candidate_ids == source_id
        if source_mask.any():
            scores[source_mask] = -np.inf

        ranked_indices = np.argsort(-scores)[:top_k]
        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            node_id = int(candidate_ids[idx])
            article = self.article_by_id[node_id]
            results.append(
                {
                    "rank": rank,
                    "id": node_id,
                    "title": article.title,
                    "display_title": prettify_title(article.title),
                    "label": article.label,
                    "excerpt": article.excerpt,
                    "canonical_title": article.title,
                    "in_degree": article.in_degree,
                    "out_degree": article.out_degree,
                    "total_degree": article.total_degree,
                    "score": float(scores[idx]),
                }
            )
        return results

    def _scores_bge(self, source_id: int) -> np.ndarray:
        src_idx = self.id_to_idx[source_id]
        return self.bge_norm[src_idx] @ self.bge_norm.T

    def _scores_n2v_uncorrected(self, source_id: int) -> np.ndarray:
        src_idx = self.id_to_idx[source_id]
        return self.n2v_norm[src_idx] @ self.n2v_norm.T

    def _scores_n2v_corrected(self, source_id: int) -> np.ndarray:
        src_idx = self.id_to_idx[source_id]
        base = self.n2v_norm[src_idx] @ self.n2v_norm.T
        return base / (self.degree_penalty[src_idx] * self.degree_penalty)

    def _scores_hybrid_corrected(self, source_id: int) -> np.ndarray:
        src_idx = self.id_to_idx[source_id]
        return self.hybrid_corrected_norm[src_idx] @ self.hybrid_corrected_norm.T

    def _scores_hybrid_uncorrected(self, source_id: int) -> np.ndarray:
        src_idx = self.id_to_idx[source_id]
        return self.hybrid_uncorrected_norm[src_idx] @ self.hybrid_uncorrected_norm.T

    def _scores_gcn_only(self, source_id: int) -> tuple[np.ndarray, np.ndarray]:
        src_idx = self.gcn_node_to_idx[source_id]
        scores = self.gcn_norm[src_idx] @ self.gcn_norm.T
        candidate_ids = np.array([self.gcn_idx_to_node[i] for i in range(len(self.gcn_idx_to_node))], dtype=np.int64)
        return candidate_ids, scores

    def _scores_sage_only(self, source_id: int) -> tuple[np.ndarray, np.ndarray]:
        src_idx = self.sage_node_to_idx[source_id]
        scores = self.sage_norm[src_idx] @ self.sage_norm.T
        candidate_ids = np.array([self.sage_idx_to_node[i] for i in range(len(self.sage_idx_to_node))], dtype=np.int64)
        return candidate_ids, scores

    def _scores_rank_fusion(self, source_id: int) -> np.ndarray:
        if source_id not in self.common_pos:
            raise KeyError(f"Source id {source_id} is not available in the shared fusion set.")

        bge_src_idx = self.id_to_idx[source_id]
        gcn_src_idx = self.gcn_node_to_idx[source_id]

        bge_scores = self.bge_norm[bge_src_idx] @ self.bge_norm[self.common_bge_indices].T
        gcn_scores = self.gcn_norm[gcn_src_idx] @ self.gcn_norm[self.common_gcn_indices].T

        ranked_bge = np.argsort(-bge_scores)
        ranked_gcn = np.argsort(-gcn_scores)

        rankpos_bge = np.empty(len(self.common_ids), dtype=np.int32)
        rankpos_gcn = np.empty(len(self.common_ids), dtype=np.int32)
        rankpos_bge[ranked_bge] = np.arange(1, len(self.common_ids) + 1, dtype=np.int32)
        rankpos_gcn[ranked_gcn] = np.arange(1, len(self.common_ids) + 1, dtype=np.int32)

        k_rrf = 60.0
        rrf_scores_common = (1.0 / (k_rrf + rankpos_bge)) + (1.0 / (k_rrf + rankpos_gcn))
        scores = np.full(len(self.ids), -np.inf, dtype=np.float32)
        scores[self.common_bge_indices] = rrf_scores_common.astype(np.float32)
        return scores

    def recommend(self, title: str, model_id: str, top_k: int = 20) -> dict[str, object]:
        resolved = self.resolve_query(title)
        if resolved is None:
            raise KeyError(f"No article found for query: {title}")
        if model_id not in {model["id"] for model in MODEL_CATALOG}:
            raise KeyError(f"Unknown model: {model_id}")

        source_id = self.title_to_id[resolved]
        if model_id == "bge-only":
            scores = self._scores_bge(source_id)
            results = self._top_results_from_scores(source_id, scores, top_k)
        elif model_id == "n2v-uncorrected":
            scores = self._scores_n2v_uncorrected(source_id)
            results = self._top_results_from_scores(source_id, scores, top_k)
        elif model_id == "n2v-corrected":
            scores = self._scores_n2v_corrected(source_id)
            results = self._top_results_from_scores(source_id, scores, top_k)
        elif model_id == "hybrid-corrected":
            scores = self._scores_hybrid_corrected(source_id)
            results = self._top_results_from_scores(source_id, scores, top_k)
        elif model_id == "gcn-only":
            candidate_ids, scores = self._scores_gcn_only(source_id)
            results = self._top_results_from_scores_with_ids(source_id, candidate_ids, scores, top_k)
        elif model_id == "hybrid-uncorrected":
            scores = self._scores_hybrid_uncorrected(source_id)
            results = self._top_results_from_scores(source_id, scores, top_k)
        elif model_id == "sage-only":
            candidate_ids, scores = self._scores_sage_only(source_id)
            results = self._top_results_from_scores_with_ids(source_id, candidate_ids, scores, top_k)
        else:
            scores = self._scores_rank_fusion(source_id)
            results = self._top_results_from_scores(source_id, scores, top_k)

        for item in results:
            rec_id = int(item["id"])
            item["linked_from_source"] = self.graph.has_edge(source_id, rec_id)
            item["linked_to_source"] = self.graph.has_edge(rec_id, source_id)

        return {
            "source": self.get_article_payload(resolved),
            "resolved_title": resolved,
            "model_id": model_id,
            "results": results,
        }

    def organize_learning_path(
        self,
        title: str,
        model_id: str,
        top_k: int = 20,
        organizer_model: str | None = None,
    ) -> dict[str, object]:
        recommendation_payload = self.recommend(title, model_id, top_k=top_k)
        organization = build_learning_path(
            recommendation_payload,
            organizer_model=organizer_model or OLLAMA_MODEL,
        )
        return {
            **organization,
            "source": recommendation_payload["source"],
            "resolved_title": recommendation_payload["resolved_title"],
            "model_id": model_id,
        }


RECOMMENDER = RepoRecommender()


def call_ollama_json(messages: list[dict[str, str]], model: str) -> tuple[str, dict[str, object]]:
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
        },
        "messages": messages,
    }
    data = json.dumps(payload).encode("utf-8")
    request = Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=140) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        raise RuntimeError(f"Ollama HTTP error: {exc.code} {exc.reason}") from exc
    except URLError as exc:
        raise RuntimeError(f"Ollama connection error: {exc.reason}") from exc
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc

    payload = json.loads(raw)
    content = payload.get("message", {}).get("content", "")
    return content, payload


def stream_ollama_content(messages: list[dict[str, str]], model: str, on_chunk) -> str:
    payload = {
        "model": model,
        "stream": True,
        "format": "json",
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
        },
        "messages": messages,
    }
    data = json.dumps(payload).encode("utf-8")
    request = Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    collected: list[str] = []
    try:
        with urlopen(request, timeout=180) as response:
            while True:
                line = response.readline()
                if not line:
                    break
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                part = json.loads(line)
                chunk = part.get("message", {}).get("content", "")
                if chunk:
                    collected.append(chunk)
                    on_chunk(chunk)
                if part.get("done"):
                    break
    except HTTPError as exc:
        raise RuntimeError(f"Ollama HTTP error: {exc.code} {exc.reason}") from exc
    except URLError as exc:
        raise RuntimeError(f"Ollama connection error: {exc.reason}") from exc
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc
    return "".join(collected)


def make_learning_path_prompt(recommendation_payload: dict[str, object]) -> list[dict[str, str]]:
    source = recommendation_payload["source"]
    candidates = []
    for item in recommendation_payload["results"]:
        candidates.append(
            {
                "title": item["title"],
                "label": item["label"],
                "rank": item["rank"],
                "score": round(float(item["score"]), 4),
                "link_pattern": linked_pattern_text(item),
            }
        )

    instructions = (
        "Return only valid JSON.\n"
        "You are assigning retrieved Wikipedia computer-science articles into learning-path buckets.\n"
        "Use only canonical candidate titles from the provided list.\n"
        "Allowed sections are exactly: foundations, core, advanced, adjacent.\n"
        "Do not invent titles.\n"
        "Do not duplicate titles.\n"
        "Keep each why field very short.\n"
        "Return exactly one placement for every candidate title.\n"
        "Output shape: {\"placements\": [{\"title\": \"Candidate_title\", \"rank\": 1, \"section\": \"foundations|core|advanced|adjacent\", \"why\": \"Short reason.\"}]}"
    )

    user_payload = {
        "target": {
            "canonical_title": source["title"],
            "label": source["label"],
        },
        "retrieval_model": recommendation_payload["model_id"],
        "candidate_articles": candidates,
        "allowed_sections": [section["id"] for section in LEARNING_PATH_SECTIONS],
    }

    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
    ]


def validate_learning_path(raw_obj: object, candidate_map: dict[str, dict[str, object]], target_title: str) -> tuple[dict[str, object], list[str]]:
    errors: list[str] = []
    if not isinstance(raw_obj, dict):
        return {}, ["Top-level JSON must be an object."]

    placements_obj = raw_obj.get("placements")
    if not isinstance(placements_obj, list):
        return {}, ["`placements` must be a list."]

    valid_section_ids = {section["id"] for section in LEARNING_PATH_SECTIONS}
    section_map: dict[str, dict[str, object]] = {}
    used_titles: set[str] = set()
    used_ranks: set[int] = set()
    for section_cfg in LEARNING_PATH_SECTIONS:
        section_map[section_cfg["id"]] = {
            "id": section_cfg["id"],
            "title": section_cfg["title"],
            "summary": section_cfg["summary"],
            "items": [],
        }

    for item in placements_obj:
        if not isinstance(item, dict):
            errors.append("Each placement must be an object.")
            continue
        title = item.get("title")
        rank = item.get("rank")
        section_id = item.get("section")
        why = item.get("why", "")
        normalized_title = canonicalize_title_for_match(str(title))
        if normalized_title not in candidate_map:
            errors.append(f"Unknown candidate title: {title}")
            continue
        if section_id not in valid_section_ids:
            errors.append(f"Invalid section id: {section_id}")
            continue
        candidate = candidate_map[normalized_title]
        if normalized_title in used_titles:
            errors.append(f"Duplicate title across placements: {title}")
            continue
        if not isinstance(rank, int):
            errors.append(f"Missing or invalid rank for title `{title}`.")
            continue
        if rank != int(candidate["rank"]):
            errors.append(f"Rank mismatch for title `{title}`: expected {candidate['rank']}, got {rank}")
            continue
        if rank in used_ranks:
            errors.append(f"Duplicate rank across placements: {rank}")
            continue
        if not isinstance(why, str) or not why.strip():
            errors.append(f"Missing or empty why for title `{title}`.")
            continue

        used_titles.add(normalized_title)
        used_ranks.add(rank)
        section_map[section_id]["items"].append(
            {
                **candidate,
                "why": " ".join(why.split()),
            }
        )

    total_items = sum(len(section["items"]) for section in section_map.values())
    if total_items != len(candidate_map):
        errors.append(f"Learning path must contain all {len(candidate_map)} ranked titles exactly once.")
    if len(used_ranks) != len(candidate_map):
        errors.append(f"Learning path must contain all {len(candidate_map)} ranks exactly once.")

    normalized = {
        "target": target_title,
        "view_type": "learning_path",
        "sections": [
            {
                **section_map[section_cfg["id"]],
                "items": sorted(section_map[section_cfg["id"]]["items"], key=lambda item: int(item["rank"])),
            }
            for section_cfg in LEARNING_PATH_SECTIONS
        ],
    }
    return normalized, errors


def heuristic_learning_path(recommendation_payload: dict[str, object]) -> dict[str, object]:
    source = recommendation_payload["source"]
    results = recommendation_payload["results"]
    by_degree = sorted(results, key=lambda item: (-int(item["total_degree"]), int(item["rank"])))
    foundations = by_degree[:5]
    foundation_ids = {item["id"] for item in foundations}

    remaining = [item for item in results if item["id"] not in foundation_ids]
    core = remaining[:7]
    remaining = remaining[7:]
    advanced = remaining[:4]
    adjacent = remaining[4:]

    def decorate(items: list[dict[str, object]], why: str) -> list[dict[str, object]]:
        return [{**item, "why": why} for item in items]

    return {
        "target": source["title"],
        "view_type": "learning_path",
        "sections": [
            {
                "id": "foundations",
                "title": "Foundations",
                "summary": "Heuristic fallback based on broader graph connectivity.",
                "items": decorate(foundations, "Chosen as a likely broad prerequisite from the retrieved set."),
            },
            {
                "id": "core",
                "title": "Core Concepts",
                "summary": "Highest-ranked central articles remaining after the foundations split.",
                "items": decorate(core, "Chosen as a central concept near the target topic."),
            },
            {
                "id": "advanced",
                "title": "Advanced Topics",
                "summary": "Lower-ranked articles that extend the core topic.",
                "items": decorate(advanced, "Chosen as a deeper or more specialized follow-up."),
            },
            {
                "id": "adjacent",
                "title": "Adjacent Topics",
                "summary": "Neighboring topics that may be useful but less central.",
                "items": decorate(adjacent, "Chosen as a supporting neighboring topic."),
            },
        ],
        "organizer_model": None,
        "mode": "heuristic_fallback",
        "warning": "The local LLM organizer was unavailable or returned invalid output, so a heuristic learning path was used instead.",
        "attempts": 0,
    }


def build_learning_path(recommendation_payload: dict[str, object], organizer_model: str) -> dict[str, object]:
    candidate_map = {
        canonicalize_title_for_match(str(item["title"])): item
        for item in recommendation_payload["results"]
    }
    base_messages = make_learning_path_prompt(recommendation_payload)
    messages = list(base_messages)
    last_error = "unknown error"
    last_content = ""

    for attempt in range(1, 2):
        try:
            content, _ = call_ollama_json(messages, organizer_model)
            last_content = content
        except RuntimeError as exc:
            last_error = str(exc)
            break

        try:
            raw_obj = json.loads(content)
        except json.JSONDecodeError as exc:
            last_error = f"JSON parse error: {exc.msg} at line {exc.lineno} column {exc.colno}"
            messages = list(base_messages) + [
                {
                    "role": "assistant",
                    "content": content,
                },
                {
                    "role": "user",
                    "content": (
                        f"Your previous output was invalid JSON. Error: {last_error}. "
                        "Return corrected JSON only. Do not include markdown or commentary."
                    ),
                },
            ]
            continue

        normalized, validation_errors = validate_learning_path(
            raw_obj,
            candidate_map=candidate_map,
            target_title=str(recommendation_payload["source"]["title"]),
        )
        if not validation_errors:
            return {
                **normalized,
                "organizer_model": organizer_model,
                "mode": "llm",
                "warning": None,
                "attempts": attempt,
                "raw_llm_output": last_content,
            }

        last_error = "; ".join(validation_errors)
        messages = list(base_messages) + [
            {
                "role": "assistant",
                "content": content,
            },
            {
                "role": "user",
                "content": (
                    "The previous JSON parsed but failed validation. "
                    f"Problems: {last_error}. "
                    "Return corrected JSON only. Use only candidate titles."
                ),
            },
        ]

    fallback = heuristic_learning_path(recommendation_payload)
    fallback["warning"] = f"{fallback['warning']} Last error: {last_error}"
    fallback["raw_llm_output"] = last_content
    return fallback


class AppHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/organize-stream":
            self._handle_organize_stream(parsed)
            return
        if parsed.path.startswith("/api/"):
            self._handle_api(parsed)
            return
        self._handle_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/organize":
            self._handle_organize()
            return
        self._send_json({"error": "Unknown API route."}, status=HTTPStatus.NOT_FOUND)

    def _handle_api(self, parsed) -> None:
        params = parse_qs(parsed.query)

        try:
            if parsed.path == "/api/health":
                self._send_json({"status": "ok"})
                return

            if parsed.path == "/api/models":
                self._send_json({"models": RECOMMENDER.list_models()})
                return

            if parsed.path == "/api/search":
                query = params.get("q", [""])[0]
                limit = int(params.get("limit", ["8"])[0])
                results = RECOMMENDER.search(query, limit=limit)
                self._send_json({"query": query, "results": results})
                return

            if parsed.path == "/api/recommend":
                title = unquote(params.get("title", [""])[0]).strip()
                model_id = params.get("model", ["hybrid-corrected"])[0]
                top_k = int(params.get("top_k", ["20"])[0])
                if not title:
                    self._send_json({"error": "Missing title parameter."}, status=HTTPStatus.BAD_REQUEST)
                    return
                payload = RECOMMENDER.recommend(title, model_id, top_k=top_k)
                self._send_json(payload)
                return

            self._send_json({"error": "Unknown API route."}, status=HTTPStatus.NOT_FOUND)
        except KeyError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # pragma: no cover - best effort local app server
            self._send_json({"error": f"Internal server error: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_organize(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length).decode("utf-8") if content_length else "{}"
            body = json.loads(raw_body)

            title = str(body.get("title", "")).strip()
            model_id = str(body.get("model_id", "hybrid-corrected"))
            top_k = int(body.get("top_k", 20))
            organizer_model = str(body.get("organizer_model", OLLAMA_MODEL))

            if not title:
                self._send_json({"error": "Missing title in request body."}, status=HTTPStatus.BAD_REQUEST)
                return

            payload = RECOMMENDER.organize_learning_path(
                title,
                model_id=model_id,
                top_k=top_k,
                organizer_model=organizer_model,
            )
            self._send_json(payload)
        except json.JSONDecodeError as exc:
            self._send_json({"error": f"Invalid JSON body: {exc.msg}"}, status=HTTPStatus.BAD_REQUEST)
        except KeyError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # pragma: no cover
            self._send_json({"error": f"Internal server error: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_organize_stream(self, parsed) -> None:
        params = parse_qs(parsed.query)
        title = unquote(params.get("title", [""])[0]).strip()
        model_id = params.get("model_id", ["hybrid-corrected"])[0]
        top_k = int(params.get("top_k", ["20"])[0])
        organizer_model = params.get("organizer_model", [OLLAMA_MODEL])[0]

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        def send_event(event_name: str, payload: dict[str, object]) -> None:
            data = json.dumps(payload, ensure_ascii=True)
            self.wfile.write(f"event: {event_name}\n".encode("utf-8"))
            self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
            self.wfile.flush()

        try:
            recommendation_payload = RECOMMENDER.recommend(title, model_id, top_k=top_k)
            candidate_map = {
                canonicalize_title_for_match(str(item["title"])): item
                for item in recommendation_payload["results"]
            }
            messages = make_learning_path_prompt(recommendation_payload)
            raw_output = stream_ollama_content(
                messages,
                organizer_model,
                lambda chunk: send_event("token", {"chunk": chunk}),
            )

            raw_obj = json.loads(raw_output)
            normalized, validation_errors = validate_learning_path(
                raw_obj,
                candidate_map=candidate_map,
                target_title=str(recommendation_payload["source"]["title"]),
            )

            if validation_errors:
                payload = heuristic_learning_path(recommendation_payload)
                payload["warning"] = (
                    "The local LLM organizer returned invalid structured output and a heuristic path was used instead. "
                    f"Last error: {'; '.join(validation_errors)}"
                )
                payload["raw_llm_output"] = raw_output
            else:
                payload = {
                    **normalized,
                    "organizer_model": organizer_model,
                    "mode": "llm",
                    "warning": None,
                    "attempts": 1,
                    "raw_llm_output": raw_output,
                }

            payload.update(
                {
                    "source": recommendation_payload["source"],
                    "resolved_title": recommendation_payload["resolved_title"],
                    "model_id": model_id,
                }
            )
            send_event("complete", payload)
        except Exception as exc:
            try:
                recommendation_payload = RECOMMENDER.recommend(title, model_id, top_k=top_k)
                payload = heuristic_learning_path(recommendation_payload)
                payload["warning"] = (
                    "The local LLM organizer was unavailable or timed out, so a heuristic path was used instead. "
                    f"Last error: {exc}"
                )
                payload["raw_llm_output"] = ""
                payload.update(
                    {
                        "source": recommendation_payload["source"],
                        "resolved_title": recommendation_payload["resolved_title"],
                        "model_id": model_id,
                    }
                )
                send_event("complete", payload)
            except Exception as inner_exc:  # pragma: no cover
                send_event("fatal", {"error": str(inner_exc)})

    def _handle_static(self, path: str) -> None:
        rel_path = "index.html" if path in {"/", ""} else path.lstrip("/")
        file_path = (APP_DIR / rel_path).resolve()
        if APP_DIR not in file_path.parents and file_path != APP_DIR:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not file_path.exists() or file_path.is_dir():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        mime_type, _ = mimetypes.guess_type(str(file_path))
        content_type = mime_type or "application/octet-stream"
        payload = file_path.read_bytes()

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    host = os.environ.get("WIKIPATH_HOST", "127.0.0.1")
    port = int(os.environ.get("WIKIPATH_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), AppHandler)
    print(f"WikiPath app available at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
