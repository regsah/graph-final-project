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


@dataclass
class ArticleRecord:
    node_id: int
    title: str
    label: str
    excerpt: str


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

        return {
            "source": self.get_article_payload(resolved),
            "resolved_title": resolved,
            "model_id": model_id,
            "results": results,
        }


RECOMMENDER = RepoRecommender()


class AppHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._handle_api(parsed)
            return
        self._handle_static(parsed.path)

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
