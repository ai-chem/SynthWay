"""
build_rag_index.py — lexical retrieval with dense-assisted expansion

Features:
  1) build           — build a TF-IDF index and a dense vocabulary index for query expansion;
  python synthway/build_rag_index.py \
  build \
  --json_dir synthway/json \
  --out_idx  synthway/index \

  2) search          — TF-IDF search with automatic semantic token expansion (BGE→TF-IDF);
  python synthway/build_rag_index.py \
  search \
  --idx synthway/index \
  --composition "Au" \
  --precursor "HAuCl4" \
  --pH 6.5 \
  --temperature 110 \
  --shape spherical \
  --min-lex-score 0.12 \
  -k 10

  3) export-context  — export retrieval_text from the top-k results.
python synthway/build_rag_index.py \
  export-context \
  --idx synthway/index \
  --composition "Au" \
  --precursor "HAuCl4" \
  --pH 6.5 \
  --temperature 110 \
  --min-lex-score 0.12 \
  -k 10 \
  --out synthway/ctx.txt


Search approach:
  • rank by TF-IDF cosine similarity with semantic query-token expansion always enabled:
    compute each token's embedding (BAAI/bge-large-en-v1.5), find similar TF-IDF vocabulary tokens by cosine similarity,
    append them to the query string, then compute TF-IDF scores.
"""


from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
import os, json, argparse, re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from rich import print as rprint
from rich.table import Table
from rich.panel import Panel

try:
    import faiss
    _FAISS_OK = True
except Exception:
    _FAISS_OK = False


# Optional sparse support for lexical TF-IDF search
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    import scipy.sparse as sp  # type: ignore
    _SPARSE_OK = True
except Exception:
    _SPARSE_OK = False

import pickle



DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_TOPK = 5

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_JSON_DIR = os.path.join(BASE_DIR, "json")
DEFAULT_INDEX_DIR = os.path.join(BASE_DIR, "index")

@dataclass
class Hit:
    sid: Optional[str]
    score: float
    composition: Optional[str]
    method: Optional[str]
    retrieval_text: Optional[str]
    path: Optional[str]
    # Extended slots for runtime access (populated from meta)
    solvent: Optional[str] = None
    temperature: Optional[float] = None
    pH: Optional[float] = None
    time: Optional[float] = None
    shape: Optional[str] = None
    crystal_system: Optional[str] = None
    size_length: Optional[float] = None
    size_width: Optional[float] = None
    size_depth: Optional[float] = None
    elements: Optional[List[str]] = None
    precursors: Optional[List[str]] = None
    additives: Optional[List[str]] = None
    seeds: Optional[List[str]] = None
    pdfno_list: Optional[List[str]] = None

# Utilities
def _coalesce_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    if s.lower() == "unknown" or s == "-999":
        return None
    return s

def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str):
            s = x.strip().replace(",", ".")
            if not s or s.lower() == "unknown" or s == "-999":
                return None
            val = float(s)
            return None if val == -999 else val
        val = float(x)
        return None if val == -999 else val
    except Exception:
        return None

def build_retrieval_text_fallback(doc: Dict[str, Any]) -> str:
    """If retrieval_text is missing from JSON, build a flat record summary on the fly."""
    parts: List[str] = []
    def add(label: str, val: Any):
        if val is None:
            return
        if isinstance(val, dict):
            for k, v in val.items():
                if v is not None and v != "":
                    parts.append(f"{label}.{k}: {v}")
        elif isinstance(val, (list, tuple)):
            for v in val:
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        if k2 == "name" and v2:
                            parts.append(f"{label}: {v2}")
                elif v not in (None, "", "-999", "unknown"):
                    parts.append(f"{label}: {v}")
        else:
            s = str(val).strip()
            if s and s.lower() != "unknown" and s != "-999":
                parts.append(f"{label}: {s}")

    add("composition", _coalesce_str(doc.get("composition")))
    add("method", _coalesce_str(doc.get("method")))
    solv = doc.get("solvent")
    if isinstance(solv, dict):
        add("solvent", _coalesce_str(solv.get("name")))
        add("solvent.volume", _to_float(solv.get("volume")))
    else:
        add("solvent", _coalesce_str(solv))

    cond = doc.get("conditions") or {}
    add("temperature", _to_float(cond.get("temperature")))
    add("pH", _to_float(cond.get("pH")))

    prod = doc.get("product") or {}
    # Without the product. prefix
    add("crystalsystem", _coalesce_str(prod.get("crystal_system")))
    add("shape", _coalesce_str(prod.get("shape")))
    size = prod.get("size") or {}
    for k in ("length", "width", "depth"):
        add(f"size.{k}", _to_float(size.get(k)))

    for p in (doc.get("precursors") or []):
        add("precursor", p)
    for a in (doc.get("additives") or []):
        add("additive", a)
    for s in (doc.get("seeds") or []):
        add("seed", s)

    if doc.get("pdfno_list"):
        add("pdfno_list", doc.get("pdfno_list"))
    return " | ".join(parts)

def format_query(
    composition: Optional[str] = None,
    method: Optional[str] = None,
    shape: Optional[str] = None,
    temperature: Optional[float] = None,
    solvent: Optional[str] = None,
    additives: Optional[List[str]] = None,
    precursors: Optional[List[str]] = None,
    seeds: Optional[List[str]] = None,
    time: Optional[float] = None,
    pH: Optional[float] = None,
    crystalsystem: Optional[str] = None,
    size_length: Optional[float] = None,
    size_width: Optional[float] = None,
    size_depth: Optional[float] = None,
    pdfno_list: Optional[List[str]] = None,
) -> str:
    """Build a query matching retrieval_text style. In 'concat' mode, use one string containing key tokens."""
    parts: List[str] = []

    # Basic slots
    if composition:
        parts.append(f"composition: {composition}")
    if method:
        parts.append(f"method: {method}")
    if solvent:
        parts.append(f"solvent: {solvent}")
    if temperature is not None:
        parts.append(f"temperature: {temperature}")
    if time is not None:
        parts.append(f"time: {time}")
    if pH is not None:
        parts.append(f"pH: {pH}")
    if shape:
        parts.append(f"shape: {shape}")

    # Extended slots from product/size/pdf
    if crystalsystem:
        parts.append(f"crystalsystem: {crystalsystem}")
    if size_length is not None:
        parts.append(f"size.length: {size_length}")
    if size_width is not None:
        parts.append(f"size.width: {size_width}")
    if size_depth is not None:
        parts.append(f"size.depth: {size_depth}")

    # List-valued parameters
    if precursors:
        for p in precursors:
            parts.append(f"precursor: {p}")
    if additives:
        for a in additives:
            parts.append(f"additive: {a}")
    if seeds:
        for s in seeds:
            parts.append(f"seed: {s}")
    if pdfno_list:
        for pdf in pdfno_list:
            parts.append(f"pdfno_list: {pdf}")

    return " | ".join(parts)


# Index class
class RagIndex:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        # The model is needed only for semantic vocabulary/token expansion
        self.model = SentenceTransformer(model_name)
        self.meta: List[Dict[str, Any]] = []
        # TF-IDF (lexical)
        self.tfidf = None
        self.spmat = None
        self._lex_row_norms = None
        self._lexical_ok = False
        # vocabulary expansion (dense over TF-IDF vocab)
        self.vocab_terms = None
        self.vocab_index = None   # faiss.IndexFlatIP or np.ndarray
        self._vocab_fallback = True

    def _encode(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(texts, normalize_embeddings=True, batch_size=128)
        return emb.astype("float32")

    def build(self, json_dir: str) -> tuple[int, dict]:
        """Collect text and metadata ONLY from the JSON directory,
        deduplicate by synthesis_id and identical retrieval_text, and return counts."""
        texts, meta = [], []
        seen_sid = set()
        seen_rt = set()
        stats = {
            "total_files": 0,
            "skipped_empty": 0,
            "skipped_dup_sid": 0,
            "skipped_dup_text": 0,
        }

        def _push_doc(d: Dict[str, Any], path: Optional[str]=None):
            rt = d.get("retrieval_text")
            if not rt or not str(rt).strip():
                rt = build_retrieval_text_fallback(d)
            rt = (rt or "").strip()
            if not rt:
                stats["skipped_empty"] += 1
                return
            sid = d.get("synthesis_id")
            if sid and sid in seen_sid:
                stats["skipped_dup_sid"] += 1
                return
            if sid:
                seen_sid.add(sid)
            if rt in seen_rt:
                stats["skipped_dup_text"] += 1
                return
            seen_rt.add(rt)

            # --- Begin slot normalization for meta ---
            # solvent
            solv = d.get("solvent")
            if isinstance(solv, dict):
                solvent_name = _coalesce_str(solv.get("name"))
            else:
                solvent_name = _coalesce_str(solv)
            # conditions
            cond = d.get("conditions") or {}
            temperature = _to_float(cond.get("temperature"))
            pH = _to_float(cond.get("pH"))
            time_val = _to_float(cond.get("time"))
            # product/shape/crystal_system/size
            prod = d.get("product") or {}
            shape = _coalesce_str(prod.get("shape"))
            crystal_system = _coalesce_str(prod.get("crystal_system"))
            size = prod.get("size") or {}
            size_length = _to_float(size.get("length"))
            size_width = _to_float(size.get("width"))
            size_depth = _to_float(size.get("depth"))
            # elements (list of strings)
            raw_elems = d.get("elements") or []
            elements = []
            for e in raw_elems:
                s = _coalesce_str(e)
                if s:
                    elements.append(s)
            # precursor/additive/seed names
            def _collect_names(lst):
                out = []
                for c in lst or []:
                    if isinstance(c, dict):
                        n = _coalesce_str(c.get("name"))
                        if n:
                            out.append(n)
                    else:
                        s = _coalesce_str(c)
                        if s:
                            out.append(s)
                return out
            prec_names = _collect_names(d.get("precursors"))
            add_names = _collect_names(d.get("additives"))
            seed_names = _collect_names(d.get("seeds"))
            # pdfno_list
            pdf_raw = d.get("pdfno_list")
            if isinstance(pdf_raw, list):
                pdf_list = [str(x).strip() for x in pdf_raw if _coalesce_str(x) is not None]
            elif pdf_raw is not None:
                pdf_list = [str(pdf_raw).strip()]
            else:
                pdf_list = []
            # --- End slot normalization ---

            meta.append({
                "synthesis_id": sid,
                "composition": d.get("composition"),
                "method": d.get("method"),
                "retrieval_text": rt,
                "path": path,
                "solvent": solvent_name,
                "temperature": temperature,
                "pH": pH,
                "time": time_val,
                "shape": shape,
                "crystal_system": crystal_system,
                "size_length": size_length,
                "size_width": size_width,
                "size_depth": size_depth,
                "elements": elements,
                "precursor_names": prec_names,
                "additive_names": add_names,
                "seed_names": seed_names,
                "pdfno_list": pdf_list,
            })
            texts.append(rt)

        if not (json_dir and os.path.isdir(json_dir)):
            raise RuntimeError("Specify an existing --json_dir containing per-route JSON files.")

        for fn in os.listdir(json_dir):
            if not fn.lower().endswith(".json"):
                continue
            stats["total_files"] += 1
            fp = os.path.join(json_dir, fn)
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    d = json.load(f)
                _push_doc(d, fp)
            except Exception:
                stats["skipped_empty"] += 1
                continue

        if not texts:
            raise RuntimeError("No valid documents found for indexing")

        self.meta = meta

        # Build the lexical TF-IDF index if dependencies are available
        if _SPARSE_OK:
            # lowercase=False preserves token forms from retrieval_text
            # token_pattern excludes standalone '|' and ':' tokens
            self.tfidf = TfidfVectorizer(
                ngram_range=(1, 2),
                lowercase=False,
                token_pattern=r"(?u)\b[^|:\s]+\b"
            )
            self.spmat = self.tfidf.fit_transform(texts).tocsr().astype(np.float32)
            # Row norms for cosine similarity
            self._lex_row_norms = np.sqrt(self.spmat.multiply(self.spmat).sum(axis=1)).A1 + 1e-12
            self._lexical_ok = True
        else:
            self.tfidf = None
            self.spmat = None
            self._lex_row_norms = None
            self._lexical_ok = False

        # --- Semantic query expansion: build a dense index over the TF-IDF vocabulary ---
        try:
            # require TF-IDF and dense model
            if self._lexical_ok and isinstance(self.tfidf, TfidfVectorizer):
                # list of terms in TF-IDF vocabulary, in index order
                feature_names = list(self.tfidf.get_feature_names_out())
                # filter out service labels and numeric-only tokens
                _stop = {"composition","method","solvent","temperature","time","pH","precursor","additive","seed","size","crystalsystem","solvent.volume","product","shape"}
                def _is_number(tok: str) -> bool:
                    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", tok))
                terms_filtered = [t for t in feature_names if t not in _stop and not _is_number(t)]
                # encode terms with dense model (if available)
                try:
                    embs = self.model.encode(terms_filtered, normalize_embeddings=True, batch_size=256).astype("float32")
                except Exception:
                    embs = None
                if embs is not None and embs.shape[0] >= 1:
                    if _FAISS_OK:
                        self.vocab_index = faiss.IndexFlatIP(embs.shape[1])
                        self.vocab_index.add(embs)
                        self._vocab_fallback = False
                    else:
                        self.vocab_index = embs    # (V, D)
                        self._vocab_fallback = True
                    self.vocab_terms = terms_filtered
        except Exception:
            # Best effort: ignore errors while building the vocabulary index
            self.vocab_terms = None
            self.vocab_index = None
            self._vocab_fallback = True


        return len(self.meta), stats

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        # Metadata
        with open(os.path.join(out_dir, "meta.jsonl"), "w", encoding="utf-8") as f:
            for m in self.meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        # Save model information for expansion
        with open(os.path.join(out_dir, "model.json"), "w", encoding="utf-8") as f:
            json.dump({"model": self.model_name}, f, ensure_ascii=False)
        # Save the lexical index, if available
        if self._lexical_ok and self.tfidf is not None and self.spmat is not None:
            with open(os.path.join(out_dir, "tfidf.pkl"), "wb") as f:
                pickle.dump(self.tfidf, f)
            sp.save_npz(os.path.join(out_dir, "tfidf.npz"), self.spmat)
            if self._lex_row_norms is not None:
                np.save(os.path.join(out_dir, "tfidf_row_norms.npy"), self._lex_row_norms)
        # Save the vocabulary index for semantic expansion, if available
        if self.vocab_terms is not None and self.vocab_index is not None:
            with open(os.path.join(out_dir, "vocab_terms.json"), "w", encoding="utf-8") as f:
                json.dump(self.vocab_terms, f, ensure_ascii=False)
            if not self._vocab_fallback and _FAISS_OK:
                faiss.write_index(self.vocab_index, os.path.join(out_dir, "vocab_index.faiss"))
            else:
                np.save(os.path.join(out_dir, "vocab_index.npy"), self.vocab_index)

    def load(self, in_dir: str):
        # Metadata
        self.meta = []
        meta_path = os.path.join(in_dir, "meta.jsonl")
        with open(meta_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    self.meta.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise json.JSONDecodeError(
                        f"{e.msg} in {meta_path} at line {line_no}",
                        e.doc,
                        e.pos,
                    ) from e
        # Expansion model
        try:
            with open(os.path.join(in_dir, "model.json"), "r", encoding="utf-8") as f:
                info = json.load(f)
            self.model_name = info.get("model", DEFAULT_MODEL)
        except Exception:
            self.model_name = DEFAULT_MODEL
        # re-init model (for token expansions)
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception:
            self.model = None
        # TF‑IDF
        self.tfidf, self.spmat, self._lex_row_norms = None, None, None
        self._lexical_ok = False
        try:
            with open(os.path.join(in_dir, "tfidf.pkl"), "rb") as f:
                self.tfidf = pickle.load(f)
            self.spmat = sp.load_npz(os.path.join(in_dir, "tfidf.npz")).tocsr()
            try:
                self._lex_row_norms = np.load(os.path.join(in_dir, "tfidf_row_norms.npy"))
            except Exception:
                self._lex_row_norms = np.sqrt(self.spmat.multiply(self.spmat).sum(axis=1)).A1 + 1e-12
            self._lexical_ok = True
        except Exception:
            self._lexical_ok = False
        # Vocabulary index for semantic expansion
        self.vocab_terms, self.vocab_index = None, None
        self._vocab_fallback = True
        try:
            terms_path = os.path.join(in_dir, "vocab_terms.json")
            if os.path.exists(terms_path):
                with open(terms_path, "r", encoding="utf-8") as f:
                    self.vocab_terms = json.load(f)
                if _FAISS_OK and os.path.exists(os.path.join(in_dir, "vocab_index.faiss")):
                    self.vocab_index = faiss.read_index(os.path.join(in_dir, "vocab_index.faiss"))
                    self._vocab_fallback = False
                elif os.path.exists(os.path.join(in_dir, "vocab_index.npy")):
                    self.vocab_index = np.load(os.path.join(in_dir, "vocab_index.npy"))
                    self._vocab_fallback = True
        except Exception:
            self.vocab_terms, self.vocab_index = None, None
            self._vocab_fallback = True
    def _analyze(self, text: str) -> List[str]:
        """Tokenize text with the same analyzer as the TF-IDF vectorizer."""
        if not self._lexical_ok or self.tfidf is None:
            return []
        analyzer = self.tfidf.build_analyzer()
        return analyzer(text)

    def expand_query_with_denselex(self, q_str: str, topk: int = 5, mincos: float = 0.35, max_new: int = 50) -> str:
        """
        Semantic query expansion: for each token in q_str, find the nearest tokens
        in the TF-IDF vocabulary by cosine similarity of BGE embeddings and append them to the query.
        """
        if self.vocab_terms is None or self.vocab_index is None or not self._lexical_ok or self.tfidf is None or self.model is None:
            return q_str
        toks = self._analyze(q_str)
        exist = set(toks)
        stop = {"composition","method","solvent","temperature","time","pH","precursor","additive","seed","size","crystalsystem","solvent.volume","product","shape"}
        def _is_number(tok: str) -> bool:
            return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", tok))
        # IDF mapping
        try:
            fn = list(self.tfidf.get_feature_names_out())
            idf_vals = list(self.tfidf.idf_)
            idf = {t: float(v) for t, v in zip(fn, idf_vals)}
        except Exception:
            idf = {}
        extra_terms: List[str] = []
        for t in toks:
            if t in stop or _is_number(t):
                continue
            try:
                qv = self.model.encode([t], normalize_embeddings=True, batch_size=1).astype("float32")[0]
            except Exception:
                continue
            if self._vocab_fallback or not _FAISS_OK or not hasattr(self.vocab_index, "search"):
                # NumPy fallback
                if isinstance(self.vocab_index, np.ndarray):
                    sims = (self.vocab_index @ qv.reshape(-1, 1)).ravel()
                    idxs = np.argsort(-sims)[:topk]
                    cands = [(self.vocab_terms[i], float(sims[i])) for i in idxs]
                else:
                    cands = []
            else:
                D, I = self.vocab_index.search(qv.reshape(1, -1), topk)
                cands = [(self.vocab_terms[i], float(s)) for s, i in zip(D[0], I[0])]
            for term, cos in cands:
                if term in exist or term in stop or _is_number(term):
                    continue
                if cos < float(mincos):
                    continue
                # Add terms using simple cosine * idf weighting
                w = cos * float(idf.get(term, 1.0))
                repeats = max(1, min(3, int(round(1 + 2 * w))))
                extra_terms.extend([term] * repeats)
                if len(extra_terms) >= max_new:
                    break
            if len(extra_terms) >= max_new:
                break
        if not extra_terms:
            return q_str
        return q_str + " | " + " | ".join(extra_terms)

    def _lex_scores(self, query: str) -> Optional[np.ndarray]:
        """Return TF-IDF cosine similarities for all documents, or None if sparse support is unavailable."""
        if not self._lexical_ok or self.tfidf is None or self.spmat is None or self.spmat.shape[0] == 0:
            return None
        try:
            q = self.tfidf.transform([query]).tocsr().astype(np.float32)
            q_norm = float(np.sqrt(q.multiply(q).sum()) + 1e-12)
            sims = (q @ self.spmat.T).toarray()[0]
            sims = sims / (q_norm * (self._lex_row_norms if self._lex_row_norms is not None else 1.0))
            return sims.astype("float32")
        except Exception:
            return None

    def retrieve_lexical(self, query: str, k: int = DEFAULT_TOPK, min_score: Optional[float] = None) -> List[Hit]:
        sims = self._lex_scores(query)
        if sims is None:
            return []
        if min_score is not None:
            # Apply the threshold, then select top-k
            mask = sims >= float(min_score)
            if not np.any(mask):
                return []
            cand = np.where(mask)[0]
            order = np.argsort(-sims[cand])[:k]
            idxs = cand[order]
        else:
            idxs = np.argsort(-sims)[:k]
        scores = sims[idxs]
        hits: List[Hit] = []
        for score, idx in zip(scores, idxs):
            if idx is None or idx < 0 or idx >= len(self.meta):
                continue
            m = self.meta[idx]
            hits.append(Hit(
                sid=m.get("synthesis_id"),
                score=float(score),
                composition=m.get("composition"),
                method=m.get("method"),
                retrieval_text=m.get("retrieval_text"),
                path=m.get("path"),
                solvent=m.get("solvent"),
                temperature=m.get("temperature"),
                pH=m.get("pH"),
                time=m.get("time"),
                shape=m.get("shape"),
                crystal_system=m.get("crystal_system"),
                size_length=m.get("size_length"),
                size_width=m.get("size_width"),
                size_depth=m.get("size_depth"),
                elements=m.get("elements"),
                precursors=m.get("precursor_names"),
                additives=m.get("additive_names"),
                seeds=m.get("seed_names"),
                pdfno_list=m.get("pdfno_list"),
            ))
        return hits


# ===== CLI =====
def show_hits(hits: List[Hit], query: str):
    rprint(Panel.fit(f"[bold]Query[/]: {query}"))
    tbl = Table(title="Top-k retrieved")
    tbl.add_column("score", justify="right")
    tbl.add_column("sid")
    tbl.add_column("composition")
    tbl.add_column("method")
    for h in hits:
        tbl.add_row(f"{h.score:.3f}", str(h.sid), str(h.composition), str(h.method))
    rprint(tbl)

def _load_json(path: Optional[str]) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def apply_post_filters(ri: RagIndex, hits: List[Hit], k: int,
                       filter_method: Optional[str]=None,
                       elements: Optional[List[str]]=None,
                       req_precursors: Optional[List[str]]=None,
                       req_additives: Optional[List[str]]=None,
                       req_seeds: Optional[List[str]]=None,
                       composition_filter: Optional[str]=None) -> List[Hit]:
    out: List[Hit] = []
    f_method = (filter_method or "").lower().strip()
    want_elems = [e for e in (elements or []) if e]
    req_prec = [s for s in (req_precursors or []) if s]
    req_add = [s for s in (req_additives or []) if s]
    req_sd  = [s for s in (req_seeds or []) if s]
    comp_f  = (composition_filter or "").lower().strip()

    def contains_ci(hay: str, needle: str) -> bool:
        return needle.lower() in hay.lower()

    for h in hits:
        # hard composition filter (if user specified composition in query)
        if comp_f:
            hc = h.composition or ""
            if not contains_ci(hc, comp_f):
                continue

        # method filter
        if f_method:
            m = h.method or ""
            if f_method not in str(m).lower():
                continue

        # element filter: require all
        if want_elems:
            elems = set(h.elements or [])
            if not all(e in elems for e in want_elems):
                continue

        # require-precursor
        if req_prec:
            names = [n for n in (h.precursors or [])]
            if not all(any(contains_ci(n, req) for n in names) for req in req_prec):
                continue

        # require-additive
        if req_add:
            names = [n for n in (h.additives or [])]
            if not all(any(contains_ci(n, req) for n in names) for req in req_add):
                continue

        # require-seed
        if req_sd:
            names = [n for n in (h.seeds or [])]
            if not all(any(contains_ci(n, req) for n in names) for req in req_sd):
                continue

        out.append(h)
        if len(out) >= k:
            break
    return out

# --- Slot coverage helpers ---
from typing import Any, Optional, Tuple

def _norm_str(x: Any) -> Optional[str]:
    s = _coalesce_str(x)
    return s.lower() if s is not None else None

def _compute_slot_coverage(h: Hit, args) -> Tuple[int, int]:
    """
    Computes (matched, total) user-specified slot coverage for a hit.
    All data is taken from Hit fields, not from JSON.
    """
    matched = 0
    total = 0

    # -- Single-valued string slots (with weights) --
    # composition is more important than method and other descriptive slots
    single_slots = [
        ("composition", getattr(args, "composition", None), h.composition, 2),
        ("method", getattr(args, "method", None), h.method, 1),
        ("shape", getattr(args, "shape", None), h.shape, 1),
        ("solvent", getattr(args, "solvent", None), h.solvent, 1),
        ("crystalsystem", getattr(args, "crystalsystem", None), h.crystal_system, 1),
    ]
    for slot, qval, docval, w in single_slots:
        qval_norm = _norm_str(qval)
        if qval_norm:
            total += w
            docval_norm = _norm_str(docval)
            if docval_norm and qval_norm in docval_norm:
                matched += w

    # -- Numeric slots --
    # (slotname, h.<field>, argval, tolerance)
    numeric_slots = [
        ("temperature", h.temperature, getattr(args, "temperature", None), 10.0),
        ("pH", h.pH, getattr(args, "pH", None), 0.3),
        ("time", h.time, getattr(args, "time", None), 10.0),
        ("size_length", h.size_length, getattr(args, "size_length", None), 0.25),
        ("size_width", h.size_width, getattr(args, "size_width", None), 0.25),
        ("size_depth", h.size_depth, getattr(args, "size_depth", None), 0.25),
    ]
    for slot, docval_raw, qval, tol in numeric_slots:
        if qval is not None:
            total += 1
            docval = _to_float(docval_raw)
            qvalf = _to_float(qval)
            if docval is not None and qvalf is not None:
                if slot in ("size_length", "size_width", "size_depth"):
                    # relative tolerance 25% if both > 0, else abs 10.0
                    if docval > 0 and qvalf > 0:
                        rel = abs(docval - qvalf) / max(abs(qvalf), abs(docval), 1e-6)
                        if rel <= 0.25:
                            matched += 1
                    else:
                        if abs(docval - qvalf) <= 10.0:
                            matched += 1
                elif slot == "time":
                    # Use abs tolerance 10.0
                    if abs(docval - qvalf) <= 10.0:
                        matched += 1
                else:
                    if abs(docval - qvalf) <= tol:
                        matched += 1

    # -- List-like slots --
    # Each: if user specified, total +=1; matched if all requested values are found (case-insensitive substring) among doc names
    def _doc_names(lst):
        # For list of strings, get lowercased names/strings
        out = []
        for c in lst or []:
            out.append(str(c).lower())
        return out
    list_slots = [
        ("precursors", getattr(args, "precursor", None), h.precursors),
        ("additives", getattr(args, "additive", None), h.additives),
        ("seeds", getattr(args, "seed", None), h.seeds),
    ]
    for slot, qvals, docvals in list_slots:
        qlist = [q for q in (qvals or []) if q]
        if qlist:
            total += 1
            docnames = _doc_names(docvals)
            # All requested values must be present (substring, case-insensitive)
            ok = True
            for req in qlist:
                req_norm = str(req).lower()
                if not any(req_norm in n for n in docnames):
                    ok = False
                    break
            if ok:
                matched += 1

    # -- pdfno_list --
    pdfnos = getattr(args, "pdfno", None)
    pdfnos_list = [x for x in (pdfnos or []) if x]
    if pdfnos_list:
        total += 1
        doc_pdf_list = []
        if isinstance(h.pdfno_list, list):
            doc_pdf_list = [str(x).strip().lower() for x in h.pdfno_list if x]
        elif h.pdfno_list is not None:
            doc_pdf_list = [str(h.pdfno_list).strip().lower()]
        # Match if any requested pdfno matches exactly (case-insensitive, stripped)
        ok = any(str(req).strip().lower() in doc_pdf_list for req in pdfnos_list)
        if ok:
            matched += 1

    return (matched, total)

def rerank_by_slot_coverage(hits: List[Hit], args) -> List[Hit]:
    if not hits:
        return hits
    coverage = {}
    total_slots = None
    for h in hits:
        m, t = _compute_slot_coverage(h, args)
        coverage[id(h)] = (m, t)
        if total_slots is None:
            total_slots = t
    if not total_slots:
        # If the user specifies no slots, retain the original ranking
        return hits
    hits_sorted = sorted(
        hits,
        key=lambda h: (coverage.get(id(h), (0, 0))[0], h.score),
        reverse=True,
    )
    return hits_sorted

def main():
    ap = argparse.ArgumentParser(description="Synthesis indexer and retriever (TF-IDF + automatic semantic expansion)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("build", help="Build an index from /json (TF-IDF and vocabulary index for expansion)")
    sp.add_argument("--json_dir", default=DEFAULT_JSON_DIR, help="Dir with per-route JSON files (default: ./json)")
    sp.add_argument("--out_idx", default=DEFAULT_INDEX_DIR, help="Output dir for index (default: ./index)")
    sp.add_argument("--model", default=DEFAULT_MODEL)

    sp = sub.add_parser("search", help="Top-k TF-IDF search with automatic semantic expansion (query in retrieval_text style)")
    sp.add_argument("--idx", default=DEFAULT_INDEX_DIR, help="Index dir (default: ./index)")
    sp.add_argument("--composition")
    sp.add_argument("--method")
    sp.add_argument("--shape")
    sp.add_argument("--temperature", type=float)
    sp.add_argument("--solvent")
    sp.add_argument("--additive", action="append")
    sp.add_argument("-k", type=int, default=DEFAULT_TOPK)

    sp.add_argument("--precursor", action="append")
    sp.add_argument("--seed", action="append")
    sp.add_argument("--time", type=float)
    sp.add_argument("--pH", type=float)
    # Expose new slots on CLI
    sp.add_argument("--crystalsystem")
    sp.add_argument("--size-length", type=float)
    sp.add_argument("--size-width", type=float)
    sp.add_argument("--size-depth", type=float)
    sp.add_argument("--pdfno", action="append")

    # weighted / concat modes
    # --mode, --w-*, --retriever, --alpha removed
    sp.add_argument("--min-lex-score", type=float, default=None, help="Minimum lexical cosine similarity (0..1)")
    # post-filters
    sp.add_argument("--filter-method")
    sp.add_argument("--element", action="append")
    sp.add_argument("--require-precursor", action="append")
    sp.add_argument("--require-additive", action="append")
    sp.add_argument("--require-seed", action="append")
    sp.add_argument("--preselect", type=int, default=10, help="Candidate multiplier before post-filtering (k*preselect)")

    sp = sub.add_parser("export-context", help="Export top-k context strings (TF-IDF + automatic semantic expansion)")
    sp.add_argument("--idx", default=DEFAULT_INDEX_DIR)
    sp.add_argument("--composition")
    sp.add_argument("--method")
    sp.add_argument("--shape")
    sp.add_argument("--temperature", type=float)
    sp.add_argument("--solvent")
    sp.add_argument("--additive", action="append")
    sp.add_argument("-k", type=int, default=DEFAULT_TOPK)

    sp.add_argument("--precursor", action="append")
    sp.add_argument("--seed", action="append")
    sp.add_argument("--pH", type=float)
    # Expose new slots on CLI for export-context
    sp.add_argument("--time", type=float)
    sp.add_argument("--crystalsystem")
    sp.add_argument("--size-length", type=float)
    sp.add_argument("--size-width", type=float)
    sp.add_argument("--size-depth", type=float)
    sp.add_argument("--pdfno", action="append")

    # weighted / concat modes
    # --mode, --w-*, --retriever, --alpha removed
    sp.add_argument("--min-lex-score", type=float, default=None, help="Minimum lexical cosine similarity (0..1)")

    # post-filters
    sp.add_argument("--filter-method")
    sp.add_argument("--element", action="append")
    sp.add_argument("--require-precursor", action="append")
    sp.add_argument("--require-additive", action="append")
    sp.add_argument("--require-seed", action="append")
    sp.add_argument("--preselect", type=int, default=10, help="Candidate multiplier before post-filtering (k*preselect)")

    sp.add_argument("--out", required=True)

    args = ap.parse_args()

    if args.cmd == "build":
        rprint(Panel.fit("[bold]Building index[/]"))
        ri = RagIndex(model_name=args.model)
        n, stats = ri.build(json_dir=args.json_dir)
        ri.save(args.out_idx)
        msg = (
            f"[green]OK[/] Indexed docs: {n}\nSaved: {args.out_idx}\n"
            f"Skipped: empty={stats['skipped_empty']}, dup_sid={stats['skipped_dup_sid']}, dup_text={stats['skipped_dup_text']}"
        )
        rprint(Panel.fit(msg))
        return

    if args.cmd == "search":
        ri = RagIndex()
        ri.load(args.idx)
        n_pre = max(args.k * (args.preselect if hasattr(args, 'preselect') else 10), args.k)

        q_str = format_query(
            composition=args.composition,
            method=args.method,
            shape=args.shape,
            temperature=args.temperature,
            solvent=args.solvent,
            additives=args.additive,
            precursors=args.precursor,
            seeds=args.seed,
            time=getattr(args, "time", None),
            pH=getattr(args, "pH", None),
            crystalsystem=getattr(args, "crystalsystem", None),
            size_length=getattr(args, "size_length", None),
            size_width=getattr(args, "size_width", None),
            size_depth=getattr(args, "size_depth", None),
            pdfno_list=getattr(args, "pdfno", None),
        )

        q_str = ri.expand_query_with_denselex(q_str)

        raw = ri.retrieve_lexical(q_str, k=n_pre, min_score=getattr(args, "min_lex_score"))
        hits = apply_post_filters(
            ri, raw, args.k,
            filter_method=getattr(args, 'filter_method'),
            elements=getattr(args, 'element'),
            req_precursors=getattr(args, 'require_precursor'),
            req_additives=getattr(args, 'require_additive'),
            req_seeds=getattr(args, 'require_seed'),
            composition_filter=getattr(args, 'composition', None),
        )
        hits = rerank_by_slot_coverage(hits, args)
        show_hits(hits, q_str)
        return

    if args.cmd == "export-context":
        ri = RagIndex()
        ri.load(args.idx)
        n_pre = max(args.k * (args.preselect if hasattr(args, 'preselect') else 10), args.k)

        q_str = format_query(
            composition=args.composition,
            method=args.method,
            shape=args.shape,
            temperature=args.temperature,
            solvent=args.solvent,
            additives=args.additive,
            precursors=args.precursor,
            seeds=args.seed,
            time=getattr(args, "time", None),
            pH=getattr(args, "pH", None),
            crystalsystem=getattr(args, "crystalsystem", None),
            size_length=getattr(args, "size_length", None),
            size_width=getattr(args, "size_width", None),
            size_depth=getattr(args, "size_depth", None),
            pdfno_list=getattr(args, "pdfno", None),
        )

        q_str = ri.expand_query_with_denselex(q_str)

        raw = ri.retrieve_lexical(q_str, k=n_pre, min_score=getattr(args, "min_lex_score"))

        hits = apply_post_filters(
            ri, raw, args.k,
            filter_method=getattr(args, 'filter_method'),
            elements=getattr(args, 'element'),
            req_precursors=getattr(args, 'require_precursor'),
            req_additives=getattr(args, 'require_additive'),
            req_seeds=getattr(args, 'require_seed'),
            composition_filter=getattr(args, 'composition', None),
        )
        hits = rerank_by_slot_coverage(hits, args)
        lines: List[str] = [h.retrieval_text for h in hits if h.retrieval_text]
        # de-dup while preserving order
        seen = set()
        out_lines: List[str] = []
        for line in lines:
            if line and line not in seen:
                seen.add(line)
                out_lines.append(line)
        with open(args.out, "w", encoding="utf-8") as f:
            for line in out_lines:
                f.write(f"- {line}\n")
        rprint(Panel.fit(f"[green]Saved context[/]\n{args.out} ({len(out_lines)} lines)"))
        return

if __name__ == "__main__":
    main()
