"""Local memory: what the user dictated, what they corrected, which terms they use.

Kept in SQLite next to the models and indexed two ways — SQLite's own
full-text search and sentence embeddings from a small local model
(intfloat/multilingual-e5-small, MIT, ~470 MB, Turkish included). Nothing
leaves the machine.

What it is for:

- Whisper's prompt holds about 223 tokens, roughly thirty to forty terms.
  Once the vocabulary grows past that, the prompt has to be *chosen* per
  dictation. The hand-written list always goes in; learned terms are ranked
  by how often the user says them, how recently, and how close the sentences
  they were learned in are to what the user is dictating right now (the last
  dictations as the query), and added until the budget is full.
- Smart correction gets, as worked examples, the user's own past corrections
  whose sentences are most similar to the new transcript, instead of simply
  the latest ones.

Embeddings are optional: until the model is loaded (or if it never is) the
same decisions fall back to full-text search and recency.
"""

from __future__ import annotations

import math
import re
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_lock  # noqa: E402

# The embedding model is the one OpenClaw's memory search uses by default:
# EmbeddingGemma 300M, quantization-aware Q8_0 GGUF, run with llama.cpp on
# Metal. If llama.cpp is not installed (it compiles from source and can fail
# on some machines), the fallback is multilingual-e5-small through
# transformers — smaller, MIT, Turkish included.
GEMMA_MODEL_ID = "ggml-org/embeddinggemma-300m-qat-q8_0-GGUF"
GEMMA_FILE_PATTERN = "*.gguf"
EMBEDDING_MODEL_ID = "intfloat/multilingual-e5-small"
EMBEDDING_DIMENSIONS = 384
PROMPT_TOKEN_BUDGET = 200
RECENT_DICTATIONS_AS_QUERY = 3
MAX_EXAMPLES = 8


def fold(text: str) -> str:
    return text.lower().replace("ı", "i").replace("i̇", "i")


def term_occurs(term: str, text: str) -> bool:
    """The term as a whole word, possibly carrying a Turkish suffix after an apostrophe or glued on."""
    pattern = r"(?<!\w)" + re.escape(fold(term)) + r"(?:'|’|(?=[a-zçğıöşü]{1,8}(?!\w))|(?!\w))"
    return re.search(pattern, fold(text)) is not None


class GemmaEmbedder:
    """EmbeddingGemma through llama.cpp, exactly the model OpenClaw ships with.

    Texts are embedded as they are, no task prefixes, and normalized to unit
    length; the GGUF is fetched once into the app's model cache.
    """

    name = GEMMA_MODEL_ID

    def __init__(self) -> None:
        import glob
        import os

        from huggingface_hub import snapshot_download
        from llama_cpp import Llama

        try:
            snapshot = snapshot_download(GEMMA_MODEL_ID, allow_patterns=[GEMMA_FILE_PATTERN], local_files_only=True)
        except Exception:
            snapshot = snapshot_download(GEMMA_MODEL_ID, allow_patterns=[GEMMA_FILE_PATTERN])
        path = sorted(glob.glob(os.path.join(snapshot, GEMMA_FILE_PATTERN)))[0]
        # On the CPU, deliberately. A 300M embedding model takes 9 ms per
        # sentence either way (measured: 8 ms on Metal), and llama.cpp's Metal
        # backend once hung the worker at start-up — spinning in its residency
        # wait while another process was loading the correction model on the
        # same GPU — with the GPU lock held, so nothing else could load. Whisper
        # and the correction model keep the GPU to themselves.
        with gpu_lock.LOCK:
            self.model = Llama(model_path=path, embedding=True, n_gpu_layers=0, n_ctx=2048, verbose=False)
        self.device = "cpu"

    def encode(self, texts: list[str], kind: str = "passage") -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            with gpu_lock.LOCK:
                vector = self.model.embed(text)
            if vector and isinstance(vector[0], list):
                vector = vector[0]
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            out.append([value / norm for value in vector])
        return out


def make_embedder() -> Any:
    """EmbeddingGemma when llama.cpp is importable, otherwise the fallback."""
    try:
        import llama_cpp  # noqa: F401
    except Exception:  # noqa: BLE001
        return Embedder()
    try:
        return GemmaEmbedder()
    except Exception as error:  # noqa: BLE001
        print(f"hafıza: EmbeddingGemma yüklenemedi ({error}); yedek modele geçiliyor", file=sys.stderr, flush=True)
        return Embedder()


class Embedder:
    """multilingual-e5-small on the GPU: `query:` for what we search with,
    `passage:` for what is stored, mean pooling, unit length."""

    name = EMBEDDING_MODEL_ID

    def __init__(self) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer

        patterns = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                    "special_tokens_map.json", "sentencepiece.bpe.model"]
        try:
            snapshot = snapshot_download(EMBEDDING_MODEL_ID, allow_patterns=patterns, local_files_only=True)
        except Exception:
            snapshot = snapshot_download(EMBEDDING_MODEL_ID, allow_patterns=patterns)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if self.device == "mps" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        model = AutoModel.from_pretrained(snapshot, local_files_only=True, dtype=dtype).eval()
        with gpu_lock.LOCK:
            self.model = model.to(self.device)
        self.torch = torch

    def encode(self, texts: list[str], kind: str = "passage") -> list[list[float]]:
        if not texts:
            return []
        torch = self.torch
        batch = self.tokenizer(
            [f"{kind}: {text}" for text in texts], padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        with gpu_lock.LOCK, torch.inference_mode():
            batch = batch.to(self.device)
            hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
        return pooled.cpu().tolist()


def _pack(vector: list[float]) -> bytes:
    import array

    return array.array("f", vector).tobytes()


def _unpack(blob: bytes) -> list[float]:
    import array

    values = array.array("f")
    values.frombytes(blob)
    return list(values)


def _dot(a: list[float], b: list[float]) -> float:
    """Cosine of two unit vectors; zero when they come from different models
    (a vector of another dimension is not comparable, not merely dissimilar)."""
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


class Memory:
    def __init__(self, path: Path, emit: Callable[[dict[str, Any]], None] | None = None,
                 embedder_factory: Callable[[], Any] | None = None) -> None:
        self.path = path
        self.emit = emit or (lambda message: None)
        self.embedder_factory = embedder_factory or make_embedder
        self.lock = threading.RLock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.embedder: Any = None
        self.embedder_state = "off"
        self.embedder_detail = ""
        self.thread: threading.Thread | None = None
        self.fts = True
        self._schema()

    # MARK: schema

    def _schema(self) -> None:
        with self.lock:
            self.db.executescript(
                """
                CREATE TABLE IF NOT EXISTS dictations(
                    id INTEGER PRIMARY KEY, created_at REAL NOT NULL, app TEXT, text TEXT NOT NULL, embedding BLOB);
                CREATE TABLE IF NOT EXISTS corrections(
                    id INTEGER PRIMARY KEY, created_at REAL NOT NULL, wrong TEXT NOT NULL, right TEXT NOT NULL,
                    context TEXT NOT NULL, safe INTEGER NOT NULL DEFAULT 0, embedding BLOB,
                    UNIQUE(wrong, right));
                CREATE TABLE IF NOT EXISTS term_usage(
                    term TEXT PRIMARY KEY, count INTEGER NOT NULL DEFAULT 0, last_used REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS vocabulary(
                    term TEXT PRIMARY KEY, source TEXT NOT NULL, position INTEGER NOT NULL DEFAULT 0, added_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS contexts(
                    id INTEGER PRIMARY KEY, text TEXT NOT NULL UNIQUE, position INTEGER NOT NULL DEFAULT 0,
                    added_at REAL NOT NULL, embedding BLOB);
                """
            )
            # Full-text copies hold folded text (lowercase, dotless ı → i), so a
            # query for "anahtari" finds "anahtarı"; FTS5's own folding does
            # not know Turkish. Corrections index the sentence plus the right
            # spelling, so a transcript that already says "Claude" still finds
            # the "Cloudy → Claude" example.
            try:
                self.db.executescript(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS dictations_fts USING fts5(text);
                    CREATE VIRTUAL TABLE IF NOT EXISTS corrections_fts USING fts5(text);
                    """
                )
            except sqlite3.OperationalError:
                self.fts = False
            self.db.commit()

    # MARK: embedder lifecycle

    def report(self, state: str, detail: str = "") -> None:
        self.embedder_state, self.embedder_detail = state, detail
        self.emit({"event": "memory_state", **self.stats()})

    def start_embedder(self) -> None:
        if self.embedder is not None or (self.thread is not None and self.thread.is_alive()):
            self.emit({"event": "memory_state", **self.stats()})
            return
        self.thread = threading.Thread(target=self._load_embedder, name="memory-embedder", daemon=True)
        self.thread.start()

    def _load_embedder(self) -> None:
        try:
            self.report("loading", "Hafıza embedding modeli yükleniyor…")
            embedder = self.embedder_factory()
            with self.lock:
                self.embedder = embedder
            self.report("ready", getattr(embedder, "name", EMBEDDING_MODEL_ID))
            self.backfill()
        except Exception as error:  # noqa: BLE001
            self.report("failed", f"Embedding modeli yüklenemedi: {error}")

    def backfill(self, limit: int = 500) -> None:
        """Rows stored before the embedder was ready get their vectors now — and
        so do rows embedded by a different model. The fallback model
        (multilingual-e5-small, 384 dimensions) and EmbeddingGemma (768) are
        not comparable; when llama.cpp is installed later and the memory moves
        to EmbeddingGemma, everything the fallback embedded is redone."""
        with self.lock:
            if self.embedder is None:
                return
            width = len(self.embedder.encode(["boyut"])[0]) * 4   # float32 bytes per vector
            for table, column in (("dictations", "text"), ("corrections", "context"), ("contexts", "text")):
                rows = self.db.execute(
                    f"SELECT id, {column} AS text FROM {table} WHERE embedding IS NULL OR length(embedding) != ? "
                    f"ORDER BY id DESC LIMIT ?",
                    (width, limit),
                ).fetchall()
                for start in range(0, len(rows), 16):
                    chunk = rows[start:start + 16]
                    vectors = self.embedder.encode([row["text"] for row in chunk])
                    for row, vector in zip(chunk, vectors):
                        self.db.execute(f"UPDATE {table} SET embedding = ? WHERE id = ?", (_pack(vector), row["id"]))
                self.db.commit()

    def _embed(self, text: str) -> bytes | None:
        if self.embedder is None:
            return None
        try:
            return _pack(self.embedder.encode([text])[0])
        except Exception as error:  # noqa: BLE001
            print(f"hafıza: embedding hesaplanamadı — {error}", file=sys.stderr, flush=True)
            return None

    def _query_vector(self, text: str) -> list[float] | None:
        if self.embedder is None or not text.strip():
            return None
        try:
            return self.embedder.encode([text], kind="query")[0]
        except Exception:  # noqa: BLE001
            return None

    # MARK: writing

    def remember_dictation(self, text: str, app: str = "", terms: list[str] | None = None) -> None:
        text = text.strip()
        if not text:
            return
        now = time.time()
        with self.lock:
            cursor = self.db.execute(
                "INSERT INTO dictations(created_at, app, text, embedding) VALUES (?, ?, ?, ?)",
                (now, app, text, self._embed(text)),
            )
            if self.fts:
                self.db.execute("INSERT INTO dictations_fts(rowid, text) VALUES (?, ?)", (cursor.lastrowid, fold(text)))
            for term in terms or []:
                if term_occurs(term, text):
                    self.db.execute(
                        "INSERT INTO term_usage(term, count, last_used) VALUES (?, 1, ?) "
                        "ON CONFLICT(term) DO UPDATE SET count = count + 1, last_used = excluded.last_used",
                        (term, now),
                    )
            self.db.commit()

    def remember_correction(self, wrong: str, right: str, context: str, safe: bool = False) -> None:
        wrong, right, context = wrong.strip(), right.strip(), context.strip()
        if not wrong or not right:
            return
        now = time.time()
        with self.lock:
            self._delete_corrections("wrong = ? COLLATE NOCASE", (wrong,))
            cursor = self.db.execute(
                "INSERT INTO corrections(created_at, wrong, right, context, safe, embedding) VALUES (?, ?, ?, ?, ?, ?)",
                (now, wrong, right, context or right, int(safe), self._embed(context or right)),
            )
            if self.fts:
                self.db.execute(
                    "INSERT INTO corrections_fts(rowid, text) VALUES (?, ?)",
                    (cursor.lastrowid, fold((context or right) + " " + right)),
                )
            self.db.execute(
                "INSERT INTO term_usage(term, count, last_used) VALUES (?, 1, ?) "
                "ON CONFLICT(term) DO UPDATE SET count = count + 1, last_used = excluded.last_used",
                (right, now),
            )
            self.db.commit()

    def _delete_corrections(self, where: str, parameters: tuple) -> None:
        ids = [row["id"] for row in self.db.execute(f"SELECT id FROM corrections WHERE {where}", parameters)]
        for row_id in ids:
            self.db.execute("DELETE FROM corrections WHERE id = ?", (row_id,))
            if self.fts:
                self.db.execute("DELETE FROM corrections_fts WHERE rowid = ?", (row_id,))

    def set_vocabulary(self, terms: list[str], source: str = "manual") -> None:
        """Replace the user's hand-written list (idempotent; order is kept as priority)."""
        now = time.time()
        with self.lock:
            self.db.execute("DELETE FROM vocabulary WHERE source = ?", (source,))
            for position, term in enumerate(dict.fromkeys(t.strip() for t in terms if t.strip())):
                self.db.execute(
                    "INSERT OR REPLACE INTO vocabulary(term, source, position, added_at) VALUES (?, ?, ?, ?)",
                    (term, source, position, now),
                )
            self.db.commit()

    def set_contexts(self, texts: list[str]) -> None:
        """Replace the user's context sentences; embeddings of unchanged ones are kept."""
        now = time.time()
        wanted = list(dict.fromkeys(t.strip() for t in texts if t.strip()))
        with self.lock:
            existing = {row["text"] for row in self.db.execute("SELECT text FROM contexts")}
            for text in existing - set(wanted):
                self.db.execute("DELETE FROM contexts WHERE text = ?", (text,))
            for position, text in enumerate(wanted):
                if text in existing:
                    self.db.execute("UPDATE contexts SET position = ? WHERE text = ?", (position, text))
                else:
                    self.db.execute(
                        "INSERT INTO contexts(text, position, added_at, embedding) VALUES (?, ?, ?, ?)",
                        (text, position, now, self._embed(text)),
                    )
            self.db.commit()

    def vocabulary(self, source: str = "manual") -> list[str]:
        with self.lock:
            return [row["term"] for row in self.db.execute(
                "SELECT term FROM vocabulary WHERE source = ? ORDER BY position", (source,))]

    def contexts(self) -> list[str]:
        with self.lock:
            return [row["text"] for row in self.db.execute("SELECT text FROM contexts ORDER BY position")]

    def learned_pairs(self) -> list[tuple[str, str]]:
        """Every correction the user taught, for the smart-correction gate."""
        with self.lock:
            return [(row["wrong"], row["right"]) for row in self.db.execute("SELECT wrong, right FROM corrections")]

    def forget_correction(self, wrong: str, right: str) -> None:
        with self.lock:
            self._delete_corrections("wrong = ? COLLATE NOCASE AND right = ?", (wrong, right))
            self.db.commit()

    def clear(self) -> None:
        with self.lock:
            self.db.execute("DELETE FROM dictations")
            self.db.execute("DELETE FROM corrections")
            self.db.execute("DELETE FROM term_usage")
            if self.fts:
                self.db.execute("DELETE FROM dictations_fts")
                self.db.execute("DELETE FROM corrections_fts")
            self.db.commit()

    # MARK: reading

    def stats(self) -> dict[str, Any]:
        with self.lock:
            dictations = self.db.execute("SELECT COUNT(*) FROM dictations").fetchone()[0]
            corrections = self.db.execute("SELECT COUNT(*) FROM corrections").fetchone()[0]
            embedded = self.db.execute(
                "SELECT (SELECT COUNT(*) FROM dictations WHERE embedding IS NOT NULL) + "
                "(SELECT COUNT(*) FROM corrections WHERE embedding IS NOT NULL)"
            ).fetchone()[0]
        with self.lock:
            manual = self.db.execute("SELECT COUNT(*) FROM vocabulary WHERE source = 'manual'").fetchone()[0]
            contexts = self.db.execute("SELECT COUNT(*) FROM contexts").fetchone()[0]
        return {
            "dictations": dictations,
            "corrections": corrections,
            "manual_terms": manual,
            "contexts": contexts,
            "embedded": embedded,
            "embedder": self.embedder_state,
            "detail": self.embedder_detail,
            "fts": self.fts,
        }

    def recent_dictations(self, limit: int = RECENT_DICTATIONS_AS_QUERY) -> list[str]:
        with self.lock:
            rows = self.db.execute("SELECT text FROM dictations ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [row["text"] for row in rows]

    def _rows_mentioning(self, term: str, limit: int = 20) -> list[sqlite3.Row]:
        """Dictations and corrections whose text contains the term."""
        rows: list[sqlite3.Row] = []
        with self.lock:
            for table, column in (("dictations", "text"), ("corrections", "context")):
                candidates = self.db.execute(
                    f"SELECT {column} AS text, embedding, created_at FROM {table} "
                    f"WHERE {column} LIKE ? ORDER BY id DESC LIMIT ?",
                    (f"%{term}%", limit),
                ).fetchall()
                rows.extend(row for row in candidates if term_occurs(term, row["text"]))
        return rows

    MANUAL_BONUS = 1.5

    def rank_terms(self, terms: list[tuple[str, str]], query: str | None = None) -> list[tuple[str, float]]:
        """Terms scored by usage, recency and relevance to the query.

        `terms` are (term, source) pairs; hand-written terms carry a bonus so
        that, at equal relevance, what the user typed goes in before what was
        learned. Relevance is the similarity between the query (the latest
        dictations) and the sentences the term was seen in.
        """
        now = time.time()
        query_vector = self._query_vector(query) if query else None
        query_words = set(fold(query or "").split())
        scored: list[tuple[str, float]] = []
        with self.lock:
            usage = {row["term"]: (row["count"], row["last_used"]) for row in self.db.execute("SELECT * FROM term_usage")}
        total = max(1, len(terms))
        for index, (term, source) in enumerate(terms):
            count, last_used = usage.get(term, (0, 0.0))
            days = (now - last_used) / 86400 if last_used else 365.0
            score = math.log1p(count) + max(0.0, 1.0 - days / 30.0) + max(0.0, 0.5 - index / total * 0.5)
            if source == "manual":
                score += self.MANUAL_BONUS
            relevance = 0.0
            for row in self._rows_mentioning(term):
                if query_vector is not None and row["embedding"] is not None:
                    relevance = max(relevance, _dot(query_vector, _unpack(row["embedding"])))
                elif query_words:
                    overlap = len(query_words & set(fold(row["text"]).split()))
                    relevance = max(relevance, min(1.0, overlap / 4.0))
            scored.append((term, score + 2.0 * relevance))
        scored.sort(key=lambda item: -item[1])
        return scored

    def rank_contexts(self, contexts: list[str], query: str | None = None) -> list[str]:
        """Context sentences, most relevant to the latest dictations first;
        without a signal, in the user's order."""
        if len(contexts) <= 1:
            return list(contexts)
        query_vector = self._query_vector(query) if query else None
        query_words = set(fold(query or "").split())
        with self.lock:
            stored = {row["text"]: row["embedding"] for row in self.db.execute("SELECT text, embedding FROM contexts")}
        scored = []
        for position, text in enumerate(contexts):
            relevance = 0.0
            blob = stored.get(text)
            if query_vector is not None and blob is not None:
                relevance = _dot(query_vector, _unpack(blob))
            elif query_words:
                relevance = min(1.0, len(query_words & set(fold(text).split())) / 4.0)
            scored.append((relevance - position * 0.01, text))
        scored.sort(key=lambda item: -item[0])
        return [text for _, text in scored]

    def compose_prompt(self, contexts: list[str] | str, terms: list[str], learned: list[str],
                       count_tokens: Callable[[str], int], budget: int = PROMPT_TOKEN_BUDGET) -> tuple[str, list[str]]:
        """Context sentences and terms chosen for this dictation within the budget.

        Order of filling: the most relevant context sentence first, then terms
        by rank (hand-written ones carry a bonus), then further context
        sentences if room is left. Returns the prompt and the terms it holds.
        """
        if isinstance(contexts, str):
            contexts = [contexts] if contexts.strip() else []
        query = " ".join(self.recent_dictations())
        ordered_contexts = self.rank_contexts([c.strip() for c in contexts if c.strip()], query)
        pairs = [(term, "manual") for term in dict.fromkeys(terms) if term.strip()]
        seen = {term for term, _ in pairs}
        pairs += [(term, "learned") for term in dict.fromkeys(learned) if term.strip() and term not in seen]
        ranked = [term for term, _ in self.rank_terms(pairs, query)]

        def build(context_parts: list[str], selected: list[str]) -> str:
            text = " ".join(context_parts)
            if not selected:
                return text
            suffix = "Özel adlar ve teknik terimler aynen şu yazımla kullanılır: " + ", ".join(selected) + "."
            return suffix if not text else text + " " + suffix

        chosen_contexts = ordered_contexts[:1]
        if chosen_contexts and count_tokens(build(chosen_contexts, [])) > budget:
            chosen_contexts = []
        chosen: list[str] = []
        for term in ranked:
            if count_tokens(build(chosen_contexts, chosen + [term])) <= budget:
                chosen.append(term)
        for extra in ordered_contexts[1:]:
            if count_tokens(build(chosen_contexts + [extra], chosen)) <= budget:
                chosen_contexts.append(extra)
        return build(chosen_contexts, chosen), chosen

    VECTOR_WEIGHT = 0.7
    KEYWORD_WEIGHT = 0.3

    def _keyword_scores(self, text: str, limit: int = 50) -> dict[int, float]:
        """BM25 ranks from full-text search, mapped to 1.0 … 0.0 by rank."""
        if not self.fts:
            return {}
        words = [word for word in set(re.findall(r"\w+", fold(text))) if len(word) > 3]
        if not words:
            return {}
        match = " OR ".join(f'"{word}"' for word in words[:40])
        with self.lock:
            try:
                hits = self.db.execute(
                    "SELECT rowid FROM corrections_fts WHERE corrections_fts MATCH ? ORDER BY bm25(corrections_fts) LIMIT ?",
                    (match, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                return {}
        return {row[0]: 1.0 - index / max(1, len(hits)) for index, row in enumerate(hits)}

    def nearest_corrections(self, text: str, limit: int = MAX_EXAMPLES) -> list[dict[str, str]]:
        """Past corrections whose sentences are most like this transcript.

        Hybrid, the way OpenClaw's memory search blends its results: vector
        similarity weighted 0.7 and keyword (BM25) rank weighted 0.3. Without
        embeddings it is keyword-only; without keywords, recency.
        """
        query_vector = self._query_vector(text)
        with self.lock:
            rows = self.db.execute("SELECT wrong, right, context, embedding, id FROM corrections ORDER BY id DESC LIMIT 2000").fetchall()
        if not rows:
            return []
        keyword = self._keyword_scores(text)
        if query_vector is not None and any(row["embedding"] is not None for row in rows):
            scored = []
            for row in rows:
                vector = _dot(query_vector, _unpack(row["embedding"])) if row["embedding"] is not None else 0.0
                scored.append((self.VECTOR_WEIGHT * vector + self.KEYWORD_WEIGHT * keyword.get(row["id"], 0.0), row))
            scored.sort(key=lambda item: -item[0])
            picked = [row for _, row in scored[:limit]]
        elif self.fts:
            words = [word for word in set(re.findall(r"\w+", fold(text))) if len(word) > 3]
            picked = []
            if words:
                match = " OR ".join(f'"{word}"' for word in words[:40])
                with self.lock:
                    try:
                        hits = self.db.execute(
                            "SELECT c.wrong, c.right, c.context FROM corrections_fts f JOIN corrections c ON c.id = f.rowid "
                            "WHERE corrections_fts MATCH ? ORDER BY bm25(corrections_fts) LIMIT ?",
                            (match, limit),
                        ).fetchall()
                    except sqlite3.OperationalError:
                        hits = []
                picked = list(hits)
            if not picked:
                picked = rows[:limit]
        else:
            picked = rows[:limit]
        return [{"text": row["context"], "wrong": row["wrong"], "right": row["right"]} for row in picked]
