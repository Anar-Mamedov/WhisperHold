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


class Embedder:
    """multilingual-e5-small on the GPU: `query:` for what we search with,
    `passage:` for what is stored, mean pooling, unit length."""

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
        self.model = AutoModel.from_pretrained(snapshot, local_files_only=True, dtype=dtype).eval().to(self.device)
        self.torch = torch

    def encode(self, texts: list[str], kind: str = "passage") -> list[list[float]]:
        if not texts:
            return []
        torch = self.torch
        batch = self.tokenizer(
            [f"{kind}: {text}" for text in texts], padding=True, truncation=True, max_length=256, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
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
    return sum(x * y for x, y in zip(a, b))


class Memory:
    def __init__(self, path: Path, emit: Callable[[dict[str, Any]], None] | None = None,
                 embedder_factory: Callable[[], Any] | None = None) -> None:
        self.path = path
        self.emit = emit or (lambda message: None)
        self.embedder_factory = embedder_factory or Embedder
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
            self.report("ready", EMBEDDING_MODEL_ID)
            self.backfill()
        except Exception as error:  # noqa: BLE001
            self.report("failed", f"Embedding modeli yüklenemedi: {error}")

    def backfill(self, limit: int = 500) -> None:
        """Rows stored before the embedder was ready get their vectors now."""
        with self.lock:
            if self.embedder is None:
                return
            for table, column in (("dictations", "text"), ("corrections", "context")):
                rows = self.db.execute(
                    f"SELECT id, {column} AS text FROM {table} WHERE embedding IS NULL ORDER BY id DESC LIMIT ?", (limit,)
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
        return {
            "dictations": dictations,
            "corrections": corrections,
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

    def rank_learned_terms(self, learned: list[str], query: str | None = None) -> list[tuple[str, float]]:
        """Learned terms scored by usage, recency and relevance to the query."""
        now = time.time()
        query_vector = self._query_vector(query) if query else None
        query_words = set(fold(query or "").split())
        scored: list[tuple[str, float]] = []
        with self.lock:
            usage = {row["term"]: (row["count"], row["last_used"]) for row in self.db.execute("SELECT * FROM term_usage")}
        for index, term in enumerate(learned):
            count, last_used = usage.get(term, (0, 0.0))
            days = (now - last_used) / 86400 if last_used else 365.0
            score = math.log1p(count) + max(0.0, 1.0 - days / 30.0) + max(0.0, 0.5 - index / len(learned) * 0.5)
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

    def compose_prompt(self, context: str, terms: list[str], learned: list[str],
                       count_tokens: Callable[[str], int], budget: int = PROMPT_TOKEN_BUDGET) -> tuple[str, list[str]]:
        """The context sentence, every hand-written term, then learned terms by
        rank until the token budget is used. Returns the prompt and the learned
        terms that made it in."""
        context = context.strip()

        def build(selected: list[str]) -> str:
            all_terms = terms + selected
            if not all_terms:
                return context
            suffix = "Özel adlar ve teknik terimler aynen şu yazımla kullanılır: " + ", ".join(all_terms) + "."
            return suffix if not context else context + " " + suffix

        ranked = [term for term, _ in self.rank_learned_terms(learned, " ".join(self.recent_dictations()))]
        chosen: list[str] = []
        for term in ranked:
            candidate = chosen + [term]
            if count_tokens(build(candidate)) > budget:
                continue
            chosen = candidate
        return build(chosen), chosen

    def nearest_corrections(self, text: str, limit: int = MAX_EXAMPLES) -> list[dict[str, str]]:
        """Past corrections whose sentences are most like this transcript."""
        query_vector = self._query_vector(text)
        with self.lock:
            rows = self.db.execute("SELECT wrong, right, context, embedding, id FROM corrections ORDER BY id DESC LIMIT 2000").fetchall()
        if not rows:
            return []
        if query_vector is not None and any(row["embedding"] is not None for row in rows):
            scored = [
                (_dot(query_vector, _unpack(row["embedding"])) if row["embedding"] is not None else -1.0, row)
                for row in rows
            ]
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
