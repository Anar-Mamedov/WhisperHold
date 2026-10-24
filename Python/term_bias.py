"""Contextual biasing: nudge Whisper toward a vocabulary while it decodes.

The decoder picks the next token from the model's scores; this logits
processor adds a constant to the score of every token that starts a listed
term, and to the token that continues a term already under way. It is the
only way to influence a word at the moment the model chooses it — once
"backendci" has come out as the real Turkish word "bekleyen", no dictionary
can tell the difference afterwards.

Measured on the archived recordings, ten synthesized sentences with software
terms and twelve genuine Turkish sentences full of look-alike words:

- Strength 2.0 was the smallest that turned "bekleyen" back into "backendci"
  on the real recording. It also lowered the word error rate on the term
  sentences (9.1 % → 6.5 %) and changed one of the twelve look-alike
  sentences (the rare "rende" became "render"). At 3.0 four of twelve
  changed; 1.5 did not fix the recording.
- On a recording longer than one 30 s window the same bias was destructive
  at every strength tried: at 1.0 whole clauses were replaced, at 1.5 the
  second window collapsed into a repetition loop, at 2.0 the prompt's own
  words leaked into the text. So biasing is applied to single-window audio
  only; long recordings keep the plain decoder.

Terms are matched as token sequences in a trie, each in two spellings: as
listed (" backend") and capitalized for sentence starts (" Backend").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

BIAS_STRENGTH = 2.0
VOCABULARY_FILE = Path(__file__).with_name("software_terms.txt")

_END = -1


def load_vocabulary(path: Path = VOCABULARY_FILE) -> list[str]:
    """Terms from the bundled list, in file order, without comments or duplicates."""
    terms: list[str] = []
    seen: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").split("\n")
    except OSError:
        return terms
    for raw in lines:
        line = raw.strip()
        if line and not line.startswith("#") and line not in seen:
            seen.add(line)
            terms.append(line)
    return terms


def spellings(term: str) -> list[str]:
    """The listed spelling and, for lowercase terms, the sentence-initial one."""
    out = [" " + term]
    if term[:1].islower():
        out.append(" " + term[0].upper() + term[1:])
    return out


class TermTrie:
    """Token-id trie over every spelling of every term."""

    def __init__(self, tokenizer: Any, terms: list[str]):
        self.root: dict[int, Any] = {}
        self.max_depth = 1
        for term in terms:
            for text in spellings(term):
                ids = tokenizer.encode(text, add_special_tokens=False)
                if not ids:
                    continue
                node = self.root
                for token in ids:
                    node = node.setdefault(token, {})
                node[_END] = True
                self.max_depth = max(self.max_depth, len(ids))
        self.start_tokens = sorted(token for token in self.root if token != _END)

    def continuations(self, recent: list[int]) -> set[int]:
        """Tokens that continue a term whose prefix ends the recent tokens."""
        out: set[int] = set()
        for start in range(len(recent)):
            node = self.root
            for token in recent[start:]:
                node = node.get(token)
                if node is None:
                    break
            else:
                if node is not self.root:
                    out.update(child for child in node if child != _END)
        return out


def make_bias_processor(torch: Any, tokenizer: Any, terms: list[str], strength: float = BIAS_STRENGTH) -> Any:
    from transformers import LogitsProcessor

    trie = TermTrie(tokenizer, terms)

    class TermBiasProcessor(LogitsProcessor):
        def __call__(self, input_ids: Any, scores: Any) -> Any:
            if strength == 0 or not trie.start_tokens:
                return scores
            starts = torch.tensor(trie.start_tokens, device=scores.device)
            recent_rows = input_ids[:, -trie.max_depth:].tolist()
            for row, recent in enumerate(recent_rows):
                scores[row, starts] += strength
                more = trie.continuations(recent)
                if more:
                    scores[row, torch.tensor(sorted(more), device=scores.device)] += strength
            return scores

    return TermBiasProcessor()
