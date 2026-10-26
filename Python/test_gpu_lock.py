#!/usr/bin/env python3
"""Every call that reaches the GPU must sit inside `with gpu_lock.LOCK:`.

PyTorch's MPS backend aborts the whole worker when two threads encode Metal
work at the same time (the "_status < MTLCommandBufferStatusCommitted"
assertion, see gpu_lock.py). The lock only protects while every GPU call
takes it, and the next person to add a `.to(device)` or a `generate()` on a
background thread would reintroduce the crash without noticing. So this
test reads the worker modules and fails for any GPU call that is not
lexically inside a `with gpu_lock.LOCK` block.

Run from the Python/ directory:  python3 -B -m unittest test_gpu_lock
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULES = ["whisper_worker.py", "term_llm.py", "memory.py", "term_bias.py"]

# Method calls that encode Metal work: moving tensors or modules between
# devices, decoding, and llama.cpp embeddings.
GPU_METHODS = {"to", "generate", "embed", "cuda", "mps"}
# Constructors and plain calls that put weights on the GPU or run it: llama.cpp's
# Llama(), mlx-lm's load() and generate().
GPU_CONSTRUCTORS = {"Llama", "load", "generate"}

# Functions that are only ever called by code already holding the lock:
# the biasing logits processor runs inside Whisper's generate().
EXEMPT_FUNCTIONS = {("term_bias.py", "__call__")}


def lock_names(node: ast.With) -> set[str]:
    names = set()
    for item in node.items:
        expr = item.context_expr
        if isinstance(expr, ast.Attribute) and expr.attr == "LOCK":
            names.add("LOCK")
        elif isinstance(expr, ast.Name) and expr.id == "LOCK":
            names.add("LOCK")
    return names


def is_gpu_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr in GPU_METHODS:
        return True
    if isinstance(func, ast.Name) and func.id in GPU_CONSTRUCTORS:
        return True
    return any(keyword.arg == "device" for keyword in node.keywords)


def unlocked_gpu_calls(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders: list[str] = []

    def visit(node: ast.AST, locked: bool, function: str | None) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function = node.name
        if isinstance(node, ast.With) and "LOCK" in lock_names(node):
            locked = True
        if isinstance(node, ast.Call) and is_gpu_call(node) and not locked:
            if (path.name, function) not in EXEMPT_FUNCTIONS:
                offenders.append(f"{path.name}:{node.lineno} in {function or '<module>'}: {ast.unparse(node)[:80]}")
        for child in ast.iter_child_nodes(node):
            visit(child, locked, function)

    visit(tree, False, None)
    return offenders


class GPULockCoverageTests(unittest.TestCase):
    def test_every_gpu_call_holds_the_lock(self) -> None:
        offenders: list[str] = []
        for name in MODULES:
            offenders += unlocked_gpu_calls(HERE / name)
        self.assertEqual(offenders, [], "GPU calls outside `with gpu_lock.LOCK`:\n" + "\n".join(offenders))

    def test_the_scan_sees_gpu_calls(self) -> None:
        """The scan must find the real calls, or an empty result proves nothing."""
        found = 0
        for name in MODULES:
            tree = ast.parse((HERE / name).read_text(encoding="utf-8"))
            found += sum(1 for node in ast.walk(tree) if isinstance(node, ast.Call) and is_gpu_call(node))
        self.assertGreaterEqual(found, 8, f"only {found} GPU calls found; is the detector broken?")

    def test_an_unlocked_call_is_reported(self) -> None:
        sample = HERE / "_gpu_lock_sample.py"
        sample.write_text(
            "import gpu_lock\n"
            "def load(model, device):\n"
            "    with gpu_lock.LOCK:\n"
            "        model.to(device)\n"
            "def bad(model, device):\n"
            "    return model.to(device)\n",
            encoding="utf-8",
        )
        try:
            offenders = unlocked_gpu_calls(sample)
        finally:
            sample.unlink()
        self.assertEqual(len(offenders), 1, offenders)
        self.assertIn("in bad", offenders[0])


if __name__ == "__main__":
    unittest.main()
