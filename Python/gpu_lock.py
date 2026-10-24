"""One lock for every use of the GPU from this process.

PyTorch's MPS backend drives a single Metal command stream and is not safe
to use from two Python threads at once: when the smart-correction model was
being moved to the GPU on a background thread while Whisper decoded a
dictation on the main thread, Metal aborted the process with
"failed assertion _status < MTLCommandBufferStatusCommitted" (measured on a
real dictation, the first one after the worker had restarted). Every piece
of code that runs tensors on the GPU — decoding, moving a model to the
device, embedding — takes this lock first. It is reentrant, so a locked
caller may call locked helpers.
"""

from __future__ import annotations

import threading

LOCK = threading.RLock()
