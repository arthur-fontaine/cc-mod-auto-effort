"""Score the effort task with a fine-tuned model: one forward pass, no sampling.

The answer is read from the logits of the six option letters at the first generated
position, so every call returns a full probability distribution over task categories. The tokens
shared by every request (system prompt, question, options) are run once and their cache
is reused, so a request only pays for its own state.
"""
import copy
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import CATEGORIES  # noqa: E402
from task import LETTERS, messages  # noqa: E402


class Classifier:
    def __init__(self, model_path, adapter_path=None):
        # Every prompt length leaves differently sized buffers in MLX's cache; unbounded,
        # they grow until macOS pages, which shows up as multi-second outliers.
        mx.set_cache_limit(512 * 1024**2)
        self.model, self.tokenizer = load(str(model_path), adapter_path=str(adapter_path) if adapter_path else None)
        self.model.eval()
        self.letter_ids = [self.tokenizer.encode(c, add_special_tokens=False)[0] for c in LETTERS]
        # Written by pipeline/calibrate.py next to the model: a temperature fitted on dev,
        # since fine-tuning leaves the letter probabilities overconfident.
        calibration = Path(model_path) / "calibration.json"
        self.temperature = json.loads(calibration.read_text())["temperature"] if calibration.exists() else 1.0
        self._prefix = None
        self._prefix_cache = None

    def tokens(self, prompt, previous_reply=None):
        ids = self.tokenizer.apply_chat_template(
            messages(prompt=prompt, previous_reply=previous_reply),
            add_generation_prompt=True, enable_thinking=False,
        )
        return list(ids["input_ids"] if hasattr(ids, "input_ids") else ids)

    def _cached_prefix(self, ids):
        """Return (cache, number of ids it covers) for the shared prompt prefix."""
        if self._prefix is None:
            # The prefix is everything up to the state's first value.
            probe = self.tokens("a")
            other = self.tokens("b")
            n = 0
            while n < min(len(probe), len(other)) and probe[n] == other[n]:
                n += 1
            # Back off a few tokens so a tokenizer merge at the boundary can't differ.
            self._prefix = probe[: max(0, n - 4)]
            cache = make_prompt_cache(self.model)
            self.model(mx.array(self._prefix)[None], cache=cache)
            mx.eval([c.state for c in cache])
            self._prefix_cache = cache
        if ids[: len(self._prefix)] != self._prefix:
            return make_prompt_cache(self.model), 0
        return copy.deepcopy(self._prefix_cache), len(self._prefix)

    def letter_logits(self, prompt, previous_reply=None, use_cache=True):
        ids = self.tokens(prompt, previous_reply)
        if use_cache:
            cache, skip = self._cached_prefix(ids)
        else:
            cache, skip = make_prompt_cache(self.model), 0
        logits = self.model(mx.array(ids[skip:])[None], cache=cache)[0, -1]
        return logits[mx.array(self.letter_ids)].astype(mx.float32), len(ids)

    def predict(self, prompt, previous_reply=None, use_cache=True):
        started = time.perf_counter()
        logits, n = self.letter_logits(prompt, previous_reply, use_cache)
        probs = mx.softmax(logits / self.temperature).tolist()
        return {
            "probabilities": dict(zip(CATEGORIES, probs)),
            "choice": CATEGORIES[max(range(len(probs)), key=probs.__getitem__)],
            "tokens": n,
            "ms": (time.perf_counter() - started) * 1000,
        }
