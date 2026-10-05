"""Task categories, and the effort level each Claude model should use for each one.

The classifier predicts a category, which describes the request and does not depend on the
model. The model comes in here: each model has its own default effort, the level Anthropic
recommends for most tasks on it, and supports its own set of levels
(https://platform.claude.com/docs/en/build-with-claude/effort and the model overview
pages, checked 2026-10-02). Levels are recalibrated per model, so the table is per model.

The tables are inferred from that guidance ("the default is right for most tasks; raise it
when skipping a file or a test run would make the result wrong"), not measured. Edit them
here; the classifier does not need retraining.
"""
LEVELS = ["low", "medium", "high", "xhigh", "max"]

CATEGORIES = ["trivial", "light", "ordinary", "multi_step", "hard", "exhaustive"]

CATEGORY_DESCRIPTIONS = {
    "trivial": "Routine work that needs no investigation: a precisely described edit, a rename, a typo, a "
               "one-line change, a question about code already in context, a quick lookup or shell command, "
               "an acknowledgement or small talk.",
    "light": "A small, well-scoped change or a focused answer that needs a little reading, but no "
             "multi-file investigation.",
    "ordinary": "A typical coding request: an ordinary feature, a normal bug fix, a focused explanation, or "
                "anything unclear.",
    "multi_step": "Multi-step work that must be verified: changes across several files, a bug that needs "
                  "several hypotheses checked, a refactor that has to be finished completely, work where "
                  "skipping a file or not running the tests would make the result wrong.",
    "hard": "Long, hard, high-stakes work: a large migration or refactor, a subtle bug across systems, an "
            "architecture decision, a security or correctness audit, or an explicit request to be thorough "
            "and verify everything.",
    "exhaustive": "Exhaustive work where cost does not matter: the request explicitly demands maximum effort "
                  "or leaving nothing unchecked, on a critical, very large task.",
}

# Default high: ordinary and multi-step work both run at the default.
_HIGH = {"trivial": "low", "light": "medium", "ordinary": "high", "multi_step": "high",
         "hard": "xhigh", "exhaustive": "max"}
# Default medium (Opus 5.5): the default already covers ordinary work, so verified
# multi-step work is where it pays to raise effort.
_MEDIUM = {"trivial": "low", "light": "medium", "ordinary": "medium", "multi_step": "high",
           "hard": "xhigh", "exhaustive": "max"}
# No xhigh: hard work stays at high rather than jumping to max, which is unbounded.
_NO_XHIGH = {**_HIGH, "hard": "high"}
_NO_MAX = {**_HIGH, "hard": "high", "exhaustive": "high"}

PROFILES = {
    "claude-opus-5-5": {"default": "medium", "supported": LEVELS, "table": _MEDIUM},
    "claude-sonnet-5-5": {"default": "high", "supported": LEVELS, "table": _HIGH},
    "claude-fable-5-1": {"default": "high", "supported": LEVELS, "table": _HIGH},
    "claude-opus-5": {"default": "high", "supported": LEVELS, "table": _HIGH},
    "claude-sonnet-5": {"default": "high", "supported": LEVELS, "table": _HIGH},
    "claude-fable-5": {"default": "high", "supported": LEVELS, "table": _HIGH},
    "claude-opus-4-8": {"default": "high", "supported": LEVELS, "table": _HIGH},
    "claude-opus-4-7": {"default": "high", "supported": LEVELS, "table": _HIGH},
    "claude-opus-4-6": {"default": "high", "supported": ["low", "medium", "high", "max"], "table": _NO_XHIGH},
    "claude-sonnet-4-6": {"default": "high", "supported": ["low", "medium", "high", "max"], "table": _NO_XHIGH},
    "claude-opus-4-5": {"default": "high", "supported": ["low", "medium", "high"], "table": _NO_MAX},
}
# Unknown or newer models: the common profile.
FALLBACK = {"default": "high", "supported": LEVELS, "table": _HIGH}

ALIASES = {"opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5-5", "fable": "claude-fable-5-1"}


def normalize(model):
    """`claude-opus-5-5[1m]`, `us.anthropic.claude-opus-5-5-v1`, `opus` -> `claude-opus-5-5`."""
    import re

    if not model:
        return None
    m = model.lower().split("/")[-1]
    m = re.sub(r"^((us|eu|apac|global)\.)?anthropic\.", "", m)
    m = re.sub(r"\[.*?\]$|:.*$|-v\d+$|-\d{8}$|-fast$", "", m).replace(".", "-")
    return ALIASES.get(m, m)


def profile(model):
    return PROFILES.get(normalize(model) or "", FALLBACK)


def supports_effort(model):
    """False only for models known to take no effort (Haiku 4.5). An unknown or missing
    model gets the common profile rather than being left alone."""
    return "haiku" not in (normalize(model) or "")


def level_for(category, model):
    return profile(model)["table"][category]


def level_probabilities(category_probs, model):
    """Sum category probabilities into level probabilities for `model`."""
    table = profile(model)["table"]
    out = {lv: 0.0 for lv in LEVELS}
    for cat, p in category_probs.items():
        out[table[cat]] += p
    return out
