import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT))

from build import assert_no_leak, example  # noqa: E402
from clean import scrub, template  # noqa: E402
from extract import clean_prompt, effort_command, normalize_model  # noqa: E402
from models import CATEGORIES, FALLBACK, LEVELS, PROFILES, level_probabilities, normalize, profile, supports_effort  # noqa: E402
from task import LETTERS, messages  # noqa: E402


class Tables(unittest.TestCase):
    def test_every_category_maps_to_a_supported_level(self):
        for name, p in {**PROFILES, "fallback": FALLBACK}.items():
            for c in CATEGORIES:
                self.assertIn(p["table"][c], p["supported"], f"{name}/{c}")

    def test_levels_never_go_down_as_work_grows(self):
        for name, p in PROFILES.items():
            levels = [LEVELS.index(p["table"][c]) for c in CATEGORIES]
            self.assertEqual(levels, sorted(levels), name)

    def test_ordinary_work_runs_at_the_default(self):
        for name, p in PROFILES.items():
            self.assertEqual(p["table"]["ordinary"], p["default"], name)

    def test_opus_5_5_raises_effort_for_multi_step_work(self):
        self.assertEqual(profile("claude-opus-5-5")["table"]["multi_step"], "high")
        self.assertEqual(profile("claude-sonnet-5-5")["table"]["multi_step"], "high")
        self.assertEqual(profile("claude-opus-5-5")["default"], "medium")

    def test_probabilities_are_summed_through_the_table(self):
        probs = dict(zip(CATEGORIES, [0.1, 0.1, 0.4, 0.3, 0.1, 0.0]))
        sonnet = level_probabilities(probs, "claude-sonnet-5-5")
        self.assertAlmostEqual(sonnet["high"], 0.7)
        opus = level_probabilities(probs, "claude-opus-5-5")
        self.assertAlmostEqual(opus["medium"], 0.5)
        self.assertAlmostEqual(sum(opus.values()), 1.0)

    def test_only_haiku_is_left_alone(self):
        self.assertFalse(supports_effort("claude-haiku-4-5"))
        self.assertTrue(supports_effort("claude-opus-5-5[1m]"))
        # A missing or unknown model gets the common profile, not "no effort".
        self.assertTrue(supports_effort(None))
        self.assertEqual(profile(None)["default"], "high")

    def test_model_names_normalize(self):
        self.assertEqual(normalize("claude-opus-5-5[1m]"), "claude-opus-5-5")
        self.assertEqual(normalize("us.anthropic.claude-opus-5-5-v1"), "claude-opus-5-5")
        self.assertEqual(normalize("opus"), "claude-opus-5-5")
        self.assertEqual(normalize_model("bedrock/us.anthropic.claude-opus-4-6-v1"), "claude-opus-4-6")
        self.assertEqual(normalize_model("claude-4.5-sonnet"), "claude-sonnet-4-5")
        self.assertIsNone(normalize_model("gpt-5.5"))
        self.assertIsNone(normalize_model("<synthetic>"))


class ModContract(unittest.TestCase):
    """The mod (hooks/policy.js) must ask the local model what it was trained on."""

    @classmethod
    def setUpClass(cls):
        import subprocess

        samples = ["short", "x" * 5000, "\U0001F600" * 3100 + "end", "naïve " * 900]
        code = (
            "import('./hooks/policy.js').then(m => console.log(JSON.stringify({"
            "question: m.CATEGORY_QUESTION, categories: m.CATEGORIES,"
            "profiles: Object.fromEntries(Object.entries(m.PROFILES).map(([k, v]) => [k, v])),"
            f"clips: {json.dumps(samples)}.map(t => [m.clip(t, 3000), m.clip(t, 800)]),"
            "})))"
        )
        out = subprocess.run(["node", "-e", code], cwd=ROOT.parent, capture_output=True, text=True, check=True)
        cls.js, cls.samples = json.loads(out.stdout), samples

    def test_category_question_matches_training(self):
        from task import OPTIONS, QUESTION

        self.assertEqual(self.js["categories"], CATEGORIES)
        self.assertEqual(self.js["question"]["instructions"], QUESTION)
        self.assertEqual(self.js["question"]["criteria"], OPTIONS)
        self.assertEqual(list(self.js["question"]["criteria"]), CATEGORIES)

    def test_tables_match(self):
        for name, p in PROFILES.items():
            self.assertEqual(self.js["profiles"][name]["default"], p["default"], name)
            self.assertEqual(self.js["profiles"][name]["table"], p["table"], name)
        self.assertEqual(set(self.js["profiles"]), set(PROFILES))

    def test_clip_matches(self):
        from task import MAX_CONTEXT_CHARS, MAX_PROMPT_CHARS, clip

        for text, (prompt, context) in zip(self.samples, self.js["clips"]):
            self.assertEqual(prompt, clip(text, MAX_PROMPT_CHARS))
            self.assertEqual(context, clip(text, MAX_CONTEXT_CHARS))


class Data(unittest.TestCase):
    def test_prompt_cleaning(self):
        self.assertEqual(clean_prompt("<timestamp>Mon</timestamp>\n<user_query>\nfix it\n</user_query>"), "fix it")
        self.assertIsNone(clean_prompt("<local-command-stdout>Compacted</local-command-stdout>"))
        self.assertIsNone(clean_prompt("[Request interrupted by user]"))
        self.assertEqual(effort_command("<command-name>/effort</command-name><command-args>xhigh</command-args>"), "xhigh")

    def test_secrets_are_scrubbed(self):
        text = "key sk-ant-api03-abcdefghijklmnopqrstuvwxyz and ghp_abcdefghijklmnopqrstuvwxyz0123 ok"
        self.assertNotIn("sk-ant", scrub(text))
        self.assertNotIn("ghp_", scrub(text))
        self.assertIn("ok", scrub(text))

    def test_templates_ignore_paths_and_numbers(self):
        self.assertEqual(template("Reproduce paper 12 in ~/papers/abc"), template("Reproduce paper 7 in ~/papers/xyz"))

    def test_examples_carry_no_hindsight(self):
        r = {"id": "x", "model": "claude-opus-5-5", "source": "s", "prompt": "fix the bug",
             "previous_reply": "Want me to?", "final_reply": "Z" * 200, "next_prompt": "thanks " * 30}
        ex = example(r, "ordinary")
        assert_no_leak(ex, r)
        state = json.loads(ex["prompt"][1]["content"])["state"]
        self.assertEqual(set(state), {"latest_user_message", "previous_assistant_reply"})
        self.assertEqual(ex["completion"], LETTERS[CATEGORIES.index("ordinary")])

    def test_task_options_match_categories(self):
        task = json.loads(messages(prompt="hi")[1]["content"])
        self.assertEqual([o["key"] for o in task["options"]], CATEGORIES)
        self.assertEqual(list(task)[-1], "state")


if __name__ == "__main__":
    unittest.main()
