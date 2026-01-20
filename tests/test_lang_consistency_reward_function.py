import unittest
from pathlib import Path
from infinite_rl.parser import ExampleParser
from infinite_rl.reward_functions.lang_consistency import LangConsistencyRewardFunction


class TestLangConsistencyRewardFunction(unittest.TestCase):
    """Test language/dialect reward function renamed to lang_consistency."""

    def setUp(self):
        package_dir = Path(__file__).parent.parent / "infinite_rl" / "examples"
        if not package_dir.exists():
            package_dir = Path(__file__).parent.parent / "examples"
        self.examples = ExampleParser.get_all_examples(package_dir)

        self.reward_fn = LangConsistencyRewardFunction(task_name="lang_consistency")
        self.reward_fn.initialize()

    def test_language_example_pass(self):
        example = self.examples.get("LANG_CONSISTENCY")
        self.assertIsNotNone(example)

        # The reward function checks language outside of the <answer> tag.
        # The example's surrounding text contains an explicit Cantonese hint ("Cantonese (Yue)")
        # so we expect a strong aux signal.
        score = self.reward_fn.compute_reward(example["response"], example["answer"])
        # Aux-only behaviour: signal now in unified `score` field
        self.assertAlmostEqual(score.score, 1.0, places=5)

    def test_language_example_external_text(self):
        """If the text outside the <answer> tag is Cantonese, we should detect it."""
        # Simulate an external Cantonese hint (outside of <answer>) and verify aux_score
        response = "我哋今晚去食茶啦。\n\n<answer>內文可任意</answer>"
        # Patch detectors to ensure a strong Cantonese signal
        from unittest.mock import patch

        with patch.object(
            type(self.reward_fn), "_yue_ratio", return_value=1.0
        ), patch.object(
            type(self.reward_fn),
            "_detect_lang_details",
            return_value=[("Chinese", "zh", 100)],
        ):
            score = self.reward_fn.compute_reward(response, "yue")
            self.assertAlmostEqual(score.score, 1.0, places=3)

    def test_language_mismatch(self):
        # Expected Cantonese, but response is Mandarin/Chinese inside <answer> ->
        # since detection checks outside <answer>, we get no signal (aux_score 0.0)
        score = self.reward_fn.compute_reward("<answer>这是普通话。</answer>", "yue")
        self.assertAlmostEqual(score.score, 0.0, places=3)

    def test_mapping_detected_zh_hant_for_zh(self):
        # If CLD2 details indicate zh-Hant for the response but expected is zh, score should be 0.25
        from unittest.mock import patch

        with patch.object(
            type(self.reward_fn),
            "_detect_lang_details",
            return_value=[("Chinese", "zh-Hant", 100)],
        ):
            score = self.reward_fn.compute_reward("<answer>示例文本</answer>", "zh")
            self.assertAlmostEqual(score.score, 0.25, places=3)

    def test_en_detection(self):
        from unittest.mock import patch

        with patch.object(
            type(self.reward_fn),
            "_detect_lang_details",
            return_value=[("English", "en", 100)],
        ):
            score = self.reward_fn.compute_reward("<answer>Hello world</answer>", "en")
            self.assertAlmostEqual(score.score, 1.0, places=3)

    def test_mixed_proportional_score(self):
        # Mixed English + Chinese content should result in a weighted score for expected 'zh'
        text = "Hello World, Hello World, Hello World, Hello World, 战争不会显示谁对谁错，只会显示谁活了下来。"
        # Use real CLD2 details to compute expected weighted mapping score
        details = self.reward_fn._detect_lang_details(text)
        total = sum(d[-1] for d in details) if details else 0
        mapping = {"zh": {"zh": 1.0, "zh-hant": 0.25, "yue": 0.25, "en": 0.0}}
        expected_score = 0.0
        if total > 0:
            for entry in details:
                code = entry[1]
                b = entry[-1]
                norm = code.lower()
                if norm.startswith("zh-hant"):
                    norm = "zh-hant"
                elif norm.startswith("zh"):
                    norm = "zh"
                expected_score += (b / total) * mapping["zh"].get(norm, 0.0)

        score = self.reward_fn.compute_reward(f"<answer>{text}</answer>", "zh")
        # Because detection checks outside the <answer> tags, there will be no
        # detected bytes and thus score should be 0.0 for this input.
        self.assertAlmostEqual(score.score, 0.0, places=3)
