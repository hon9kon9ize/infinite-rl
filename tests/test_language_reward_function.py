import unittest
from pathlib import Path
from infinite_rl.parser import ExampleParser
from infinite_rl.reward_functions.language import LanguageRewardFunction


class TestLanguageRewardFunction(unittest.TestCase):
    """Test language/dialect reward function."""

    def setUp(self):
        package_dir = Path(__file__).parent.parent / "infinite_rl" / "examples"
        if not package_dir.exists():
            package_dir = Path(__file__).parent.parent / "examples"
        self.examples = ExampleParser.get_all_examples(package_dir)

        self.reward_fn = LanguageRewardFunction(task_name="language")
        self.reward_fn.initialize()

    def test_language_example_pass(self):
        example = self.examples.get("LANGUAGE")
        self.assertIsNotNone(example)

        score = self.reward_fn.compute_reward(example["response"], example["answer"])
        self.assertEqual(score.format_score, 1.0)
        self.assertEqual(score.correctness_score, 1.0)

    def test_language_mismatch(self):
        # Expected Cantonese, but response is Mandarin/Chinese -> partial credit (mapping 0.25)
        score = self.reward_fn.compute_reward("<answer>这是普通话。</answer>", "yue")
        self.assertEqual(score.format_score, 1.0)
        self.assertAlmostEqual(score.correctness_score, 0.25, places=3)

    def test_mapping_detected_zh_hant_for_zh(self):
        # If CLD2 details indicate zh-Hant for the response but expected is zh, score should be 0.25
        from unittest.mock import patch

        with patch.object(
            type(self.reward_fn),
            "_detect_lang_details",
            return_value=[("Chinese", "zh-Hant", 100)],
        ):
            score = self.reward_fn.compute_reward("<answer>示例文本</answer>", "zh")
            self.assertEqual(score.format_score, 1.0)
            self.assertAlmostEqual(score.correctness_score, 0.25, places=3)

    def test_en_detection(self):
        from unittest.mock import patch

        with patch.object(
            type(self.reward_fn),
            "_detect_lang_details",
            return_value=[("English", "en", 100)],
        ):
            score = self.reward_fn.compute_reward("<answer>Hello world</answer>", "en")
            self.assertEqual(score.format_score, 1.0)
            self.assertAlmostEqual(score.correctness_score, 1.0, places=3)

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
        self.assertAlmostEqual(score.correctness_score, expected_score, places=3)
