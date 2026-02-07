import pytest
from infinite_rl.reward_functions.repetition import ngram_repetition_reward


def test_short_text_no_penalty():
    # Short text can't have n-gram repetition, so returns perfect score
    assert ngram_repetition_reward("hello world", n=3) == 1.0


def test_no_repetition_returns_perfect_score():
    text = "this is a test of unique ngrams without repeats"
    # No repetition should return perfect score (1.0)
    assert ngram_repetition_reward(text, n=2) == 1.0


def test_simple_repetition_penalty():
    text = "spam spam spam spam spam"
    # tokens = ['spam', 'spam', 'spam', 'spam', 'spam']
    # for n=2 ngrams = [('spam','spam') x4] -> counts {('spam','spam'):4} duplicates = 3
    # duplicates/len(ngrams) = 3/4 = 0.75 -> score = 1.0 - 0.75 = 0.25
    score = ngram_repetition_reward(text, n=2)
    assert pytest.approx(0.25, rel=1e-6) == score


def test_punctuation_and_case_handling():
    text = "Hello, hello! Hello. world? hello"
    # tokens -> ['hello','hello','hello','world','hello'] n=1 ngrams count = 5 duplicates = 3 -> ratio = 3/5 = 0.6 -> score = 1.0 - 0.6 = 0.4
    score = ngram_repetition_reward(text, n=1)
    assert pytest.approx(0.4, rel=1e-6) == score


def test_invalid_n_raises():
    with pytest.raises(ValueError):
        ngram_repetition_reward("some text", n=0)


def test_tokenize_chinese():
    from infinite_rl.reward_functions.repetition import _tokenize

    text = "Hello World, 战争不会显示谁对谁错，只会显示谁活了下来。"
    tokens = _tokenize(text)

    # Latin tokens preserved
    assert tokens[0:2] == ["hello", "world"]

    # Chinese should be split into single-character tokens
    assert "战" in tokens
    assert "来" in tokens
    assert len([t for t in tokens if len(t) == 1 and not t.isalnum()]) == 0


def test_perfect_repetition_score():
    # Text with no repetition should get perfect score
    text = "the quick brown fox jumps over the lazy dog"
    score = ngram_repetition_reward(text, n=2)
    assert score == 1.0


def test_maximum_repetition_score():
    # Text with maximum repetition should get minimum score
    text = "word word word word word"
    score = ngram_repetition_reward(text, n=1)
    # All 5 tokens are the same, so repetition_ratio = 4/5 = 0.8, score = 1.0 - 0.8 = 0.2
    assert pytest.approx(0.2, rel=1e-6) == score


def test_partial_repetition_score():
    # Text with some repetition
    text = "hello world hello universe world"
    score = ngram_repetition_reward(text, n=1)
    # tokens: ['hello', 'world', 'hello', 'universe', 'world']
    # duplicates: hello appears 2x, world appears 2x -> duplicates = (2-1) + (2-1) = 2
    # repetition_ratio = 2/5 = 0.4, score = 1.0 - 0.4 = 0.6
    assert pytest.approx(0.6, rel=1e-6) == score
    from infinite_rl.reward_functions.repetition import _tokenize

    # U+20000 is a supplementary plane CJK unified ideograph
    ch = "\U00020000"
    text = f"start {ch} end"
    tokens = _tokenize(text)

    # Should include the supplementary CJK character as a single token
    assert ch in tokens
    assert "start" in tokens and "end" in tokens
