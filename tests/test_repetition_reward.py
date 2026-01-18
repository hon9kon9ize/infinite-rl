import pytest
from infinite_rl.reward_functions.repetition import ngram_repetition_reward


def test_short_text_no_penalty():
    assert ngram_repetition_reward("hello world", n=3) == 0.0


def test_no_repetition_returns_zero():
    text = "this is a test of unique ngrams without repeats"
    assert ngram_repetition_reward(text, n=2) == 0.0


def test_simple_repetition_penalty():
    text = "spam spam spam spam spam"
    # tokens = ['spam', 'spam', 'spam', 'spam', 'spam']
    # for n=2 ngrams = [('spam','spam') x4] -> counts {('spam','spam'):4} duplicates = 3
    # duplicates/len(ngrams) = 3/4 = 0.75 -> penalty = 0.75 * -0.1 = -0.075
    p = ngram_repetition_reward(text, n=2, weight=-0.1)
    assert pytest.approx(-0.075, rel=1e-6) == p


def test_punctuation_and_case_handling():
    text = "Hello, hello! Hello. world? hello"
    # tokens -> ['hello','hello','hello','world','hello'] n=1 ngrams count = 5 duplicates = 3 -> penalty = 3/5 * -0.2 = -0.12
    p = ngram_repetition_reward(text, n=1, weight=-0.2)
    assert pytest.approx(-0.12, rel=1e-6) == p


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


def test_tokenize_extended_cjk():
    from infinite_rl.reward_functions.repetition import _tokenize

    # U+20000 is a supplementary plane CJK unified ideograph
    ch = "\U00020000"
    text = f"start {ch} end"
    tokens = _tokenize(text)

    # Should include the supplementary CJK character as a single token
    assert ch in tokens
    assert "start" in tokens and "end" in tokens
