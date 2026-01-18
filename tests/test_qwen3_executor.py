import re
import pytest
from infinite_rl import Executor


def test_qwen3_executor_behaviour():
    executor = Executor(timeout=1)
    # Behaviour depends on whether the qwen3 runtime and cache are available.
    out, err = executor.run_single(("doc", "query"), "qwen3_embed")

    if executor._modules.get("qwen3_embed") is None:
        # Runtime not present: should return an error
        assert out is None
        assert isinstance(err, str)
        assert err.startswith("Executor Error:")
    else:
        # Runtime present: either returns a float-like similarity string, or an Executor Error
        if out is None:
            assert isinstance(err, str)
            assert err.startswith("Executor Error:")
        else:
            assert isinstance(out, str)
            val = float(out)
            assert -1.0 <= val <= 1.0


def test_qwen3_input_formats():
    executor = Executor(timeout=1)
    inputs = [("doc1", "q1"), "doc2|||q2", "doc3\n---\nq3"]

    for input_value in inputs:
        out, err = executor.run_single(input_value, "qwen3_embed")

        if executor._modules.get("qwen3_embed") is None:
            assert out is None
            assert isinstance(err, str)
            assert err.startswith("Executor Error:")
        else:
            # If the runtime exists but cache is missing, we still may get an Executor Error
            if out is None:
                assert isinstance(err, str)
                assert err.startswith("Executor Error:")
            else:
                # Otherwise, we expect a float-like similarity string
                val = float(out)
                assert -1.0 <= val <= 1.0
