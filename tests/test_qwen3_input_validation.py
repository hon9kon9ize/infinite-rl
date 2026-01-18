import unittest
import importlib.util

wasmtime_spec = importlib.util.find_spec("wasmtime")

if wasmtime_spec is None:
    raise unittest.SkipTest(
        "wasmtime not available in environment; skipping qwen3 input validation tests"
    )

from infinite_rl.executor import Executor


class TestQwen3InputValidation(unittest.TestCase):
    def setUp(self):
        self.executor = Executor()
        # Stub out the module so we can reach input validation without requiring a real wasm
        self.executor._modules["qwen3_embed"] = object()

        # Patch the linker's instantiate to return a dummy instance so the test
        # exercises input validation only and doesn't invoke the real wasm runtime.
        class _DummyInstance:
            def exports(self, store):
                return {"_start": lambda s: None}

        self.executor.linker.instantiate = lambda store, module: _DummyInstance()

    def test_swapped_tuple_raises_value_error(self):
        # Swapped order: (query, document) -> should raise a clear ValueError
        swapped = ("How does X work?", "This passage explains how X works.")
        with self.assertRaises(ValueError):
            self.executor._execute_wasm("qwen3_embed", swapped)

    def test_correct_tuple_does_not_raise_value_error(self):
        # Correct order: (document, query) -> should not raise ValueError (may raise later due to stub module)
        correct = ("This passage explains how X works.", "How does X work?")
        try:
            self.executor._execute_wasm("qwen3_embed", correct)
        except Exception as e:
            # Should not be a ValueError stemming from validation
            self.assertNotIsInstance(e, ValueError)


if __name__ == "__main__":
    unittest.main()
