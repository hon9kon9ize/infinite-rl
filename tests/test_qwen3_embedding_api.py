import unittest
import json
import importlib.util

wasmtime_spec = importlib.util.find_spec("wasmtime")
if wasmtime_spec is None:
    raise unittest.SkipTest(
        "wasmtime not available in environment; skipping qwen3 embedding API tests"
    )

from infinite_rl.executor import Executor


class TestQwen3EmbeddingAPI(unittest.TestCase):
    def setUp(self):
        self.executor = Executor()
        # Stub out the module so we can reach code paths without a real wasm module
        self.executor._modules["qwen3_embed"] = object()

        # Monkeypatch _execute_wasm to return deterministic embeddings
        def _fake_execute_wasm(lang, input):
            # input is a tuple (doc, query)
            if lang != "qwen3_embed":
                return "", ""
            doc, qry = input[0], input[1]
            if doc and not qry:
                return json.dumps({"embedding": [1.0, 0.0, 0.0]}), ""
            if qry and not doc:
                return json.dumps({"embedding": [0.0, 1.0, 0.0]}), ""
            if doc and qry:
                return json.dumps({"cosine_similarity": 0.5}), ""
            return "", "no input"

        self.executor._execute_wasm = _fake_execute_wasm

    def test_get_embedding_document(self):
        emb, err = self.executor.get_embedding("doc text", role="document")
        self.assertEqual(emb, [1.0, 0.0, 0.0])
        self.assertEqual(err, "")

    def test_get_embedding_query(self):
        emb, err = self.executor.get_embedding("query text", role="query")
        self.assertEqual(emb, [0.0, 1.0, 0.0])
        self.assertEqual(err, "")

    def test_cosine_similarity_and_embedding_similarity(self):
        e1, _ = self.executor.get_embedding("d1", role="document")
        e2, _ = self.executor.get_embedding("q1", role="query")
        sim = self.executor.cosine_similarity(e1, e2)
        # orthogonal vectors -> 0.0
        self.assertEqual(sim, 0.0)

        sim2, stderr = self.executor.embedding_similarity("d1", "q1")
        self.assertEqual(sim2, 0.0)
        self.assertEqual(stderr, "")

    def test_run_single_backward_compatibility(self):
        out, err = self.executor.run_single(("doc", "qry"), "qwen3_embed")
        # fake _execute_wasm returns cosine_similarity 0.5
        self.assertEqual(out, "0.5")
        self.assertEqual(err, "")

    def test_get_embedding_similarity_json_raises(self):
        # If the runtime returns a similarity JSON when we asked for a single embedding,
        # surface a clear ValueError explaining the likely cause.
        def _fake_sim_return(lang, input):
            return json.dumps({"cosine_similarity": 0.321}), ""

        self.executor._execute_wasm = _fake_sim_return

        with self.assertRaises(ValueError) as cm:
            self.executor.get_embedding("some text", role="document")

        self.assertIn("cosine_similarity", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
