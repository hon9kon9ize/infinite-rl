import os
import wasmtime
import tempfile
from importlib import resources
from typing import Union


class Executor:
    def __init__(self, timeout=5):
        self.timeout = timeout
        self.engine = wasmtime.Engine()
        self.linker = wasmtime.Linker(self.engine)
        self.linker.define_wasi()

        # Load available modules; be tolerant if optional runtimes are not present
        self._modules = {
            "javascript": self._load_wasm_module("universal_js.wasm"),
            "python": self._load_wasm_module("micropython.wasm"),
            "qwen3_embed": None,
        }

        # qwen3 is optional: don't fail initialization if it's missing
        try:
            self._modules["qwen3_embed"] = self._load_wasm_module("qwen3_embed.wasm")
        except FileNotFoundError:
            # Qwen3 runtime not available in this installation; feature is optional
            pass

    def _load_wasm_module(self, filename):
        # Local development check first
        local_path = os.path.join(os.path.dirname(__file__), "runtimes", filename)
        if os.path.exists(local_path):
            return wasmtime.Module.from_file(self.engine, local_path)

        # Package-based check
        try:
            path = resources.files("infinite_rl.runtimes").joinpath(filename)
            return wasmtime.Module.from_file(self.engine, str(path))
        except Exception:
            raise FileNotFoundError(f"Could not find {filename}")

    def _execute_wasm(self, lang: str, input: Union[str, tuple]):
        module = self._modules.get(lang)
        if module is None:
            raise FileNotFoundError(f"No wasm module registered for language '{lang}'")

        store = wasmtime.Store(self.engine)
        wasi_config = wasmtime.WasiConfig()

        # Support multiple input shapes. For most runtimes we write the input to stdin;
        # for the qwen3 embedding CLI we accept only (document, query) tuples and
        # translate them into CLI args.
        stdin_content = ""
        qwen3_document = None
        qwen3_query = None

        if lang == "qwen3_embed":
            # Accept only list/tuple (document, query) pairs to avoid ambiguous string parsing.
            if isinstance(input, (list, tuple)) and len(input) >= 2:
                qwen3_document, qwen3_query = str(input[0]), str(input[1])
                # Sanity check: document should usually be longer than query. This helps catch
                # accidental tuple swaps which can cause the qwen3 wasm to error in unclear ways.
                if len(qwen3_document.strip()) < len(qwen3_query.strip()):
                    raise ValueError(
                        "For 'qwen3_embed', the first element should be the document and the second the query.\n"
                        "Detected document shorter than query — did you swap the tuple?"
                    )
            else:
                raise TypeError(
                    "For 'qwen3_embed', 'input' must be a (document, query) tuple or list with at least 2 elements."
                )
        else:
            stdin_content = input if isinstance(input, str) else ""

        # Create temp files for Input, Output, and Error
        with tempfile.NamedTemporaryFile(
            mode="wb", delete=False
        ) as in_f, tempfile.NamedTemporaryFile(
            mode="w+", delete=False
        ) as out_f, tempfile.NamedTemporaryFile(
            mode="w+", delete=False
        ) as err_f:

            # Write stdin content if provided
            in_f.write(stdin_content.encode("utf-8"))
            in_f.close()

            wasi_config.stdin_file = in_f.name
            wasi_config.stdout_file = out_f.name
            wasi_config.stderr_file = err_f.name

            # Expose the packaged runtimes directory. For qwen3 we attempt to make
            # a directory named `qwen3_local_cache` available at the root of the
            # WASI filesystem (this matches how the CLI resolves the cache path).
            runtime_dir = os.path.join(os.path.dirname(__file__), "runtimes")

            # Preopen the packaged runtimes directory (useful for the wasm binary) and
            # optionally preopen a specific cache directory if provided.
            cache_dir_arg = None
            preopened_runtime = False
            try:
                if os.path.exists(runtime_dir):
                    # Map the package runtimes directory into the WASI root so that
                    # 'qwen3_local_cache' inside it can be referenced as a relative path.
                    wasi_config.preopen_dir(runtime_dir, ".")
                    preopened_runtime = True

                # 1) Explicit environment variable wins
                qcache_env = os.environ.get("QWEN3_CACHE_DIR")
                if qcache_env and os.path.isdir(qcache_env):
                    # Preopen the absolute cache path under a stable guest basename
                    guest_name = (
                        os.path.basename(os.path.abspath(qcache_env))
                        or "qwen3_local_cache"
                    )
                    wasi_config.preopen_dir(qcache_env, guest_name)
                    cache_dir_arg = guest_name
                else:
                    # 2) Otherwise, only look for a cache that lives directly under the
                    #    packaged runtimes directory. If we already preopened the parent
                    #    runtime dir, don't preopen the subdir (may fail); instead pass a
                    #    relative cache path that the wasm can resolve inside the preopened dir.
                    qwen3_cache = os.path.join(runtime_dir, "qwen3_local_cache")
                    if os.path.isdir(qwen3_cache):
                        if preopened_runtime:
                            cache_dir_arg = "qwen3_local_cache"
                        else:
                            wasi_config.preopen_dir(qwen3_cache, "qwen3_local_cache")
                            cache_dir_arg = "qwen3_local_cache"
            except Exception as e:
                # If preopen fails for any reason, we proceed without setting cache_dir_arg
                cache_dir_arg = None

            # Set argv appropriately per runtime
            if lang == "python":
                wasi_config.argv = ["micropython", "-c", code]
            elif lang == "qwen3_embed":
                # Map to the same CLI as in main.rs
                # Document and query may contain spaces; pass them as two separate args
                argv = [
                    "qwen3_embed",
                    "--document",
                    qwen3_document or "",
                    "--query",
                    qwen3_query or "",
                ]
                # If we discovered a cache directory, pass it explicitly via --cache-dir
                if cache_dir_arg:
                    argv.extend(["--cache-dir", cache_dir_arg])
                wasi_config.argv = argv

            # Instantiate and run the wasm module (capture any runtime exceptions but continue to read outputs)
            try:
                # Apply WASI config and instantiate the module
                store.set_wasi(wasi_config)
                instance = self.linker.instantiate(store, module)
                start = instance.exports(store)["_start"]
                start(store)
            except Exception as e:
                import sys, traceback

                print("instantiate/start failed:", e, file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

            # Read the outputs
            with open(out_f.name, "r") as f:
                stdout = f.read()
            with open(err_f.name, "r") as f:
                stderr = f.read()

            # Clean up all temp files
            for path in [in_f.name, out_f.name, err_f.name]:
                if os.path.exists(path):
                    os.remove(path)

            return stdout, stderr

    def run_single(self, input: Union[str, tuple], lang: str):
        import json

        lang = lang.lower()

        # For qwen3 text embedding we require passing (document, query) tuples/lists directly
        try:
            stdout, stderr = self._execute_wasm(lang, input)

            # If this was a qwen3 run, ensure we have meaningful output and parse it
            if lang == "qwen3_embed":
                if not stdout or not stdout.strip():
                    # Convert empty stdout into an Executor-style error
                    err_msg = stderr.strip() or "qwen3 runtime produced no output"
                    return None, f"Executor Error: {err_msg}"

                try:
                    parsed = json.loads(stdout)
                    if isinstance(parsed, dict) and "cosine_similarity" in parsed:
                        return str(parsed["cosine_similarity"]).strip(), stderr.strip()
                except Exception:
                    # Fall back to returning raw stdout if JSON parsing fails
                    return stdout.strip(), stderr.strip()

            return stdout.strip(), stderr.strip()
        except Exception as e:
            return None, f"Executor Error: {str(e)}"

    def batch_run(self, completions, lang):
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor() as executor:
            return list(executor.map(lambda c: self.run_single(c, lang), completions))
