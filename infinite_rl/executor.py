import os
import json
import wasmtime
import tempfile
from importlib import resources
from typing import Union, Tuple, List


class Executor:
    def __init__(self, timeout=5):
        self.timeout = timeout
        config = wasmtime.Config()
        config.wasm_simd = True  # enable SIMD extension
        config.wasm_threads = True  # enable threads/atomics
        config.cranelift_opt_level = "speed"  # optional: favor speed
        self.engine = wasmtime.Engine(config)
        self.linker = wasmtime.Linker(self.engine)
        self.linker.define_wasi()

        # Load available modules; be tolerant if optional runtimes are not present
        self._modules = {
            "javascript": self._load_wasm_module("puzzle_js.wasm"),
        }

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

    def _execute_wasm(
        self, lang: str, input: Union[str, tuple], argv: List[str] = None
    ):
        module = self._modules.get(lang)
        if module is None:
            raise FileNotFoundError(f"No wasm module registered for language '{lang}'")

        store = wasmtime.Store(self.engine)
        wasi_config = wasmtime.WasiConfig()

        # For supported runtimes we write string inputs to stdin.
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

            # Expose the packaged runtimes directory to the WASI filesystem if present.
            runtime_dir = os.path.join(os.path.dirname(__file__), "runtimes")
            try:
                if os.path.exists(runtime_dir):
                    wasi_config.preopen_dir(runtime_dir, ".")
            except Exception:
                pass

            # Set argv appropriately per runtime
            if argv:
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

    def run_single(self, input: Union[str, tuple], lang: str, argv: List[str] = None):
        lang = lang.lower()
        try:
            stdout, stderr = self._execute_wasm(lang, input, argv)
            return stdout.strip(), stderr.strip()
        except Exception as e:
            return None, f"Executor Error: {str(e)}"

    def batch_run(self, completions, lang):
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor() as executor:
            return list(executor.map(lambda c: self.run_single(c, lang), completions))
