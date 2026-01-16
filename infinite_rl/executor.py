import os
import wasmtime
import tempfile
from importlib import resources


class Executor:
    def __init__(self, timeout=5):
        self.timeout = timeout
        self.engine = wasmtime.Engine()
        self.linker = wasmtime.Linker(self.engine)
        self.linker.define_wasi()

        self._modules = {
            "js": self._load_wasm_module("universal_js.wasm"),
            "python": self._load_wasm_module("micropython.wasm"),
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

    def _execute_wasm(self, lang, code):
        module = self._modules.get(lang)
        store = wasmtime.Store(self.engine)
        wasi_config = wasmtime.WasiConfig()

        # Create temp files for Input, Output, and Error
        with tempfile.NamedTemporaryFile(
            mode="wb", delete=False
        ) as in_f, tempfile.NamedTemporaryFile(
            mode="w+", delete=False
        ) as out_f, tempfile.NamedTemporaryFile(
            mode="w+", delete=False
        ) as err_f:

            # Write code to stdin file and close handle so Wasm can read it
            in_f.write(code.encode("utf-8"))
            in_f.close()

            wasi_config.stdin_file = in_f.name
            wasi_config.stdout_file = out_f.name
            wasi_config.stderr_file = err_f.name

            if lang == "python":
                wasi_config.argv = ["micropython", "-c", code]
            else:
                wasi_config.argv = ["javy"]

            store.set_wasi(wasi_config)

            try:
                instance = self.linker.instantiate(store, module)
                start = instance.exports(store)["_start"]
                start(store)
            except Exception:
                pass

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

    def run_single(self, code, lang):
        lang = lang.lower()
        if lang in ["js", "javascript"]:
            lang = "js"

        try:
            stdout, stderr = self._execute_wasm(lang, code)
            return stdout.strip(), stderr.strip()
        except Exception as e:
            return None, f"Executor Error: {str(e)}"

    def batch_run(self, completions, lang):
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor() as executor:
            return list(executor.map(lambda c: self.run_single(c, lang), completions))
