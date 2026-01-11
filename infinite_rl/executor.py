import subprocess
import os
import tempfile
import re
from concurrent.futures import ThreadPoolExecutor


class RewardExecutor:
    def __init__(self, timeout=5):
        self.timeout = timeout
        # Add Rust to path if not present
        rust_path = os.path.expanduser("~/.cargo/bin")
        if rust_path not in os.environ["PATH"]:
            os.environ["PATH"] += f":{rust_path}"

    def _execute_python(self, code, cwd):
        fpath = os.path.join(cwd, "script.py")
        with open(fpath, "w") as f:
            f.write(code)
        return subprocess.run(
            ["python3", fpath], capture_output=True, text=True, timeout=self.timeout
        )

    def _execute_js(self, code, cwd):
        fpath = os.path.join(cwd, "script.js")
        with open(fpath, "w") as f:
            f.write(code)
        return subprocess.run(
            ["node", fpath], capture_output=True, text=True, timeout=self.timeout
        )

    def _execute_cpp(self, code, cwd):
        fpath = os.path.join(cwd, "main.cpp")
        out = os.path.join(cwd, "cpp_bin")
        with open(fpath, "w") as f:
            f.write(code)
        subprocess.run(["g++", fpath, "-o", out], check=True, capture_output=True)
        return subprocess.run(
            [out], capture_output=True, text=True, timeout=self.timeout
        )

    def _execute_rust(self, code, cwd):
        fpath = os.path.join(cwd, "main.rs")
        out = os.path.join(cwd, "rs_bin")
        with open(fpath, "w") as f:
            f.write(code)
        subprocess.run(["rustc", fpath, "-o", out], check=True, capture_output=True)
        return subprocess.run(
            [out], capture_output=True, text=True, timeout=self.timeout
        )

    def _execute_java(self, code, cwd):
        # Java requires the filename to match the public class name
        class_match = re.search(r"public\s+class\s+(\w+)", code)
        class_name = class_match.group(1) if class_match else "Main"
        fpath = os.path.join(cwd, f"{class_name}.java")
        with open(fpath, "w") as f:
            f.write(code)
        subprocess.run(["javac", fpath], check=True, capture_output=True)
        return subprocess.run(
            ["java", "-cp", cwd, class_name],
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )

    def _execute_typescript(self, code, cwd):
        fpath = os.path.join(cwd, "script.ts")
        js_path = os.path.join(cwd, "script.js")

        with open(fpath, "w") as f:
            f.write(code)

        # Try ts-node first
        try:
            # Add --compiler-options to force CommonJS which is more reliable for simple scripts
            result = subprocess.run(
                [
                    "ts-node",
                    "--transpile-only",
                    "--compiler-options",
                    '{"module":"commonjs","target":"es2020"}',
                    fpath,
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            if result.returncode == 0:
                return result
        except FileNotFoundError:
            pass

        # Try tsx
        try:
            result = subprocess.run(
                ["tsx", fpath],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            if result.returncode == 0:
                return result
        except FileNotFoundError:
            pass

        # Try tsc compilation
        try:
            # Individual files without outFile is often more reliable
            compile_result = subprocess.run(
                [
                    "tsc",
                    fpath,
                    "--target",
                    "es2020",
                    "--module",
                    "commonjs",
                    "--esModuleInterop",
                    "--skipLibCheck",
                ],
                capture_output=True,
                timeout=self.timeout,
            )
            if compile_result.returncode == 0 and os.path.exists(js_path):
                return subprocess.run(
                    ["node", js_path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
        except FileNotFoundError:
            pass

        # Last resort: TypeScript is valid JavaScript, run with node directly
        return subprocess.run(
            ["node", fpath],
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )

    def run_single(self, code, lang):
        lang = lang.lower()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                if lang == "python":
                    res = self._execute_python(code, tmpdir)
                elif lang in ["js", "javascript"]:
                    res = self._execute_js(code, tmpdir)
                elif lang in ["ts", "typescript"]:
                    res = self._execute_typescript(code, tmpdir)
                elif lang in ["c++", "cpp"]:
                    res = self._execute_cpp(code, tmpdir)
                elif lang == "rust":
                    res = self._execute_rust(code, tmpdir)
                elif lang == "java":
                    res = self._execute_java(code, tmpdir)
                else:
                    return None, "Unsupported Language"

                return res.stdout.strip(), res.stderr.strip()
            except subprocess.TimeoutExpired:
                return None, "Timeout"
            except subprocess.CalledProcessError as e:
                return None, (
                    e.stderr.decode() if isinstance(e.stderr, bytes) else str(e)
                )
            except Exception as e:
                return None, str(e)

    def batch_run(self, completions, lang):
        """Runs a group of completions in parallel for GRPO efficiency."""
        with ThreadPoolExecutor() as executor:
            return list(executor.map(lambda c: self.run_single(c, lang), completions))
