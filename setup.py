import os
import json
import urllib.request
from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

# Configuration
RUNTIME_FILES = ["universal_js.wasm", "micropython.wasm"]
GITHUB_REPO = os.environ.get("RUNTIME_GITHUB_REPO", "hon9kon9ize/infinite-rl")


def read_version():
    try:
        with open("VERSION.txt", "r") as f:
            return f.read().strip()
    except:
        return "0.1.16"  # Fallback version


PACKAGE_VERSION = read_version()


def download_runtimes():
    """Download assets directly into the source tree before build."""
    # 1. Determine the path relative to this setup.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dest_dir = os.path.join(base_dir, "infinite_rl", "runtimes")
    os.makedirs(dest_dir, exist_ok=True)

    # 2. Define the tag (Prioritize the version we are currently installing)
    # We use 'runtimes-v' prefix to match your repo structure
    tag = f"runtimes-v{PACKAGE_VERSION}"
    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/tags/{tag}"

    print(f"[info] Fetching runtimes for {tag}...")

    try:
        # Headers to identify the request (helps avoid some generic blocks)
        headers = {"User-Agent": "Mozilla/5.0 (Python urllib)"}
        req = urllib.request.Request(api_url, headers=headers)

        with urllib.request.urlopen(req) as resp:
            release_info = json.load(resp)

        assets = {
            a["name"]: a["browser_download_url"] for a in release_info.get("assets", [])
        }

        for fname in RUNTIME_FILES:
            target_path = os.path.join(dest_dir, fname)

            # Skip if already exists (e.g. during local dev)
            if os.path.exists(target_path):
                print(f"[info] {fname} already exists, skipping.")
                continue

            url = assets.get(fname)
            if not url:
                print(f"[error] {fname} not found in release assets for {tag}")
                continue

            print(f"[info] Downloading {fname}...")
            urllib.request.urlretrieve(url, target_path)

    except Exception as e:
        print(f"[error] Failed to download runtimes: {e}")
        # We raise an error here because the package won't work without them
        raise RuntimeError(
            f"Could not download required wasm runtimes from GitHub: {e}"
        )


class build_py(_build_py):
    """Inject the download process into the build lifecycle."""

    def run(self):
        download_runtimes()
        super().run()


setup(
    name="infinite_rl",
    version=PACKAGE_VERSION,
    packages=[
        "infinite_rl",
        "infinite_rl.reward_functions",
        "infinite_rl.examples",
        "infinite_rl.runtimes",
    ],
    include_package_data=True,
    package_data={
        "infinite_rl.runtimes": ["*.wasm"],
        "infinite_rl": ["VERSION.txt"],
    },
    install_requires=[
        "wasmtime",
        "sympy",
        "antlr4-python3-runtime==4.11.1",
        "pycld2",
        "cantonesedetect",
    ],
    cmdclass={"build_py": build_py},
)
