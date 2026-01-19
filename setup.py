import os
import json
import urllib.request
from setuptools import setup
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as _build_py

# Configuration
RUNTIME_FILES = ["universal_js.wasm", "micropython.wasm"]
GITHUB_REPO = os.environ.get("RUNTIME_GITHUB_REPO", "hon9kon9ize/infinite-rl")


def get_version():
    try:
        with open("VERSION.txt", "r") as f:
            return f.read().strip()
    except:
        return "0.1.16"


PACKAGE_VERSION = get_version()


def download_assets(target_dir):
    """Core download logic."""
    tag = f"runtimes-v{PACKAGE_VERSION}"
    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/tags/{tag}"

    print(f"--> [Action] Downloading runtimes to: {target_dir}")
    os.makedirs(target_dir, exist_ok=True)

    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            assets = {
                a["name"]: a["browser_download_url"]
                for a in json.load(resp).get("assets", [])
            }

        for fname in RUNTIME_FILES:
            if fname in assets:
                dest = os.path.join(target_dir, fname)
                print(f"    Downloading {fname}...")
                urllib.request.urlretrieve(assets[fname], dest)
            else:
                print(f"    [!] Missing asset in release: {fname}")
    except Exception as e:
        print(f"    [!] Download failed: {e}")
        # If runtimes are mandatory, raise an error to stop installation
        raise RuntimeError(
            f"Could not download runtimes. Package will be incomplete. Error: {e}"
        )


class build_py(_build_py):
    def run(self):
        # 1. First, let the standard build process run.
        # This creates 'self.build_lib/infinite_rl/runtimes'
        super().run()

        # 2. Determine where setuptools is staging the files for the wheel
        # We must inject the files into the build_lib directory
        build_runtime_dir = os.path.join(self.build_lib, "infinite_rl", "runtimes")

        # 3. Download directly into the build staging area
        download_assets(build_runtime_dir)


setup(
    name="infinite_rl",
    version=PACKAGE_VERSION,
    packages=find_packages(),
    include_package_data=True,
    # This is critical for finding files inside the package
    package_data={
        "infinite_rl.runtimes": ["*.wasm"],
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
