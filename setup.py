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


def download_runtimes(dest_dir):
    """Downloads files into the actual build directory."""
    tag = f"runtimes-v{PACKAGE_VERSION}"
    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/tags/{tag}"

    print(f"--> [Triggered] Downloading runtimes for {tag} to {dest_dir}")
    os.makedirs(dest_dir, exist_ok=True)

    try:
        # Request with a User-Agent to avoid generic bot blocks
        req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            release_info = json.load(resp)

        assets = {
            a["name"]: a["browser_download_url"] for a in release_info.get("assets", [])
        }

        for fname in RUNTIME_FILES:
            if fname not in assets:
                print(f" [!] Warning: {fname} not found in GitHub assets.")
                continue

            target = os.path.join(dest_dir, fname)
            print(f" --> Downloading {fname}...")
            urllib.request.urlretrieve(assets[fname], target)

    except Exception as e:
        print(f" [!] Error downloading assets: {e}")
        # If this is critical, raise an error to stop the pip install
        raise RuntimeError(f"Required runtimes failed to download: {e}")


class build_py(_build_py):
    """Custom build command to download assets before the build starts."""

    def run(self):
        # Determine the package directory inside the build environment
        # 'self.build_lib' is where pip gathers files for the wheel
        base_dir = os.path.dirname(os.path.abspath(__file__))
        target_dir = os.path.join(base_dir, "infinite_rl", "runtimes")

        # Download files into the source tree so they are picked up
        download_runtimes(target_dir)

        # Now run the standard build which will copy these files to the wheel
        super().run()


setup(
    name="infinite_rl",
    version=PACKAGE_VERSION,
    packages=find_packages(),
    include_package_data=True,
    package_data={
        # This tells setuptools to look for .wasm files in that specific folder
        "infinite_rl.runtimes": ["*.wasm"],
    },
    install_requires=[
        "wasmtime",
        "sympy",
        "antlr4-python3-runtime==4.11.1",
        "pycld2",
        "cantonesedetect",
    ],
    cmdclass={
        "build_py": build_py,
    },
)
