import os
import json
import urllib.request
from setuptools import setup
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as _build_py

# Configuration
RUNTIME_FILES = ["universal_js.wasm", "micropython.wasm"]
GITHUB_REPO = os.environ.get("RUNTIME_GITHUB_REPO", "hon9kon9ize/infinite-rl")
# Use a known-good tag if the version-specific one fails
FALLBACK_TAG = "runtimes-v0.1.16"


def get_version():
    try:
        with open("VERSION.txt", "r") as f:
            return f.read().strip()
    except:
        return "0.1.16"


PACKAGE_VERSION = get_version()


def download_assets(target_dir):
    """Downloads files directly without relying on the GitHub API JSON."""
    os.makedirs(target_dir, exist_ok=True)

    # We try the current version tag first, then fallback
    tags_to_try = [f"runtimes-v{PACKAGE_VERSION}", FALLBACK_TAG]

    for tag in tags_to_try:
        success = True
        print(f"--> [Action] Attempting to download runtimes from tag: {tag}")

        for fname in RUNTIME_FILES:
            # Direct Download URL format (doesn't require API/tokens)
            url = f"https://github.com/{GITHUB_REPO}/releases/download/{tag}/{fname}"
            dest = os.path.join(target_dir, fname)

            try:
                print(f"    Downloading {fname}...")
                # Add a user agent to avoid being blocked
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req) as response, open(
                    dest, "wb"
                ) as out_file:
                    out_file.write(response.read())
            except Exception as e:
                print(f"    [!] Failed to download {fname} from {tag}: {e}")
                success = False
                break

        if success:
            print(f"--> [Success] All runtimes downloaded from {tag}")
            return

    raise RuntimeError(
        "Could not download runtimes from any known tags. Check your release tag names."
    )


class build_py(_build_py):
    def run(self):
        super().run()
        # Inject into the build_lib so it ends up in the Wheel
        build_runtime_dir = os.path.join(self.build_lib, "infinite_rl", "runtimes")
        download_assets(build_runtime_dir)


setup(
    name="infinite_rl",
    version=PACKAGE_VERSION,
    packages=find_packages(),
    include_package_data=True,
    package_data={"infinite_rl.runtimes": ["*.wasm"]},
    install_requires=[
        "wasmtime",
        "sympy",
        "antlr4-python3-runtime==4.11.1",
        "pycld2",
        "pylatexenc",
        "cantonesedetect",
    ],
    cmdclass={"build_py": build_py},
)
