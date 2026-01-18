import os
import json
import urllib.request
from setuptools import setup
from setuptools.command.install import install as _install

RUNTIME_FILES = [
    "universal_js.wasm",
    "micropython.wasm",
    "qwen3_embed.wasm",
    "qwen3_local_cache.zip",
]
GITHUB_REPO = os.environ.get("RUNTIME_GITHUB_REPO", "hon9kon9ize/infinite-rl")


# Central version source
def read_version():
    try:
        with open("VERSION.txt", "r") as f:
            return f.read().strip()
    except Exception:
        return "0.0.0"


PACKAGE_VERSION = read_version()


def download_runtimes_from_release(tag=None, dest_dir="infinite_rl/runtimes"):
    """Download runtime assets for a specific GitHub release tag.

    If `tag` is None, the function defaults to `PACKAGE_VERSION`.
    Tag is normalized to include a leading 'v' (e.g., 'v0.1.13').
    """
    if not tag:
        tag = PACKAGE_VERSION

    # Normalize tag to start with 'v'
    if not str(tag).startswith("runtimes-v"):
        tag = f"runtimes-{tag}"

    os.makedirs(dest_dir, exist_ok=True)
    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/tags/{tag}"

    try:
        with urllib.request.urlopen(api_url) as resp:
            release_info = json.load(resp)
    except Exception as e:
        print(f"[warning] Could not fetch release info from {api_url}: {e}")
        return

    assets = {
        a.get("name"): a.get("browser_download_url")
        for a in release_info.get("assets", [])
    }

    # Print resolved URLs for debugging
    print(f"[info] Resolved release tag: {tag}")
    for fname in RUNTIME_FILES:
        url = assets.get(fname)
        if url:
            print(f"[info] {fname} -> {url}")
        else:
            print(f"[warn] {fname} not found in release {tag}")

    for fname in RUNTIME_FILES:
        # Special handling for zipped cache archives (e.g., qwen3_local_cache.zip)
        if fname.endswith(".zip"):
            # We expect the zip to extract into a folder (e.g., qwen3_local_cache)
            cache_folder = os.path.splitext(fname)[0]
            cache_target_dir = os.path.join(dest_dir, cache_folder)
            if os.path.isdir(cache_target_dir):
                print(
                    f"[info] {cache_folder} already present, skipping download/extract"
                )
                continue
            url = assets.get(fname)
            if not url:
                print(f"[warning] {fname} not found in release assets for {api_url}")
                continue
            zip_target = os.path.join(dest_dir, fname)
            try:
                print(f"[info] Downloading {fname} from {url} -> {zip_target}")
                urllib.request.urlretrieve(url, zip_target)
                print(f"[info] Extracting {fname} to {dest_dir}")
                import zipfile

                with zipfile.ZipFile(zip_target, "r") as zf:
                    zf.extractall(dest_dir)
                # Optional: remove the zip after extracting
                try:
                    os.remove(zip_target)
                except Exception:
                    pass
            except Exception as e:
                print(f"[warning] Failed to download or extract {fname}: {e}")
            continue

        # Normal file (e.g., wasm)
        target = os.path.join(dest_dir, fname)
        if os.path.exists(target):
            print(f"[info] {fname} already present, skipping download")
            continue
        url = assets.get(fname)
        if not url:
            print(f"[warning] {fname} not found in release assets for {api_url}")
            continue
        try:
            print(f"[info] Downloading {fname} from {url} -> {target}")
            urllib.request.urlretrieve(url, target)
            print(f"[info] Saved {fname}")
        except Exception as e:
            print(f"[warning] Failed to download {fname}: {e}")


class install(_install):
    """Custom install command which ensures runtimes are present after installation."""

    def run(self):
        _install.run(self)
        # Use RUNTIME_RELEASE_TAG env var if present; otherwise default to package version
        tag = os.environ.get("RUNTIME_RELEASE_TAG")
        if not tag:
            tag = PACKAGE_VERSION
        # Normalize to 'v' prefix
        if not str(tag).startswith("v"):
            tag = f"v{tag}"
        download_runtimes_from_release(tag=tag)


setup(
    name="infinite_rl",
    version=PACKAGE_VERSION,
    packages=["infinite_rl", "infinite_rl.reward_functions", "infinite_rl.examples"],
    include_package_data=True,
    package_data={
        "infinite_rl": ["VERSION.txt"],
        "infinite_rl.examples": ["*.md"],
        "infinite_rl.reward_functions": ["*.py"],
        "infinite_rl.runtimes": ["*.wasm", "qwen3_local_cache/*"],
    },
    install_requires=[
        "wasmtime",
        "sympy",
        "antlr4-python3-runtime==4.11.1",
        "pycld2",
        "cantonesedetect",
    ],
    cmdclass={"install": install},
)
