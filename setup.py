import os
import json
import urllib.request
from setuptools import setup
from setuptools.command.install import install as _install

RUNTIME_FILES = ["universal_js.wasm", "micropython.wasm"]
GITHUB_REPO = os.environ.get("RUNTIME_GITHUB_REPO", "hon9kon9ize/infinite-rl")


def download_runtimes_from_release(tag=None, dest_dir="infinite_rl/runtimes"):
    os.makedirs(dest_dir, exist_ok=True)
    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/{'tags/' + tag if tag else 'latest'}"

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

    for fname in RUNTIME_FILES:
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
        # Try to download runtimes using optional environment override tag
        tag = os.environ.get("RUNTIME_RELEASE_TAG")
        download_runtimes_from_release(tag=tag)


setup(
    name="infinite_rl",
    version="0.1.13",
    packages=["infinite_rl", "infinite_rl.reward_functions", "infinite_rl.examples"],
    include_package_data=True,
    package_data={
        "infinite_rl.examples": ["*.md"],
        "infinite_rl.reward_functions": ["*.py"],
        "infinite_rl.runtimes": ["*.wasm"],
    },
    install_requires=["wasmtime"],
    cmdclass={"install": install},
)
