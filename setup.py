import os
import json
import urllib.request
from setuptools import setup
from setuptools.command.install import install as _install

RUNTIME_FILES = [
    "universal_js.wasm",
    "micropython.wasm",
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

    Behavior:
    - If `tag` is provided, that tag is used (tag may be like 'vX.Y.Z' or 'runtimes-vX.Y.Z').
    - If `tag` is None, the function will attempt to discover the latest release that
      contains runtimes assets by enumerating releases and selecting the first
      release whose `tag_name` starts with `runtimes-`.
    - If discovery fails, it falls back to using the package version.
    """
    # If no explicit tag provided, try to discover the latest runtimes release
    if not tag:
        try:
            api_releases = f"https://api.github.com/repos/{GITHUB_REPO}/releases"
            req = urllib.request.Request(api_releases)
            gh_token = os.environ.get("GITHUB_TOKEN")
            if gh_token:
                req.add_header("Authorization", f"token {gh_token}")
            with urllib.request.urlopen(req) as resp:
                releases = json.load(resp)

            for rel in releases:
                t = rel.get("tag_name", "")
                if t.startswith("runtimes-"):
                    tag = t
                    print(f"[info] Found runtimes release tag: {tag}")
                    break
        except Exception as e:
            print(f"[warning] Could not enumerate releases to find runtimes tag: {e}")

    # If still no tag discovered, fall back to the package version
    if not tag:
        tag = PACKAGE_VERSION

    # Normalize tag to expected runtimes tag name when necessary
    if not str(tag).startswith("runtimes-v"):
        tag = f"runtimes-{tag}"

    os.makedirs(dest_dir, exist_ok=True)
    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/tags/{tag}"

    try:
        req = urllib.request.Request(api_url)
        gh_token = os.environ.get("GITHUB_TOKEN")
        if gh_token:
            req.add_header("Authorization", f"token {gh_token}")
        with urllib.request.urlopen(req) as resp:
            release_info = json.load(resp)
    except Exception as e:
        print(f"[warning] Could not fetch release info from {api_url}: {e}")
        return

    assets = {
        a.get("name"): a.get("browser_download_url")
        for a in release_info.get("assets", [])
    }

    # Print resolved URLs and available assets for debugging
    print(f"[info] Resolved release tag: {tag}")
    print(f"[info] Available release asset names: {sorted(list(assets.keys()))}")
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
        # If RUNTIME_RELEASE_TAG is explicitly set, use it. Otherwise, allow
        # download_runtimes_from_release() to try discovering the latest runtimes release.
        tag = os.environ.get("RUNTIME_RELEASE_TAG")
        if tag:
            # If the user passed a bare version (e.g., '1.2.3'), normalize to 'v' prefix.
            if not (str(tag).startswith("v") or str(tag).startswith("runtimes-")):
                tag = f"v{tag}"
            download_runtimes_from_release(tag=tag)
        else:
            # No explicit tag: discover latest runtimes release (falls back to PACKAGE_VERSION)
            download_runtimes_from_release()


# Ensure runtimes are downloaded when building wheels from git URLs.
# pip builds a wheel (using the build backend) when installing from a git URL,
# so we hook into the build_py command to fetch runtimes at build time so that
# the resulting wheel contains the wasm files.
try:
    from setuptools.command.build_py import build_py as _build_py

    class build_py(_build_py):
        def run(self):
            # Try to download runtimes and fail the build if essential files are missing.
            try:
                download_runtimes_from_release()
            except Exception as e:
                print(f"[warning] Failed to download runtimes during build_py: {e}")

            # Verify that at least one of the core runtime files is present; fail fast
            dest_dir = os.path.join(
                os.path.dirname(__file__), "infinite_rl", "runtimes"
            )
            found = []
            try:
                for fname in RUNTIME_FILES:
                    path = os.path.join(dest_dir, fname)
                    if os.path.exists(path):
                        found.append(fname)
            except Exception:
                found = []

            if not found:
                # Provide helpful diagnostic output and fail the build so CI surfaces the issue
                print("[error] Runtimes not found after build-time download attempt.")
                try:
                    print("[error] Target runtimes directory contents:")
                    print(sorted(os.listdir(dest_dir)))
                except Exception:
                    print("[error] Could not list runtimes directory (may not exist)")
                raise RuntimeError(
                    "Required runtime wasm files were not downloaded during build. "
                    "Set GITHUB_TOKEN in the environment to avoid rate limits or ensure the release 'runtimes-*' exists and contains assets."
                )

            print(f"[info] Runtimes present: {found}")
            super().run()

except Exception:
    build_py = None

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
        "infinite_rl": ["VERSION.txt"],
        "infinite_rl.examples": ["*.md"],
        "infinite_rl.reward_functions": ["*.py"],
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
        k: v
        for k, v in {"install": install, "build_py": build_py}.items()
        if v is not None
    },
)
