#!/usr/bin/env python3
"""Download runtime WASM assets from GitHub release tag.

Usage:
  python scripts/download_runtimes.py [--tag vX.Y.Z] [--repo owner/repo]

Environment:
  RUNTIME_RELEASE_TAG - override tag
  RUNTIME_GITHUB_REPO  - override repo (default hon9kon9ize/infinite-rl)
"""
import argparse
import json
import os
import sys
from urllib.request import urlopen, urlretrieve

RUNTIME_FILES = [
    "universal_js.wasm",
    "micropython.wasm",
    "qwen3_embed.wasm",
    "qwen3_local_cache.zip",
]
DEFAULT_REPO = os.environ.get("RUNTIME_GITHUB_REPO", "hon9kon9ize/infinite-rl")


def read_version():
    try:
        with open("VERSION.txt", "r") as f:
            v = f.read().strip()
            return v if v.startswith("v") else f"v{v}"
    except Exception:
        return "v0.0.0"


def download_releases(tag, repo=DEFAULT_REPO, dest_dir="infinite_rl/runtimes"):
    os.makedirs(dest_dir, exist_ok=True)
    api_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    try:
        with urlopen(api_url) as resp:
            release_info = json.load(resp)
    except Exception as e:
        print(f"[error] Could not fetch release info: {e}")
        return 1

    assets = {
        a["name"]: a["browser_download_url"] for a in release_info.get("assets", [])
    }
    exit_code = 0
    for fname in RUNTIME_FILES:
        target = os.path.join(dest_dir, fname)
        if os.path.exists(target):
            print(f"[info] {fname} already exists at {target}, skipping")
            continue
        url = assets.get(fname)
        if not url:
            print(f"[warn] {fname} not found in release {tag}")
            exit_code = 1
            continue
        try:
            print(f"[info] Downloading {fname} from {url} -> {target}")
            urlretrieve(url, target)
            print(f"[info] Saved {fname}")
        except Exception as e:
            print(f"[error] Failed to download {fname}: {e}")
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, default=os.environ.get("RUNTIME_RELEASE_TAG"))
    p.add_argument(
        "--repo", type=str, default=os.environ.get("RUNTIME_GITHUB_REPO", DEFAULT_REPO)
    )
    p.add_argument("--dest", type=str, default="infinite_rl/runtimes")
    args = p.parse_args()
    tag = args.tag or read_version()
    if not tag.startswith("v"):
        tag = f"v{tag}"
    rc = download_releases(tag, repo=args.repo, dest_dir=args.dest)
    if rc != 0:
        print("Some downloads failed or were missing; check logs")
    sys.exit(rc)
