#!/bin/bash

# Configuration - Update these paths if necessary
JAVY_VERSION="v8.0.0" # Check GitHub for latest
RUNTIME_DIR="infinite_rl/runtimes"
BUILD_SRC="build_src"
MICROPYTHON_URL="https://github.com/vmware-labs/webassembly-language-runtimes/releases/download/python%2F3.12.0%2B20231211-040d5a6/python-3.12.0.wasm"
# Local path for the javy binary
LOCAL_JAVY="$BUILD_SRC/javy"

# Ensure directories exist
mkdir -p "$RUNTIME_DIR"
mkdir -p "$BUILD_SRC"

# --- 1. Detect Architecture for Javy ---
OS_TYPE=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH_TYPE=$(uname -m)

if [ "$ARCH_TYPE" == "x86_64" ]; then
    JAVY_ARCH="x86_64"
elif [ "$ARCH_TYPE" == "arm64" ] || [ "$ARCH_TYPE" == "aarch64" ]; then
    JAVY_ARCH="arm"
else
    echo "Unsupported architecture: $ARCH_TYPE"
    exit 1
fi

if [ "$OS_TYPE" == "darwin" ]; then
    JAVY_OS="macos"
elif [ "$OS_TYPE" == "linux" ]; then
    JAVY_OS="linux"
else
    echo "Unsupported OS: $OS_TYPE"
    exit 1
fi

JAVY_OS=$([ "$OS_TYPE" = "darwin" ] && echo "macos" || echo "linux")

echo "Detected platform: $JAVY_OS / $JAVY_ARCH"
echo "Javy version to download: $JAVY_VERSION"

JAVY_ASSET="javy-$JAVY_ARCH-$JAVY_OS-$JAVY_VERSION"
JAVY_URL="https://github.com/bytecodealliance/javy/releases/download/$JAVY_VERSION/$JAVY_ASSET.gz"

echo "Javy download URL: $JAVY_URL"

if [ ! -f "$LOCAL_JAVY" ]; then
    echo "Attempting to download Javy asset: $JAVY_ASSET"

    # Helper: try downloading and ensuring the binary matches expected format
    try_download() {
        local url="$1"
        local out="$2"
        echo "Downloading from: $url"
        # Retry a few times in case of transient network issues
        for i in 1 2 3; do
            if curl -fL "$url" -o javy_tmp.gz; then
                gunzip -c javy_tmp.gz > "$out" || true
                chmod +x "$out" || true
                rm -f javy_tmp.gz
                return 0
            else
                echo "Download attempt $i failed, retrying..."
                sleep 1
            fi
        done
        return 1
    }

    if try_download "$JAVY_URL" "$LOCAL_JAVY"; then
        echo "Downloaded candidate Javy to $LOCAL_JAVY"
    else
        echo "Initial download failed for $JAVY_URL"
    fi

    # Validate binary format (ELF on Linux, Mach-O on macOS)
    file_type=$(file -b "$LOCAL_JAVY" 2>/dev/null || true)
    echo "Downloaded javy file type: $file_type"

    ok=false
    if [ "$JAVY_OS" = "linux" ]; then
        if echo "$file_type" | grep -qi 'elf'; then
            ok=true
        fi
    elif [ "$JAVY_OS" = "macos" ]; then
        if echo "$file_type" | grep -qi 'mach-o'; then
            ok=true
        fi
    fi

    # If validation failed, try to discover the right asset via GitHub API and re-download
    if [ "$ok" = false ]; then
        echo "Downloaded Javy binary does not match expected platform ($JAVY_OS/$JAVY_ARCH). Trying to find a suitable asset via GitHub Releases API..."
        API_URL="https://api.github.com/repos/bytecodealliance/javy/releases/tags/$JAVY_VERSION"
        echo "Querying: $API_URL"
        assets_json=$(curl -sL "$API_URL" || true)
        # Try a few heuristics to find an asset name that contains arch and os
        candidates=$(echo "$assets_json" | grep -o '"browser_download_url": *"[^"]*"' | sed -E 's/"browser_download_url": *"([^"]*)"/\1/' | grep -i "$JAVY_ARCH" | grep -i "$JAVY_OS" || true)

        if [ -n "$candidates" ]; then
            echo "Found candidate assets:"
            echo "$candidates"
            success=false
            while read -r url; do
                [ -z "$url" ] && continue
                echo "Trying candidate: $url"
                if try_download "$url" "$LOCAL_JAVY"; then
                    file_type=$(file -b "$LOCAL_JAVY" 2>/dev/null || true)
                    echo "Candidate file type: $file_type"
                    if [ "$JAVY_OS" = "linux" ] && echo "$file_type" | grep -qi 'elf'; then
                        success=true
                        break
                    elif [ "$JAVY_OS" = "macos" ] && echo "$file_type" | grep -qi 'mach-o'; then
                        success=true
                        break
                    fi
                fi
            done <<EOF
$candidates
EOF

            if [ "$success" = true ]; then
                ok=true
            else
                echo "No suitable asset found from release matching $JAVY_ARCH/$JAVY_OS"
            fi
        else
            echo "No matching assets found in release $JAVY_VERSION"
        fi
    fi

    if [ "$ok" = false ]; then
        echo "ERROR: Could not obtain a compatible Javy binary for $JAVY_ARCH/$JAVY_OS."
        echo "Downloaded file info:"
        ls -lh "$LOCAL_JAVY" || true
        file -b "$LOCAL_JAVY" || true
        echo "Please run build locally or set up a compatible javy binary at $LOCAL_JAVY"
        exit 1
    fi

    echo "✓ Javy downloaded and validated locally to $BUILD_SRC"
fi

# --- 3. Bundle JS Runner ---
echo "[1/3] Bundling JS Runner..."
esbuild "$BUILD_SRC/runner.js" --bundle --outfile="$BUILD_SRC/bundled_runner.js" --format=esm

# --- 4. Build JS Runner ---
echo "[2/3] Compiling JS Runner..."
"$LOCAL_JAVY" build "$BUILD_SRC/bundled_runner.js" -o "$RUNTIME_DIR/puzzle_js.wasm"

# --- 5. Fetch MicroPython ---
echo "[3/3] Skipping MicroPython (no longer needed for puzzles)"

echo "--- All Runtimes Ready in $RUNTIME_DIR ---"
ls -lh "$RUNTIME_DIR"