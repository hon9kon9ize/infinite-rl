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

if [ ! -f "$LOCAL_JAVY" ]; then
    JAVY_ASSET="javy-$JAVY_ARCH-$JAVY_OS-$JAVY_VERSION"
    JAVY_URL="https://github.com/bytecodealliance/javy/releases/download/$JAVY_VERSION/$JAVY_ASSET.gz"

    echo $JAVY_URL
    
    echo "Downloading Javy to $LOCAL_JAVY..."
    curl -fL "$JAVY_URL" -o javy_tmp.gz
    gunzip -c javy_tmp.gz > "$LOCAL_JAVY"
    chmod +x "$LOCAL_JAVY"
    rm javy_tmp.gz
    echo "✓ Javy downloaded locally to $BUILD_SRC"
fi

# --- 3. Build JS Runner ---
echo "[1/2] Compiling JS Runner..."
"$LOCAL_JAVY" build "$BUILD_SRC/runner.js" -o "$RUNTIME_DIR/universal_js.wasm"

# --- 4. Fetch MicroPython ---
echo "[2/2] Fetching MicroPython WASI..."
if [ ! -f "$RUNTIME_DIR/micropython.wasm" ]; then
    curl -L "$MICROPYTHON_URL" -o "$RUNTIME_DIR/micropython.wasm"
fi

echo "--- All Runtimes Ready in $RUNTIME_DIR ---"
ls -lh "$RUNTIME_DIR"