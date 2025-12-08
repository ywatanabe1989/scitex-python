#!/bin/bash
# Build Sphinx documentation and optionally deploy to scitex-cloud
# Usage: ./build_docs.sh [--deploy]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
GALLERY_SRC="/tmp/test_gallery"
GALLERY_DST="$SCRIPT_DIR/_static/gallery"
CLOUD_DIR="$HOME/proj/scitex-cloud/static/docs"

# Generate gallery images if not exist
if [ ! -d "$GALLERY_SRC" ] || [ -z "$(ls -A $GALLERY_SRC 2>/dev/null)" ]; then
    echo "Generating gallery images..."
    python3 -c "import scitex as stx; stx.plt.gallery.generate('$GALLERY_SRC')"
fi

# Copy gallery images to sphinx static
echo "Copying gallery images..."
rm -rf "$GALLERY_DST"
cp -r "$GALLERY_SRC" "$GALLERY_DST"

# Build HTML documentation
echo "Building HTML documentation..."
sphinx-build -b html "$SCRIPT_DIR" "$BUILD_DIR/html"

echo "Documentation built: $BUILD_DIR/html/index.html"

# Deploy to scitex-cloud if requested
if [ "$1" = "--deploy" ]; then
    if [ -d "$HOME/proj/scitex-cloud" ]; then
        echo "Deploying to scitex-cloud..."
        mkdir -p "$CLOUD_DIR"
        rm -rf "$CLOUD_DIR"/*
        cp -r "$BUILD_DIR/html/"* "$CLOUD_DIR/"
        echo "Deployed to: $CLOUD_DIR"
    else
        echo "Warning: scitex-cloud directory not found at $HOME/proj/scitex-cloud"
        exit 1
    fi
fi

echo "Done!"
