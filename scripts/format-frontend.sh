#!/usr/bin/env bash
# format-frontend.sh - Auto-format all frontend files using Prettier
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "=============================="
echo "  Formatting Frontend Files"
echo "=============================="

# Install dev dependencies if node_modules is missing
if [ ! -d "node_modules" ]; then
    echo "Installing dev dependencies..."
    npm install
fi

echo ""
echo "Running Prettier on frontend files..."
npx prettier --write "frontend/**/*.{html,js,css}"

echo ""
echo "=============================="
echo "  Formatting complete!"
echo "=============================="
