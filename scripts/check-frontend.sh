#!/usr/bin/env bash
# check-frontend.sh - Run all frontend code quality checks
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "=============================="
echo "  Frontend Quality Checks"
echo "=============================="

# Install dev dependencies if node_modules is missing
if [ ! -d "node_modules" ]; then
    echo "Installing dev dependencies..."
    npm install
fi

echo ""
echo "Checking Prettier formatting..."
npx prettier --check "frontend/**/*.{html,js,css}"

echo ""
echo "=============================="
echo "  All checks passed!"
echo "=============================="
