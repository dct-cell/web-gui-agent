#!/bin/bash
set -e

echo "=== Cloning ShowUI ==="
if [ ! -d "third_party/ShowUI" ]; then
    mkdir -p third_party
    git clone https://github.com/showlab/ShowUI.git third_party/ShowUI
fi

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

echo "=== Installing ShowUI dependencies ==="
pip install -r third_party/ShowUI/requirements.txt

echo "=== Installing Playwright browsers ==="
playwright install chromium

echo "=== Done ==="
