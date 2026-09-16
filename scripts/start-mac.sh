#!/bin/bash
set -e

cd ~/Projects/nordic-ai-cup-2026

echo "Pulling latest changes..."
git pull --rebase

echo "Activating environment..."
source .venv/bin/activate

echo "Opening VS Code..."
code .
