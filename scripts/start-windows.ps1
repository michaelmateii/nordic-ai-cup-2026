Set-Location "$HOME\Projects\nordic-ai-cup-2026"

Write-Host "Pulling latest changes..."
git pull --rebase

Write-Host "Checking environment..."
& .\.venv\Scripts\python.exe scripts\check_env.py

Write-Host "Opening VS Code..."
code .
