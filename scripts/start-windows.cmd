@echo off
cd /d %USERPROFILE%\Projects\nordic-ai-cup-2026
git pull --rebase
.venv\Scripts\python.exe scripts\check_env.py
code .
