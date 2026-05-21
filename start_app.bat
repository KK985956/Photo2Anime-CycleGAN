@echo off
set "ROOT=%~dp0"
powershell -ExecutionPolicy Bypass -NoProfile -File "%ROOT%start_app.ps1"
