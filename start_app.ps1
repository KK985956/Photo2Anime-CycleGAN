$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$Port = 7860
$InUse = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue | Select-Object -First 1
if ($InUse) {
    $Port = 7861
}

& $Python "web_demo\app.py" `
    --checkpoint "auto" `
    --outputs "outputs\app" `
    --host "127.0.0.1" `
    --port $Port `
    --open
