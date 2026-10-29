# Frees port 8000, then starts the API in judge-only mode.
Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
$env:NLI_ENABLED = "0"
uvicorn src.api.main:app --port 8000