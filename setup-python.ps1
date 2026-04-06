param(
    [switch]$Run,
    [string]$ExecutionProvider = "auto"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "The Python launcher 'py' was not found. Install Python 3.11 first."
}

$pythonVersion = & py -3.11 --version 2>$null
if (-not $pythonVersion) {
    throw "Python 3.11 was not found. Install Python 3.11 first."
}

$venvPython = Join-Path $PWD ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Creating Python virtual environment..."
    & py -3.11 -m venv .venv
}

Write-Host "Installing Python dependencies..."
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r .\python\requirements.txt

if ($Run) {
    & $venvPython .\python\live_face_swap.py --execution-provider=$ExecutionProvider
}
