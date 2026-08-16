# Builds the React client into a standalone Windows executable.
# Output: React_Client\release\CarCalClient.exe + config.json
# Run on a machine with Node.js and Python installed; the resulting
# release\ folder can then be copied to any machine and run standalone.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

Write-Host "Building React app..."
npm install
npm run build

$venvPath = Join-Path $root "server\.venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "Creating packaging venv..."
    python -m venv $venvPath
}
$pip = Join-Path $venvPath "Scripts\pip.exe"
$pyinstaller = Join-Path $venvPath "Scripts\pyinstaller.exe"

& $pip install --quiet --upgrade pyinstaller

Write-Host "Packaging exe..."
$distPath = Join-Path $root "dist"
& $pyinstaller --onefile --name CarCalClient `
    --add-data "$distPath;dist" `
    --distpath release `
    --workpath build\pyinstaller `
    --specpath build\pyinstaller `
    server\serve.py

# Seed release/config.json from the same config.json used by `npm run dev`,
# so the exe points at the same API endpoint by default. It's copied, not
# linked — once deployed, edit release\config.json (or the copy on the
# target machine) independently without touching this one.
Copy-Item -Path (Join-Path $root "config.json") -Destination (Join-Path $root "release\config.json") -Force

Write-Host "Done. Copy the release\ folder to any machine and run CarCalClient.exe."
