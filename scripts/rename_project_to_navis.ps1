$ErrorActionPreference = 'Stop'

$source = 'D:\video_compression'
$destination = 'D:\NAVIS'

if (-not (Test-Path -LiteralPath $source -PathType Container)) {
    throw "Project directory does not exist: $source"
}
if (Test-Path -LiteralPath $destination) {
    throw "Destination already exists: $destination"
}
if (-not (Test-Path -LiteralPath (Join-Path $source 'src\common') -PathType Container)) {
    throw 'The common package is missing; refusing to move an unexpected directory.'
}
if (-not (Test-Path -LiteralPath (Join-Path $source 'src\scene_factorization\scene_factorization_output') -PathType Container)) {
    throw 'The scene output is missing; refusing to move an incomplete project.'
}

Set-Location -LiteralPath 'D:\'
Move-Item -LiteralPath $source -Destination $destination

$python = Join-Path $destination '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $python) {
    uv pip install --python $python --reinstall --no-deps --editable $destination
    if ($LASTEXITCODE -ne 0) {
        throw 'Project was renamed, but the editable environment needs to be refreshed.'
    }
}

Write-Host "NAVIS project is now at $destination"
