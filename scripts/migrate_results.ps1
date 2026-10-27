$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$taskMappings = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'result_migration.json') -Raw | ConvertFrom-Json
$taskAudit = Join-Path $taskRoot 'experiments/results/migration.jsonl'
foreach ($taskPair in $taskMappings) {
    $taskSource = [IO.Path]::GetFullPath((Join-Path $taskRoot $taskPair[0]))
    $taskDestination = [IO.Path]::GetFullPath((Join-Path $taskRoot $taskPair[1]))
    if (-not $taskSource.StartsWith($taskRoot + '\') -or -not $taskDestination.StartsWith($taskRoot + '\experiments\results\')) {
        throw 'Migration path escapes the workspace/experiments/results boundary'
    }
    if (-not (Test-Path -LiteralPath $taskSource)) { continue }
    if (Test-Path -LiteralPath $taskDestination) { throw "Destination already exists: $taskDestination" }
    $taskFiles = @(Get-ChildItem -LiteralPath $taskSource -File -Recurse)
    $taskBytes = ($taskFiles | Measure-Object -Property Length -Sum).Sum
    New-Item -ItemType Directory -Path (Split-Path -Parent $taskDestination) -Force | Out-Null
    Move-Item -LiteralPath $taskSource -Destination $taskDestination
    $taskAfter = @(Get-ChildItem -LiteralPath $taskDestination -File -Recurse)
    if ($taskAfter.Count -ne $taskFiles.Count -or ($taskAfter | Measure-Object -Property Length -Sum).Sum -ne $taskBytes) {
        throw "Migration inventory mismatch: $taskDestination"
    }
    @{source=$taskPair[0]; destination=$taskPair[1]; files=$taskFiles.Count; bytes=$taskBytes} |
        ConvertTo-Json -Compress | Add-Content -LiteralPath $taskAudit -Encoding UTF8
    Write-Output "$($taskPair[0]) -> $($taskPair[1]) ($($taskFiles.Count) files)"
}
