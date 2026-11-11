param([ValidateSet('install', 'dev', 'build', 'test', 'typecheck')][string]$Task = 'dev')
$ErrorActionPreference = 'Stop'
$ui = Join-Path $PSScriptRoot '..\phase13-ui-original\artifacts\mockup-sandbox'
$portable = Join-Path $PSScriptRoot '..\.tmp'
if (-not (Get-Command node -ErrorAction SilentlyContinue) -and (Test-Path "$portable\node.exe")) {
    $env:PATH = "$portable;$env:PATH"
}
$npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
$npmCli = Join-Path $portable 'package\bin\npm-cli.js'
function Invoke-LocalNpm([string[]]$Arguments) {
    if ($npm) { & $npm.Source @Arguments }
    elseif (Test-Path $npmCli) { & node $npmCli @Arguments }
    else { throw 'Install Node.js 22 LTS (including npm), then reopen PowerShell.' }
}
Push-Location $ui
try {
    # Standalone npm manifest avoids the Replit monorepo's Linux-only overrides.
    if ($Task -eq 'install') { Invoke-LocalNpm @('ci', '--workspaces=false', '--no-audit', '--no-fund') }
    else { Invoke-LocalNpm @('run', $Task) }
    if ($LASTEXITCODE -ne 0) { throw "UI $Task failed ($LASTEXITCODE)" }
} finally { Pop-Location }
