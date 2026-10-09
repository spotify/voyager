# Load the runner's x64 Visual Studio toolchain without a JavaScript action.
$ErrorActionPreference = 'Stop'

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if ($LASTEXITCODE -ne 0 -or -not $installation) {
    throw 'Could not find a Visual Studio installation with the x64 C++ tools.'
}

$previousEnvironment = @{}
Get-ChildItem Env: | ForEach-Object { $previousEnvironment[$_.Name] = $_.Value }

& "$installation/Common7/Tools/Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    throw 'Visual Studio initialization did not put cl.exe on PATH.'
}

# Persist only values changed by Visual Studio for subsequent workflow steps.
Get-ChildItem Env: | Where-Object {
    $previousEnvironment[$_.Name] -cne $_.Value
} | ForEach-Object {
    "$($_.Name)=$($_.Value)" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
}
