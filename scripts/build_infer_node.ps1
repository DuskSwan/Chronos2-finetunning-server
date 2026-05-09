param(
  [string]$PythonExe = ".\.venv\Scripts\python.exe",
  [string]$DistDir = "dist",
  [switch]$Clean
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

if (-not (Test-Path -LiteralPath $PythonExe)) {
  throw "Python executable not found: $PythonExe"
}

$pyInstallerArgs = @(
  "-m", "PyInstaller",
  "--noconfirm",
  "--onedir",
  "--name", "chronos_infer_node",
  "--distpath", $DistDir,
  "--collect-all", "chronos",
  "--collect-all", "transformers",
  "--collect-all", "accelerate",
  "--collect-all", "peft",
  "--collect-all", "tokenizers",
  "--hidden-import", "torch",
  "--hidden-import", "pandas",
  "--hidden-import", "numpy",
  "--hidden-import", "zmq",
  "app/cli/infer_node_cli.py"
)

if ($Clean) {
  $pyInstallerArgs = @("-m", "PyInstaller", "--clean") + $pyInstallerArgs[2..($pyInstallerArgs.Length - 1)]
}

Write-Host "[build] install pyinstaller"
& $PythonExe -m pip --version | Out-Null
if ($LASTEXITCODE -ne 0) {
  Write-Host "[build] pip not found, running ensurepip"
  & $PythonExe -m ensurepip --upgrade
  if ($LASTEXITCODE -ne 0) {
    throw "Failed to bootstrap pip in selected Python: $PythonExe"
  }
}

& $PythonExe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
  throw "Failed to upgrade pip"
}

& $PythonExe -m pip install pyinstaller
if ($LASTEXITCODE -ne 0) {
  throw "Failed to install PyInstaller"
}

Write-Host "[build] pyinstaller start"
& $PythonExe @pyInstallerArgs
if ($LASTEXITCODE -ne 0) {
  throw "PyInstaller failed"
}

$exePath = Join-Path -Path $repoRoot -ChildPath (Join-Path $DistDir "chronos_infer_node\chronos_infer_node.exe")
if (-not (Test-Path -LiteralPath $exePath)) {
  throw "Build finished but exe not found: $exePath"
}

Write-Host "[build] success: $exePath"
