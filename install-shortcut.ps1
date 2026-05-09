# Creates a clickable "AI DJ" shortcut on the Windows desktop that launches
# launch.ps1 with no visible console window. Run this once on Windows:
#
#     powershell.exe -ExecutionPolicy Bypass -File install-shortcut.ps1
#
# (or right-click → Run with PowerShell). It just writes a .lnk to the
# Desktop — no install, nothing system-wide.

$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$icon  = Join-Path $projectRoot "assets\icon.ico"
$entry = Join-Path $projectRoot "launch.ps1"

if (-not (Test-Path $entry)) { throw "launch.ps1 not found at $entry" }
if (-not (Test-Path $icon))  { throw "icon not found at $icon" }

$desktop = [Environment]::GetFolderPath("Desktop")
$lnkPath = Join-Path $desktop "AI DJ.lnk"

$wsh = New-Object -ComObject WScript.Shell
$lnk = $wsh.CreateShortcut($lnkPath)
$lnk.TargetPath       = "powershell.exe"
$lnk.Arguments        = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$entry`""
$lnk.WorkingDirectory = $projectRoot
$lnk.IconLocation     = "$icon,0"
$lnk.Description      = "AI DJ"
$lnk.Save()

Write-Host "Created: $lnkPath"
Write-Host "Double-click it on the Desktop to launch."
