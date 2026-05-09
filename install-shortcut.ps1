# Adds an "AI DJ" entry to the Windows Start menu (with the planet icon) that
# launches launch.ps1 silently. Run this once on Windows:
#
#     powershell.exe -ExecutionPolicy Bypass -File install-shortcut.ps1
#
# After running, AI DJ shows up in the Start menu (searchable) and can be
# right-click → Pin to taskbar from there. No system-wide install — the .lnk
# lives under your own Start Menu folder.

$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$icon  = Join-Path $projectRoot "assets\icon.ico"
$entry = Join-Path $projectRoot "launch.ps1"

Write-Host "project: $projectRoot"
if (-not (Test-Path $entry)) { throw "launch.ps1 not found at $entry" }
if (-not (Test-Path $icon))  { throw "icon not found at $icon" }

$startMenu = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
$lnkPath   = Join-Path $startMenu "AI DJ.lnk"

$wsh = New-Object -ComObject WScript.Shell
$lnk = $wsh.CreateShortcut($lnkPath)
$lnk.TargetPath       = "powershell.exe"
$lnk.Arguments        = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$entry`""
$lnk.WorkingDirectory = $projectRoot
$lnk.IconLocation     = "$icon,0"
$lnk.Description      = "AI DJ"
$lnk.Save()

Write-Host ""
Write-Host "Created Start menu entry: $lnkPath"
Write-Host "Open Start, type 'AI DJ' — right-click it to pin to taskbar."
