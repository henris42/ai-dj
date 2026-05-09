# Launches AI DJ on Windows. Boots Qdrant in WSL if it isn't already running,
# waits until it accepts connections, then starts the GUI. Designed to be
# invoked from a Start menu shortcut (see install-shortcut.ps1) — no console
# window, errors surface as a Windows message box instead of vanishing.

$ErrorActionPreference = "Stop"
$logDir = Join-Path $PSScriptRoot "data"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
$logPath = Join-Path $logDir "launch.log"

function Log([string]$msg) {
    "$([DateTime]::Now.ToString('HH:mm:ss'))  $msg" | Out-File -FilePath $logPath -Append -Encoding utf8
}

function Show-Error([string]$summary, [string]$detail = "") {
    $full = $summary
    if ($detail) { $full += "`n`n" + $detail }
    $full += "`n`nFull log: $logPath"
    Add-Type -AssemblyName System.Windows.Forms | Out-Null
    [System.Windows.Forms.MessageBox]::Show($full, "AI DJ", "OK", "Error") | Out-Null
}

function Wait-Qdrant([int]$timeoutSeconds = 20) {
    $deadline = (Get-Date).AddSeconds($timeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:6333/" -UseBasicParsing -TimeoutSec 2
            if ($r.StatusCode -eq 200) { return $true }
        } catch { Start-Sleep -Milliseconds 500 }
    }
    return $false
}

function Find-Uv {
    foreach ($p in @(
        (Join-Path $env:LOCALAPPDATA "Programs\uv\uv.exe"),
        (Join-Path $env:USERPROFILE ".local\bin\uv.exe"),
        "C:\Users\$env:USERNAME\.local\bin\uv.exe"
    )) {
        if (Test-Path $p) { return $p }
    }
    $cmd = Get-Command uv.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

try {
    Set-Location -Path $PSScriptRoot
    Log "==== launch ===="
    Log "cwd: $(Get-Location)"

    # 1. Find uv.exe.
    $uv = Find-Uv
    if (-not $uv) {
        Show-Error "uv.exe not found." "Install uv from https://github.com/astral-sh/uv and put it on PATH."
        exit 2
    }
    Log "uv: $uv"

    # 2. Make sure Qdrant is running in WSL. Idempotent — qdrant_up.sh is a no-op
    #    if the container is already up.
    Log "starting Qdrant via WSL..."
    $wslOut = & wsl.exe bash -lc "/home/hs/ai-dj/scripts/qdrant_up.sh" 2>&1
    Log ("wsl out: " + ($wslOut -join " | "))

    if (-not (Wait-Qdrant -timeoutSeconds 20)) {
        Show-Error "Qdrant never came up on http://127.0.0.1:6333" ($wslOut -join "`n")
        exit 3
    }
    Log "Qdrant: reachable"

    # 3. Launch the GUI. Stream stdout/stderr to the log so a crash isn't silent.
    Log "starting GUI..."
    & $uv run python -m ai_dj.gui.app *>> $logPath
    $code = $LASTEXITCODE
    Log "GUI exited with code $code"
    if ($code -ne 0) {
        Show-Error "AI DJ exited unexpectedly (code $code)."
        exit $code
    }
    exit 0
} catch {
    Log ("FATAL: " + $_.Exception.Message)
    Log ($_ | Out-String)
    Show-Error "Couldn't start AI DJ" $_.Exception.Message
    exit 1
}
