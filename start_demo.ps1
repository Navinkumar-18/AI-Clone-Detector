# start_demo.ps1 — VoiceGuard demo launcher (PowerShell)
#
# Starts the FastAPI backend (HTTPS on 8443), then exposes it via ngrok.
# The resulting public URL can be pasted into the app's settings screen
# so the demo works from ANY network (venue WiFi, mobile data, etc).
#
# Prerequisites:
#   - Python environment with project requirements installed
#   - ngrok installed and authenticated (ngrok authtoken <TOKEN>)
#
# Usage:
#   .\start_demo.ps1

$ErrorActionPreference = "Stop"

Write-Host "=============================================="
Write-Host "  VoiceGuard Demo Launcher"
Write-Host "=============================================="
Write-Host ""

# Step 1: Generate TLS certificates (idempotent)
Write-Host "[1/3] Checking TLS certificates..."
python generate_cert.py
Write-Host ""

# Step 2: Start the FastAPI backend in background
Write-Host "[2/3] Starting VoiceGuard backend (HTTPS on port 8443)..."
$backendJob = Start-Process -FilePath "python" -ArgumentList "backend.py" -PassThru -NoNewWindow
Write-Host "  Backend PID: $($backendJob.Id)"

# Wait for backend to be ready
Write-Host "  Waiting for backend to start..."
Start-Sleep -Seconds 8

# Quick health check (ignore cert errors for self-signed)
try {
    [System.Net.ServicePointManager]::ServerCertificateValidationCallback = { $true }
    $response = Invoke-WebRequest -Uri "https://localhost:8443/health" -UseBasicParsing -TimeoutSec 5 -SkipCertificateCheck
    Write-Host "  ✅ Backend is running on https://localhost:8443"
} catch {
    Write-Host "  ⚠  Backend may still be loading the model. Proceeding with ngrok..."
}
Write-Host ""

# Step 3: Start ngrok
Write-Host "[3/3] Starting ngrok tunnel..."
Write-Host ""

# Check if ngrok is available
$ngrokPath = Get-Command ngrok -ErrorAction SilentlyContinue
if (-not $ngrokPath) {
    Write-Host "ERROR: ngrok is not installed or not on PATH."
    Write-Host "Install from: https://ngrok.com/download"
    Write-Host ""
    Write-Host "Backend is still running at https://localhost:8443 (PID: $($backendJob.Id))"
    Write-Host "Kill it with: Stop-Process -Id $($backendJob.Id)"
    exit 1
}

Write-Host "=============================================="
Write-Host ""
Write-Host "  ngrok is starting. Look for the 'Forwarding' line below."
Write-Host "  Copy the https://xxxx.ngrok-free.app URL and paste it"
Write-Host "  into the VoiceGuard app's Settings screen (gear icon)."
Write-Host ""
Write-Host "  Press Ctrl+C to stop everything."
Write-Host ""
Write-Host "=============================================="
Write-Host ""

try {
    # Run ngrok in foreground (Ctrl+C stops it)
    ngrok http https://localhost:8443
} finally {
    # Cleanup when ngrok is stopped
    Write-Host ""
    Write-Host "Stopping backend (PID: $($backendJob.Id))..."
    Stop-Process -Id $backendJob.Id -Force -ErrorAction SilentlyContinue
    Write-Host "Done."
}
