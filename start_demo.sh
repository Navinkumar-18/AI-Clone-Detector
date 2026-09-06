#!/usr/bin/env bash
# start_demo.sh — VoiceGuard demo launcher
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
#   chmod +x start_demo.sh
#   ./start_demo.sh

set -e

echo "=============================================="
echo "  VoiceGuard Demo Launcher"
echo "=============================================="
echo ""

# Step 1: Generate TLS certificates (idempotent)
echo "[1/3] Checking TLS certificates..."
python generate_cert.py
echo ""

# Step 2: Start the FastAPI backend in background
echo "[2/3] Starting VoiceGuard backend (HTTPS on port 8443)..."
python backend.py &
BACKEND_PID=$!
echo "  Backend PID: $BACKEND_PID"

# Wait for backend to be ready
echo "  Waiting for backend to start..."
sleep 8

# Quick health check
if curl -sk https://localhost:8443/health > /dev/null 2>&1; then
    echo "  ✅ Backend is running on https://localhost:8443"
else
    echo "  ⚠  Backend may still be loading the model. Proceeding with ngrok..."
fi
echo ""

# Step 3: Start ngrok
echo "[3/3] Starting ngrok tunnel..."
echo ""

# Check if ngrok is available
if ! command -v ngrok &> /dev/null; then
    echo "ERROR: ngrok is not installed or not on PATH."
    echo "Install from: https://ngrok.com/download"
    echo ""
    echo "Backend is still running at https://localhost:8443 (PID: $BACKEND_PID)"
    echo "Kill it with: kill $BACKEND_PID"
    exit 1
fi

echo "=============================================="
echo ""
echo "  ngrok is starting. Look for the 'Forwarding' line below."
echo "  Copy the https://xxxx.ngrok-free.app URL and paste it"
echo "  into the VoiceGuard app's Settings screen (gear icon)."
echo ""
echo "  Press Ctrl+C to stop everything."
echo ""
echo "=============================================="
echo ""

# Run ngrok in foreground (Ctrl+C stops it)
ngrok http https://localhost:8443

# Cleanup when ngrok is stopped
echo ""
echo "Stopping backend (PID: $BACKEND_PID)..."
kill $BACKEND_PID 2>/dev/null || true
echo "Done."
