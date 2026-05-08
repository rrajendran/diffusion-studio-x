#!/bin/bash

# Diffusion Studio X Development Startup Script
# This script provides convenient options to start the development environment

set -e  # Exit on any error

echo "🚀 Diffusion Studio X Development Startup"
echo "========================================"

# Function to kill existing processes
kill_existing_processes() {
    echo "🛑 Killing any existing processes on ports 5173, 3001, 8001..."
    npm run kill-ports
    echo "✅ Processes killed"
}

# Function to start full development server
start_full_dev() {
    echo "🔄 Starting full development server (Vite + Bridge + Image Server)..."
    echo "   - Vite UI: http://localhost:5173"
    echo "   - Bridge Server: http://localhost:3001"
    echo "   - Image Server: http://localhost:8001"
    echo ""
    npm run dev
}

# Function to start Tauri desktop app
start_tauri() {
    echo "🖥️  Starting Tauri desktop application..."
    echo "   This will build the bridge server and launch the desktop app"
    echo ""
    npm run tauri:dev
}

# Function to start image server only
start_image_server() {
    echo "🖼️  Starting image server only..."
    echo "   - Image Server: http://localhost:8001"
    echo ""
    npm run dev:images
}

# Main menu
echo "Choose what to start:"
echo "1) Full Development Server (recommended)"
echo "2) Image Server Only"
echo "3) Tauri Desktop App"
echo "4) Kill existing processes only"
echo ""

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        kill_existing_processes
        echo ""
        start_full_dev
        ;;
    2)
        kill_existing_processes
        echo ""
        start_image_server
        ;;
    3)
        kill_existing_processes
        echo ""
        start_tauri
        ;;
    4)
        kill_existing_processes
        echo "✅ Done! No servers started."
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again and choose 1, 2, 3, or 4."
        exit 1
        ;;
esac