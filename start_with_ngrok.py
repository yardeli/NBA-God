#!/usr/bin/env python3
"""
Start NBA-God with ngrok tunnel
================================
Launches the NBA-God server on port 5052 and creates a public ngrok tunnel
"""

import subprocess
import time
import sys
import os
import socket
from pathlib import Path

def is_port_open(port):
    """Check if a port is listening"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result == 0

# Start the Flask server
print("🚀 Starting NBA-God server on port 5052...")
server_proc = subprocess.Popen(
    [sys.executable, "web/server.py"],
    cwd=Path(__file__).parent,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

# Wait for server to start and be ready
print("⏳ Waiting for server to start...")
for attempt in range(30):
    time.sleep(1)
    if is_port_open(5052):
        print("✅ NBA-God server is running on port 5052!")
        break
    print(f"  Attempt {attempt + 1}/30...")
else:
    print("❌ Server failed to start!")
    sys.exit(1)

# Start ngrok
print("🌐 Starting ngrok tunnel...")
try:
    ngrok_path = r"C:\Users\yarden\AppData\Roaming\npm\ngrok.cmd"
    ngrok_proc = subprocess.Popen(
        [ngrok_path, "http", "5052", "--log=stdout"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Read ngrok output to find the public URL
    print("\n" + "="*60)
    print("⏳ Waiting for ngrok to establish tunnel...")
    print("="*60 + "\n")
    
    public_url = None
    for line in ngrok_proc.stdout:
        print(line.rstrip())
        if "forwarding" in line.lower() or "https://" in line:
            if "https://" in line:
                parts = line.split("https://")
                if len(parts) > 1:
                    url_part = parts[1].split()[0]
                    public_url = f"https://{url_part}"
    
    if public_url:
        print("\n" + "="*60)
        print(f"✅ PUBLIC URL: {public_url}")
        print("="*60)
        print(f"\n📤 Share this link with your friends:\n   {public_url}\n")
        print("Local server: http://localhost:5052\n")
    else:
        print("\n⚠️ Could not extract URL from ngrok output")
        print("Check your ngrok authentication at: https://dashboard.ngrok.com")
    
    print("Tunnel is active. Press Ctrl+C to stop.\n")
    ngrok_proc.wait()
    
except KeyboardInterrupt:
    print("\n\n🛑 Stopping ngrok and server...")
    ngrok_proc.terminate()
    server_proc.terminate()
    print("Done!")
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nMake sure ngrok is installed:")
    print("  npm install -g ngrok")
