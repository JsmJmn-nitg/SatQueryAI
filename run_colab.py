import os
import re
import sys
import time
import subprocess

def setup_and_launch():
    print("🛰️  Setting up SatQuery AI on Colab...")

    # 1. Download standalone cloudflared binary (No signup / No token required)
    if not os.path.exists("/usr/local/bin/cloudflared"):
        print("📦 Installing Cloudflare Quick Tunnel...")
        subprocess.run([
            "curl", "-s", "-L",
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
            "-o", "/usr/local/bin/cloudflared"
        ], check=True)
        subprocess.run(["chmod", "+x", "/usr/local/bin/cloudflared"], check=True)

    # 2. Start FastAPI Backend in background on port 8000
    print("🚀 Starting FastAPI Server & Static Frontend on port 8000...")
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    time.sleep(3)

    # 3. Start Cloudflare Tunnel
    print("🌐 Creating Public Tunnel (zero-auth)...")
    tunnel_proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", "http://localhost:8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # 4. Extract public trycloudflare.com URL from logs
    public_url = None
    start_time = time.time()
    while time.time() - start_time < 35:
        line = tunnel_proc.stderr.readline()
        if "trycloudflare.com" in line:
            match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
            if match:
                public_url = match.group(0)
                break

    if public_url:
        print("\n" + "=" * 65)
        print("🎉 SUCCESS! Open your SatQuery AI application here:")
        print(f"👉  {public_url}")
        print("=" * 65 + "\n")
        print("⚠️ Keep this Colab cell running while using the app.")
        
        # Keep process alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down servers...")
            backend_proc.terminate()
            tunnel_proc.terminate()
    else:
        print("❌ Failed to obtain tunnel URL. Please re-run the cell.")

if __name__ == "__main__":
    setup_and_launch()
