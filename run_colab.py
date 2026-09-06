import os
import re
import sys
import time
import subprocess
import urllib.request

def wait_for_backend(url="http://127.0.0.1:8000/api/health", timeout=300):
    """Wait up to 5 minutes - LLaVA needs time to download on first run"""
    print("⏳ Waiting for backend (model download + load may take 3-5 min first time)...")
    start = time.time()
    attempt = 0
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HealthChecker"})
            with urllib.request.urlopen(req, timeout=3) as response:
                if response.status == 200:
                    elapsed = int(time.time() - start)
                    print(f"\n✅ Backend READY after {elapsed}s!")
                    return True
        except Exception:
            pass
        attempt += 1
        time.sleep(3)
        # Progress indicator every 15s
        if attempt % 5 == 0:
            elapsed = int(time.time() - start)
            print(f"  still loading... ({elapsed}s elapsed)")
    return False

def main():
    print("🛸 Starting SatQuery AI on Colab...")

    # 1. Build frontend if needed
    dist_dir = os.path.abspath("dist")
    if not os.path.exists(dist_dir) or not os.path.exists(os.path.join(dist_dir, "index.html")):
        print("📦 Building React frontend...")
        subprocess.run(["npm", "run", "build"], check=True)

    # 2. Kill stale processes
    subprocess.run(["fuser", "-k", "8000/tcp"], stderr=subprocess.DEVNULL)
    time.sleep(1)

    # 3. Launch backend
    print("🚀 Launching backend (LLaVA downloading on first run - be patient)...")
    backend_proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "backend.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--log-level", "info",
            "--timeout-keep-alive", "120",
        ],
        stdout=None,
        stderr=None,
    )

    # 4. Wait with long timeout
    if not wait_for_backend(timeout=300):
        print("\n❌ Backend failed to start in 5 minutes. Check logs above.")
        backend_proc.terminate()
        return

    # 5. Install cloudflared
    if not os.path.exists("/usr/local/bin/cloudflared"):
        print("🔧 Installing Cloudflare tunnel...")
        subprocess.run([
            "curl", "-s", "-L",
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
            "-o", "/usr/local/bin/cloudflared"
        ], check=True)
        subprocess.run(["chmod", "+x", "/usr/local/bin/cloudflared"], check=True)

    # 6. Start tunnel
    print("🌐 Starting Cloudflare tunnel...")
    tunnel_proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", "http://127.0.0.1:8000", "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # 7. Get public URL
    public_url = None
    start = time.time()
    while time.time() - start < 35:
        line = tunnel_proc.stderr.readline()
        if "trycloudflare.com" in line:
            match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
            if match:
                public_url = match.group(0)
                break

    if public_url:
        print("\n" + "=" * 65)
        print("🎉 SatQuery AI is LIVE:")
        print(f"🔗  {public_url}")
        print("=" * 65)
    else:
        print("⚠️  Cloudflare tunnel failed. Trying localtunnel backup...")
        subprocess.Popen(["npx", "-y", "localtunnel", "--port", "8000"])

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        backend_proc.terminate()
        tunnel_proc.terminate()

if __name__ == "__main__":
    main()
