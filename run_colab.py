import os
import re
import sys
import time
import subprocess
import urllib.request

def wait_for_backend(url="http://127.0.0.1:8000/api/health", timeout=90):
    print("⏳ Waiting for backend to load model into VRAM and bind port 8000...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HealthChecker"})
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    print("\n✅ Backend is READY and responding on port 8000!")
                    return True
        except Exception:
            time.sleep(2)
            print(".", end="", flush=True)
    return False

def main():
    print("🛰️  Starting SatQuery AI on Colab...")

    # 1. Verify / Build frontend
    dist_dir = os.path.abspath("dist")
    if not os.path.exists(dist_dir) or not os.path.exists(os.path.join(dist_dir, "index.html")):
        print("📦 Building React frontend bundle...")
        subprocess.run(["npm", "run", "build"], check=True)

    # 2. Kill any stale processes on port 8000
    subprocess.run(["fuser", "-k", "8000/tcp"], stderr=subprocess.DEVNULL)
    time.sleep(1)

    # 3. Launch FastAPI backend and pipe output to terminal
    print("🚀 Launching backend server on port 8000...")
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"],
        stdout=None,  # Print directly to Colab console
        stderr=None
    )

    # 4. Wait for backend to be completely ready
    if not wait_for_backend():
        print("\n❌ Backend failed to start within 90 seconds. Check logs above.")
        backend_proc.terminate()
        return

    # 5. Install Cloudflare binary if not present
    if not os.path.exists("/usr/local/bin/cloudflared"):
        print("🌐 Installing Cloudflare Quick Tunnel...")
        subprocess.run([
            "curl", "-s", "-L",
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
            "-o", "/usr/local/bin/cloudflared"
        ], check=True)
        subprocess.run(["chmod", "+x", "/usr/local/bin/cloudflared"], check=True)

    # 6. Launch Cloudflare Tunnel pointing to 127.0.0.1 (avoids IPv6 issues)
    print("🔗 Connecting Cloudflare Tunnel to http://127.0.0.1:8000...")
    tunnel_proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", "http://127.0.0.1:8000", "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # 7. Extract the public trycloudflare.com URL
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
        print("🎉 SUCCESS! SatQuery AI is live at your public link:")
        print(f"👉  {public_url}")
        print("=" * 65)
        print("✅ The backend was verified 200 OK before generating this link.")
        print("Keep this Colab cell running while using the application.\n")
    else:
        print("⚠️ Cloudflare tunnel failed to produce a link. Trying backup tunnel (LocalTunnel)...")
        # Backup Zero-Auth Tunnel
        subprocess.Popen(["npx", "-y", "localtunnel", "--port", "8000"])

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nTerminating processes...")
        backend_proc.terminate()
        tunnel_proc.terminate()

if __name__ == "__main__":
    main()
