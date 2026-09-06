import os
import re
import sys
import time
import subprocess

def main():
    print("🛰️  Starting SatQuery AI on Colab...")

    # 1. Build frontend if 'dist' folder is missing
    dist_dir = os.path.abspath("dist")
    if not os.path.exists(dist_dir) or not os.path.exists(os.path.join(dist_dir, "index.html")):
        print("📦 Building React frontend with Vite...")
        subprocess.run(["npm", "install"], check=True)
        subprocess.run(["npm", "run", "build"], check=True)
        print("✅ Frontend build completed successfully.")

    # 2. Download Cloudflare standalone binary (no login/keys required)
    if not os.path.exists("/usr/local/bin/cloudflared"):
        print("🌐 Installing Cloudflare Quick Tunnel...")
        subprocess.run([
            "curl", "-s", "-L",
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
            "-o", "/usr/local/bin/cloudflared"
        ], check=True)
        subprocess.run(["chmod", "+x", "/usr/local/bin/cloudflared"], check=True)

    # 3. Launch FastAPI backend serving on port 8000
    print("🚀 Launching backend server on port 8000...")
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    time.sleep(3)

    # 4. Open Cloudflare public tunnel
    print("🔗 Generating public URL...")
    tunnel_proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", "http://localhost:8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # 5. Read tunnel output to extract the trycloudflare.com URL
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
        print("=" * 65 + "\n")
        print("Keep this Colab cell running while using the application.")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nTerminating processes...")
            backend_proc.terminate()
            tunnel_proc.terminate()
    else:
        print("❌ Could not obtain a tunnel URL. Please rerun the script.")
        backend_proc.terminate()
        tunnel_proc.terminate()

if __name__ == "__main__":
    main()
