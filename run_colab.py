import os
import re
import sys
import time
import subprocess

<<<<<<< HEAD
def setup_and_launch():
    print("🛰️  Setting up SatQuery AI on Colab...")

    # 1. Download standalone cloudflared binary (No signup / No token required)
    if not os.path.exists("/usr/local/bin/cloudflared"):
        print("📦 Installing Cloudflare Quick Tunnel...")
=======
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
>>>>>>> 737f43b (second commit)
        subprocess.run([
            "curl", "-s", "-L",
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
            "-o", "/usr/local/bin/cloudflared"
        ], check=True)
        subprocess.run(["chmod", "+x", "/usr/local/bin/cloudflared"], check=True)

<<<<<<< HEAD
    # 2. Start FastAPI Backend in background on port 8000
    print("🚀 Starting FastAPI Server & Static Frontend on port 8000...")
=======
    # 3. Launch FastAPI backend serving on port 8000
    print("🚀 Launching backend server on port 8000...")
>>>>>>> 737f43b (second commit)
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    time.sleep(3)

<<<<<<< HEAD
    # 3. Start Cloudflare Tunnel
    print("🌐 Creating Public Tunnel (zero-auth)...")
=======
    # 4. Open Cloudflare public tunnel
    print("🔗 Generating public URL...")
>>>>>>> 737f43b (second commit)
    tunnel_proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", "http://localhost:8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

<<<<<<< HEAD
    # 4. Extract public trycloudflare.com URL from logs
    public_url = None
    start_time = time.time()
    while time.time() - start_time < 35:
=======
    # 5. Read tunnel output to extract the trycloudflare.com URL
    public_url = None
    start = time.time()
    while time.time() - start < 35:
>>>>>>> 737f43b (second commit)
        line = tunnel_proc.stderr.readline()
        if "trycloudflare.com" in line:
            match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
            if match:
                public_url = match.group(0)
                break

    if public_url:
        print("\n" + "=" * 65)
<<<<<<< HEAD
        print("🎉 SUCCESS! Open your SatQuery AI application here:")
        print(f"👉  {public_url}")
        print("=" * 65 + "\n")
        print("⚠️ Keep this Colab cell running while using the app.")
        
        # Keep process alive
=======
        print("🎉 SUCCESS! SatQuery AI is live at your public link:")
        print(f"👉  {public_url}")
        print("=" * 65 + "\n")
        print("Keep this Colab cell running while using the application.")

>>>>>>> 737f43b (second commit)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
<<<<<<< HEAD
            print("\nShutting down servers...")
            backend_proc.terminate()
            tunnel_proc.terminate()
    else:
        print("❌ Failed to obtain tunnel URL. Please re-run the cell.")

if __name__ == "__main__":
    setup_and_launch()
=======
            print("\nTerminating processes...")
            backend_proc.terminate()
            tunnel_proc.terminate()
    else:
        print("❌ Could not obtain a tunnel URL. Please rerun the script.")
        backend_proc.terminate()
        tunnel_proc.terminate()

if __name__ == "__main__":
    main()
>>>>>>> 737f43b (second commit)
