#!/usr/bin/env python3
# ==============================================
# 🚀 AI Radiology Assistant Launcher (run_app.py)
# ==============================================
"""
This script is the main entry point for launching the Explainable AI for Chest X-Ray Diagnosis system.

🔍 Overview:
------------
This script provides a polished and professional interface for launching the Streamlit-based
frontend of the AI-powered Chest X-Ray Diagnosis System with Explainable AI (XAI) features.

✨ Features:
------------
1. Performs pre-launch system checks (Python, Streamlit, directories).
2. Displays a styled startup banner with system information.
3. Automatically launches the main Streamlit application.
4. Offers clear and colored console messages (mimicking a real application launcher).

💡 Usage:
---------
    python run_app.py

Developed by:
👨‍💻 Amal Prasad Trivedi
🎓 B.Tech (CS - AI & ML) | Explainable AI Researcher
"""

import os
import sys
import subprocess
import platform
import shutil
import time


def print_banner():
    """Display a beautiful ASCII banner with icons."""
    print("\n" + "=" * 65)
    print("🩻  Explainable AI for Chest X-Ray Diagnosis")
    print("=" * 65)
    print("💡  Powered by Deep Learning (CNN + Transfer Learning)")
    print("🧠  Explainability via Grad-CAM and SHAP")
    print("🌐  Frontend: Streamlit | Developer: Amal Prasad Trivedi")
    print("=" * 65)
    print()


def check_environment():
    """Perform basic environment checks before running Streamlit."""
    print("🔍 Running system checks...\n")

    # 1️⃣ Python version check
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        sys.exit(1)
    else:
        print(f"✅ Python Version: {platform.python_version()}")

    # 2️⃣ Check if Streamlit is installed
    if shutil.which("streamlit") is None:
        print("❌ Streamlit not found. Installing it now...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"], check=True)
    else:
        print("✅ Streamlit is installed")

    # 3️⃣ Check app directory and Home.py
    app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app")
    home_file = os.path.join(app_dir, "Home.py")

    if not os.path.exists(app_dir):
        print("❌ Missing 'app/' directory. Please check your project structure.")
        sys.exit(1)
    elif not os.path.exists(home_file):
        print("❌ Missing 'Home.py' inside the app folder.")
        sys.exit(1)
    else:
        print("✅ App directory and Home.py found")

    print("\n🧩 Environment check complete!\n")
    return home_file


def launch_streamlit(home_file):
    """Launch the Streamlit app."""
    print("🚀 Launching Streamlit application...\n")
    print("🌍 Opening local development server...")
    print("💻 Press CTRL + C to stop the application.\n")
    time.sleep(1.5)

    try:
        subprocess.run(["streamlit", "run", home_file], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Streamlit launch failed: {e}")
        sys.exit(1)


def main():
    """Main function to control the launcher workflow."""
    print_banner()
    home_file = check_environment()
    launch_streamlit(home_file)


# ==============================================
# 🏁 Entry Point
# ==============================================
if __name__ == "__main__":
    main()
