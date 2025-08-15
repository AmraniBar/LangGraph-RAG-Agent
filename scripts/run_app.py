#!/usr/bin/env python3
"""
Startup script for the Cybersecurity RAG Agent Streamlit application.
This script handles the application launch and provides helpful information.
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if required packages are installed"""
    try:
        import streamlit
        import langchain
        import langgraph
        return True
    except ImportError:
        return False

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "../requirements.txt"])
        return True
    except subprocess.CalledProcessError:
        print("Failed to install requirements. Please install manually:")
        print("pip install -r requirements.txt")
        return False

def main():
    print("🔒 Cybersecurity RAG Agent")
    print("=" * 50)
    
    # Check if requirements are installed
    if not check_requirements():
        print("⚠️  Required packages not found.")
        install_choice = input("Would you like to install them now? (y/n): ").lower().strip()
        
        if install_choice == 'y':
            if not install_requirements():
                sys.exit(1)
        else:
            print("Please install requirements manually: pip install -r requirements.txt")
            sys.exit(1)
    
    print("✅ All requirements satisfied!")
    print("\n📝 Note: Make sure you have the required NIST PDF documents in the correct path.")
    print("Current paths expected:")
    print("- data/documents/NIST.CSWP.29.pdf")
    print("- data/documents/NIST.SP.800-53r5.pdf") 
    print("- data/documents/NIST.SP.800-61r3.pdf")
    print("- data/documents/nist.sp.800-61r2.pdf")
    print("\n💡 You may need to update the paths in main.py if your documents are located elsewhere.")
    
    print("\n🚀 Starting Streamlit application...")
    print("The application will open in your default web browser.")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    print("\nPress Ctrl+C to stop the application.\n")
    
    try:
        # Launch Streamlit app
        subprocess.run([sys.executable, "-m", "streamlit", "run", "../app.py", "--server.headless", "false"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped. Thank you for using Cybersecurity RAG Agent!")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()