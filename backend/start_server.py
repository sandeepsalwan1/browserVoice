#!/usr/bin/env python3
"""
Startup script for the Voice Browser Server
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Import the app
from voice_browser_server import app

def main():
    """Start the Voice Browser Server"""
    print("🚀 Starting Voice Browser Server...")
    print("📡 Server will be available at: http://localhost:8000")
    print("🔌 WebSocket endpoint: ws://localhost:8000/ws")
    print("🎤 Voice commands will be processed through WebSocket")
    print("💻 Frontend should connect to: http://localhost:3000")
    print("\n" + "="*50)
    
    # Start the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False  # Set to True for development
    )

if __name__ == "__main__":
    main() 