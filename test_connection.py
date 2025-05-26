#!/usr/bin/env python3
"""
Simple test script to verify the WebSocket connection between frontend and backend
"""

import asyncio
import websockets
import json

async def test_websocket():
    try:
        print("🔌 Testing WebSocket connection to backend...")
        
        # Connect to the backend WebSocket
        uri = "ws://localhost:8000/ws"
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to backend successfully!")
            
            # Send a test command
            test_message = {
                "type": "command",
                "text": "hello world test"
            }
            
            print(f"📤 Sending test message: {test_message}")
            await websocket.send(json.dumps(test_message))
            
            # Wait for response
            print("📥 Waiting for response...")
            response = await websocket.recv()
            data = json.loads(response)
            print(f"📨 Received: {data}")
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure the backend server is running: python backend/voice_browser_server.py")
        print("2. Check if port 8000 is available")
        print("3. Verify your OPENAI_API_KEY is set in .env file")

if __name__ == "__main__":
    asyncio.run(test_websocket()) 