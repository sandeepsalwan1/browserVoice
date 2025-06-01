#!/usr/bin/env python3
"""
Test WebSocket client for voice browser
"""

import asyncio
import json
import websockets

async def test_command():
    uri = "ws://localhost:8000/ws"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to WebSocket")
            
            # Wait for initial connection message
            message = await websocket.recv()
            print(f"📨 Received: {json.loads(message)}")
            
            # Send a test command
            command = {
                "type": "command",
                "text": "go to example.com and tell me what you see"
            }
            
            print(f"\n📤 Sending command: {command['text']}")
            await websocket.send(json.dumps(command))
            
            # Listen for responses
            print("\n📨 Listening for responses...")
            for i in range(30):  # Listen for up to 30 seconds
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    print(f"\n📨 Message {i+1}:")
                    print(f"  Type: {data.get('type')}")
                    print(f"  Data: {data.get('data')}")
                    
                    # If we get a result or error, we can stop
                    if data.get('type') in ['result', 'error']:
                        if data.get('type') == 'error':
                            print("\n❌ Error occurred!")
                        else:
                            print("\n✅ Task completed!")
                        break
                        
                except asyncio.TimeoutError:
                    continue
                    
            print("\n✅ Test completed")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_command()) 