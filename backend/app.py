from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import logging
from voice_browser import VoiceBrowser

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS for localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_browsers = {}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    browser_id = id(websocket)
    logger.info(f"WebSocket connection established: {browser_id}")
    
    try:
        # Create voice browser instance
        voice_browser = VoiceBrowser(websocket)
        active_browsers[browser_id] = voice_browser
        
        # Send initial connection message
        await voice_browser.send_message("status", {"message": "Voice browser connected and ready"})
        
        # Start the browser in background
        browser_task = asyncio.create_task(voice_browser.run())
        
        # Handle incoming messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                logger.info(f"Received message: {message}")
                
                if message["type"] == "command":
                    voice_browser.add_command(message["text"])
                elif message["type"] == "interrupt":
                    voice_browser.interrupt_speech = True
                elif message["type"] == "cancel":
                    if voice_browser.current_agent:
                        voice_browser.current_agent.stop()
                        await voice_browser.send_message("status", {"message": "Task cancelled"})
                        
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
                break
            except json.JSONDecodeError:
                logger.error("Invalid JSON received")
            except Exception as e:
                logger.error(f"Error handling message: {e}")
                await voice_browser.send_message("error", {"message": str(e)})
                    
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if browser_id in active_browsers:
            await active_browsers[browser_id].cleanup()
            del active_browsers[browser_id]
        if 'browser_task' in locals():
            browser_task.cancel()
            try:
                await browser_task
            except asyncio.CancelledError:
                pass

@app.get("/")
async def root():
    return {"message": "Voice Browser API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "active_connections": len(active_browsers)} 