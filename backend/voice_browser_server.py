import asyncio
import os
import time
import threading
import subprocess
import queue
import json
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser, BrowserConfig
from dotenv import load_dotenv
import speech_recognition as sr

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceBrowserServer:
    """
    Voice browser server that handles both voice commands and WebSocket connections
    """
    
    def __init__(self):
        # Core components
        self.browser: Optional[Browser] = None  # Add type hint for clarity
        
        # Setup LLM
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError('OPENAI_API_KEY not found in environment')
        self.llm = ChatOpenAI(model='gpt-4o', api_key=openai_api_key)
        
        # Speech recognition
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 1500
        self.recognizer.dynamic_energy_threshold = True
        
        # Command management
        self.command_queue = queue.Queue()
        self.is_processing = False
        self.should_stop = False
        self.current_agent = None
        self.speaking = False
        self.interrupt_speech = False
        
        # WebSocket connections
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Voice listening thread
        self.listen_thread = None
        self.voice_enabled = False
    
    async def connect_websocket(self, websocket: WebSocket, client_id: str):
        """Connect a new WebSocket client"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"WebSocket client {client_id} connected")
        
        # Send initial status
        await self.send_to_client(client_id, {
            "type": "status",
            "data": {"message": "Connected to voice browser server"}
        })
    
    def disconnect_websocket(self, client_id: str):
        """Disconnect a WebSocket client"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"WebSocket client {client_id} disconnected")
    
    async def send_to_client(self, client_id: str, message: Dict[str, Any]):
        """Send message to specific WebSocket client"""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to client {client_id}: {e}")
                self.disconnect_websocket(client_id)
    
    async def broadcast_to_all(self, message: Dict[str, Any]):
        """Send message to all connected WebSocket clients"""
        logger.info(f"📡 Broadcasting to {len(self.active_connections)} clients: {message['type']} - {str(message['data'])[:100]}")
        
        for client_id in list(self.active_connections.keys()):
            await self.send_to_client(client_id, message)
    
    def speak(self, text: str, short: bool = False):
        """Text-to-speech with brevity and interruption"""
        logger.info(f'🔊 {text}')
        
        # Send to WebSocket clients for frontend to speak
        asyncio.create_task(self.broadcast_to_all({
            "type": "speak",
            "data": {"text": text, "short": short}
        }))
        
        # Truncate long outputs if requested
        if short and len(text) > 100:
            first_sentence_end = text.find('. ')
            if first_sentence_end > 0 and first_sentence_end < 100:
                speech_text = text[:first_sentence_end + 1]
            else:
                speech_text = text[:100] + '...'
        else:
            speech_text = text
        
        # Enable interruption
        self.speaking = True
        self.interrupt_speech = False
        
        # Platform-specific speech (only if voice is enabled)
        if self.voice_enabled and os.name == 'posix':  # macOS or Linux
            speech_thread = threading.Thread(target=lambda: subprocess.run(['say', speech_text]))
            speech_thread.start()
            
            # Monitor for interruption
            while speech_thread.is_alive() and not self.interrupt_speech:
                time.sleep(0.1)
            
            if self.interrupt_speech and speech_thread.is_alive():
                try:
                    subprocess.run(['killall', 'say'])
                except:
                    pass
        
        self.speaking = False
    
    def start_voice_listening(self):
        """Start the voice listening thread"""
        if not self.listen_thread or not self.listen_thread.is_alive():
            self.voice_enabled = True
            self.listen_thread = threading.Thread(target=self.continuous_listen)
            self.listen_thread.daemon = True
            self.listen_thread.start()
            logger.info("Voice listening started")
    
    def stop_voice_listening(self):
        """Stop the voice listening"""
        self.voice_enabled = False
        if self.listen_thread:
            logger.info("Voice listening stopped")
    
    def continuous_listen(self):
        """Continuously listen for voice commands"""
        self.speak('Voice browser ready')
        
        while self.voice_enabled and not self.should_stop:
            try:
                # Show status
                if self.is_processing:
                    logger.info("🔄 Processing... (say 'cancel' to stop)")
                else:
                    logger.info('🎤 Listening...')
                
                # Listen for command
                with sr.Microphone() as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.recognizer.listen(source, phrase_time_limit=5)
                
                try:
                    text = self.recognizer.recognize_google(audio).lower()
                    logger.info(f'📢 Heard: {text}')
                    
                    # Process the voice command using asyncio.run_coroutine_threadsafe
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self.process_voice_command(text))
                    loop.close()
                
                except sr.UnknownValueError:
                    pass  # Speech wasn't understood
                except sr.RequestError as e:
                    logger.error(f'Recognition error: {e}')
            
            except Exception as e:
                logger.error(f'Listening error: {e}')
                time.sleep(1)
    
    async def process_voice_command(self, text: str):
        """Process a voice command"""
        # Check for interruption commands
        if self.speaking and any(cmd in text for cmd in ['next', 'stop talking', 'quiet', 'skip']):
            logger.info('Speech interrupted')
            self.interrupt_speech = True
            return
        
        # Check for cancel command
        if text in ['cancel', 'stop', 'cancel that', 'never mind', 'interrupt', 'quit'] and self.is_processing:
            logger.info("Cancel command received")
            if self.current_agent:
                try:
                    self.current_agent.stop()
                    logger.info("Agent stopped successfully")
                except Exception as e:
                    logger.error(f"Error stopping agent: {e}")
                
            self.interrupt_speech = True  # Stop any ongoing speech
            self.speak('Cancelling task')
            self.is_processing = False
            self.current_agent = None
            await self.broadcast_to_all({
                "type": "status",
                "data": {"message": "Task cancelled"}
            })
            return
        
        # Exit commands
        if text in ['exit browser', 'quit browser', 'goodbye browser', 'shutdown']:
            self.should_stop = True
            self.speak('Shutting down')
            await self.broadcast_to_all({
                "type": "status",
                "data": {"message": "Shutting down"}
            })
            return
        
        # Treat all other input as commands when not already processing
        if not self.is_processing:
            await self.execute_command(text)
    
    async def process_websocket_message(self, client_id: str, message: Dict[str, Any]):
        """Process a message from WebSocket client"""
        message_type = message.get('type')
        text = message.get('text', '')
        
        if message_type == 'command':
            await self.execute_command(text)
        elif message_type == 'cancel':
            logger.info("Cancel command received via WebSocket")
            if self.current_agent and self.is_processing:
                try:
                    self.current_agent.stop()
                    logger.info("Agent stopped via WebSocket")
                except Exception as e:
                    logger.error(f"Error stopping agent via WebSocket: {e}")
                
            self.interrupt_speech = True  # Stop any ongoing speech
            self.is_processing = False
            self.current_agent = None
            await self.send_to_client(client_id, {
                "type": "status",
                "data": {"message": "Task cancelled"}
            })
        elif message_type == 'interrupt':
            self.interrupt_speech = True
    
    async def execute_command(self, command: str):
        """Execute a browser command"""
        if self.is_processing:
            logger.warning(f"Command '{command}' ignored - already processing")
            return
        
        # Check if browser is initialized
        if self.browser is None:
            error_msg = "Browser not initialized. Please restart the server."
            logger.error(error_msg)
            self.speak(error_msg)
            await self.broadcast_to_all({
                "type": "error",
                "data": {"message": error_msg}
            })
            return
        
        self.is_processing = True
        logger.info(f"🚀 Starting command execution: {command}")
        
        # Send initial processing status (no speech)
        await self.broadcast_to_all({
            "type": "status",
            "data": {"message": f"Processing: {command}"}
        })
        
        try:
            logger.info(f"Creating agent with task: {command}")
            # Create and run the agent - let Agent create its own browser if needed
            self.current_agent = Agent(
                task=command,
                llm=self.llm,
            )
            
            logger.info("🤖 Starting agent execution...")
            result = await self.current_agent.run()
            logger.info(f"Agent completed. Result available: {bool(result.final_result())}")
            
            if result.final_result():
                # Get the full result for accessibility
                full_result = result.final_result()
                summary = self.summarize_result(full_result)
                
                logger.info(f"📤 Sending result to frontend (length: {len(full_result)} chars)")
                logger.info(f"📋 Result preview: {full_result[:200]}...")
                
                # Send ONLY the result message for speech synthesis
                # The frontend will handle speaking this
                await self.broadcast_to_all({
                    "type": "result",
                    "data": {"text": full_result}
                })
                
                # Send a separate status message (no speech) for UI display
                await self.broadcast_to_all({
                    "type": "status",
                    "data": {"message": f"Completed: {summary}"}
                })
                
                logger.info(f"✅ Task completed successfully. Summary: {summary}")
            else:
                logger.warning("📤 No explicit result from agent - this might indicate a browser issue")
                # Check if we have any browser-related errors
                browser_status = "Browser status unknown"
                try:
                    if self.current_agent and hasattr(self.current_agent, 'browser'):
                        browser_status = f"Browser available: {self.current_agent.browser is not None}"
                except:
                    browser_status = "Browser check failed"
                
                logger.info(f"🔍 {browser_status}")
                
                # Send a more informative message
                await self.broadcast_to_all({
                    "type": "result",
                    "data": {"text": f"Task completed but no specific result was returned. This might be due to browser limitations in the deployment environment. {browser_status}"}
                })
                await self.broadcast_to_all({
                    "type": "status",
                    "data": {"message": "Task completed (no result)"}
                })
                
        except Exception as e:
            error_msg = f'Error: {str(e)[:200]}'
            logger.error(f'❌ Agent error: {str(e)}')
            logger.error(f'❌ Error type: {type(e).__name__}')
            
            # Check if it's a browser-related error
            if 'browser' in str(e).lower() or 'chrome' in str(e).lower() or 'playwright' in str(e).lower():
                error_msg = f"Browser error: {str(e)[:150]}. This might be due to headless environment limitations."
            
            await self.broadcast_to_all({
                "type": "error",
                "data": {"message": error_msg}
            })
        finally:
            self.current_agent = None
            self.is_processing = False
            logger.info(f"🏁 Command execution finished: {command}")
            
            # Send ready status (no speech)
            await self.broadcast_to_all({
                "type": "status",
                "data": {"message": "Ready for next command"}
            })
    
    def summarize_result(self, result_text: str) -> str:
        """Create a brief summary of the result for status messages"""
        if not result_text or len(result_text) < 50:
            return result_text
        
        # Extract key information from long results
        # Keep first sentence or first 100 chars for status
        if len(result_text) > 200:
            first_period = result_text.find('.')
            if 10 < first_period < 100:
                return result_text[:first_period + 1]
            else:
                return result_text[:100] + '...'
        return result_text
    
    async def cleanup(self):
        """Cleanup resources"""
        self.should_stop = True
        self.voice_enabled = False
        await self.browser.close()

# Create FastAPI app
app = FastAPI(title="Voice Browser Server")

# Add CORS middleware with specific origins for better security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create server instance
voice_server = VoiceBrowserServer()

@app.on_event("startup")
async def startup_event():
    """Start the voice browser server"""
    logger.info("Starting Voice Browser Server...")
    
    # Initialize browser
    try:
        # Detect if we're running in a headless environment (like Railway)
        is_headless_env = os.getenv('RAILWAY_ENVIRONMENT') or os.getenv('VERCEL') or not os.getenv('DISPLAY')
        headless_mode = is_headless_env or os.getenv('HEADLESS', 'false').lower() == 'true'
        
        logger.info(f"Environment detection - Headless: {headless_mode}, Railway: {bool(os.getenv('RAILWAY_ENVIRONMENT'))}")
        
        browser_config = BrowserConfig(headless=headless_mode)
        voice_server.browser = Browser(config=browser_config)
        logger.info(f"Browser initialized successfully. Headless: {headless_mode}, Type: {type(voice_server.browser)}")
    except Exception as e:
        logger.error(f"Failed to initialize browser: {e}")
        logger.error(f"Environment variables: DISPLAY={os.getenv('DISPLAY')}, RAILWAY_ENVIRONMENT={os.getenv('RAILWAY_ENVIRONMENT')}")
        voice_server.browser = None
    
    # Optionally start voice listening on startup
    # voice_server.start_voice_listening()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Voice Browser Server...")
    await voice_server.cleanup()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for frontend communication"""
    client_id = f"client_{int(time.time() * 1000)}"
    
    try:
        await voice_server.connect_websocket(websocket, client_id)
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Process the message
            await voice_server.process_websocket_message(client_id, message)
            
    except WebSocketDisconnect:
        voice_server.disconnect_websocket(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        voice_server.disconnect_websocket(client_id)

@app.get("/")
async def root():
    """Health check endpoint"""
    browser_status = "not initialized"
    if voice_server.browser:
        browser_status = "initialized"
    
    return {
        "message": "Voice Browser Server is running",
        "browser_status": browser_status,
        "environment": {
            "railway": bool(os.getenv('RAILWAY_ENVIRONMENT')),
            "display": bool(os.getenv('DISPLAY')),
            "headless": os.getenv('HEADLESS', 'auto')
        }
    }

@app.post("/start-voice")
async def start_voice():
    """Start voice listening"""
    voice_server.start_voice_listening()
    return {"message": "Voice listening started"}

@app.post("/stop-voice")
async def stop_voice():
    """Stop voice listening"""
    voice_server.stop_voice_listening()
    return {"message": "Voice listening stopped"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 