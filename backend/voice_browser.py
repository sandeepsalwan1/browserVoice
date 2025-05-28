import asyncio
import os
import json
import logging
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser, BrowserConfig
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

"""
Voice browser that communicates via WebSocket
- receives commands from frontend
- sends status updates and speech text back
- simplified for web interface
"""

class VoiceBrowser:
	def __init__(self, websocket):
		self.websocket = websocket
		self.browser = Browser(config=BrowserConfig(headless=False))
		
		# Setup LLM
		openai_api_key = os.getenv('OPENAI_API_KEY')
		if not openai_api_key:
			raise ValueError('OPENAI_API_KEY not found in environment')
		self.llm = ChatOpenAI(model='gpt-4o', api_key=openai_api_key)
		
		# Command management - using asyncio.Queue instead of queue.Queue
		self.command_queue = asyncio.Queue()
		self.is_processing = False
		self.current_agent = None
		self.interrupt_speech = False
		self.should_stop = False
	
	async def send_message(self, type, data):
		"""Send message to frontend via WebSocket"""
		try:
			message = json.dumps({
				"type": type,
				"data": data
			})
			await self.websocket.send_text(message)
			logger.info(f"Sent message: {type} - {data}")
		except Exception as e:
			logger.error(f"Error sending message: {e}")
	
	async def speak(self, text, short=False):
		"""Send text to frontend for speech synthesis"""
		await self.send_message("speak", {
			"text": text,
			"short": short
		})
	
	def add_command(self, command):
		"""Add command to queue (thread-safe)"""
		try:
			self.command_queue.put_nowait(command)
			logger.info(f"Command added to queue: {command}")
		except asyncio.QueueFull:
			logger.error("Command queue is full")
	
	async def run(self):
		"""Main execution loop"""
		await self.send_message("status", {"message": "Voice browser ready"})
		
		try:
			while not self.should_stop:
				try:
					# Check for commands with timeout
					try:
						command = await asyncio.wait_for(self.command_queue.get(), timeout=1.0)
						await self.send_message("status", {"message": f"Processing: {command}"})
						
						self.is_processing = True
						
						# Run the agent
						self.current_agent = Agent(
							task=command,
							llm=self.llm,
							browser=self.browser,
							enable_memory=False,
						)
						
						try:
							logger.info(f"Starting agent with task: {command}")
							result = await self.current_agent.run()
							
							if result.final_result():
								summary = self.summarize_result(result.final_result())
								await self.speak(f'Task completed. {summary}', short=True)
								await self.send_message("status", {"message": f"Completed: {summary}"})
							else:
								await self.speak('Task completed')
								await self.send_message("status", {"message": "Task completed"})
								
						except Exception as e:
							error_msg = f'Error: {str(e)[:100]}'
							await self.speak(error_msg, short=True)
							await self.send_message("error", {"message": str(e)})
							logger.error(f"Agent error: {e}")
						finally:
							self.current_agent = None
						
						self.is_processing = False
						await self.speak('Ready for next command')
						await self.send_message("status", {"message": "Ready for next command"})
					
					except asyncio.TimeoutError:
						# No command received, continue loop
						continue
				
				except Exception as e:
					logger.error(f"Error in main loop: {e}")
					await self.send_message("error", {"message": f"System error: {str(e)}"})
					await asyncio.sleep(1)
		
		except asyncio.CancelledError:
			logger.info("Voice browser cancelled")
			raise
		except Exception as e:
			logger.error(f"Fatal error in voice browser: {e}")
			await self.send_message("error", {"message": f"Fatal error: {str(e)}"})
	
	async def cleanup(self):
		"""Clean up resources"""
		try:
			self.should_stop = True
			if self.current_agent:
				self.current_agent.stop()
			await self.browser.close()
			logger.info("Voice browser cleaned up")
		except Exception as e:
			logger.error(f"Error during cleanup: {e}")
	
	def summarize_result(self, result_text):
		"""Create a brief summary of the result"""
		if not result_text or len(result_text) < 50:
			return result_text
		
		if len(result_text) > 200:
			first_period = result_text.find('.')
			if 10 < first_period < 100:
				return result_text[:first_period + 1]
			else:
				return result_text[:100] + '...'
		return result_text
