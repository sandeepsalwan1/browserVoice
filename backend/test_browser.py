#!/usr/bin/env python3
"""
Test script to verify browser configuration
"""

import asyncio
import os
import logging
from browser_use import Agent
from browser_use.browser import BrowserSession, BrowserProfile
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

async def test_browser_config():
    """Test different browser configurations"""
    
    # Check environment
    print("=" * 50)
    print("Environment Check:")
    print(f"DISPLAY: {os.getenv('DISPLAY')}")
    print(f"BROWSER_HEADLESS: {os.getenv('BROWSER_HEADLESS')}")
    print(f"RAILWAY_ENVIRONMENT: {os.getenv('RAILWAY_ENVIRONMENT')}")
    print(f"PORT: {os.getenv('PORT')}")
    print("=" * 50)
    
    # Test 1: Headless browser
    print("\nTest 1: Testing headless browser...")
    try:
        profile = BrowserProfile(headless=True, user_data_dir=None)
        session = BrowserSession(browser_profile=profile)
        await session.start()
        print("✅ Headless browser session created successfully")
        await session.close()
    except Exception as e:
        print(f"❌ Headless browser failed: {e}")
    
    # Test 2: Non-headless browser (if display available)
    if os.getenv('DISPLAY'):
        print("\nTest 2: Testing non-headless browser...")
        try:
            profile = BrowserProfile(headless=False, user_data_dir=None)
            session = BrowserSession(browser_profile=profile)
            await session.start()
            print("✅ Non-headless browser session created successfully")
            await session.close()
        except Exception as e:
            print(f"❌ Non-headless browser failed: {e}")
    else:
        print("\nTest 2: Skipping non-headless test (no DISPLAY)")
    
    # Test 3: Browser with agent
    print("\nTest 3: Testing browser with agent...")
    try:
        # Use headless for this test to ensure it works
        profile = BrowserProfile(
            headless=True,
            user_data_dir=None,
            disable_security=False,
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )
        session = BrowserSession(browser_profile=profile)
        await session.start()
        
        # Setup LLM
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            print("❌ OPENAI_API_KEY not found")
            return
        
        llm = ChatOpenAI(model='gpt-4o', api_key=openai_api_key)
        
        # Create agent
        agent = Agent(
            task="Go to example.com and tell me the title",
            llm=llm,
            browser_session=session,
            enable_memory=False
        )
        
        print("Running agent...")
        result = await agent.run()
        
        if result.final_result():
            print(f"✅ Agent completed: {result.final_result()[:100]}...")
        else:
            print("✅ Agent completed (no explicit result)")
        
        await session.close()
        
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_browser_config()) 