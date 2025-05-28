#!/usr/bin/env python3
"""
Test LLM connection with browser-use
"""

import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from browser_use import Agent
from browser_use.browser import BrowserSession, BrowserProfile

# Load environment variables
load_dotenv()

async def test_llm_connection():
    """Test if LLM connection works"""
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    print(f"API Key exists: {bool(api_key)}")
    print(f"API Key length: {len(api_key) if api_key else 0}")
    
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return
    
    # Test direct LLM connection
    print("\nTest 1: Testing direct LLM connection...")
    try:
        llm = ChatOpenAI(model='gpt-4o', api_key=api_key, temperature=0.0)
        response = await llm.ainvoke("Say 'Hello World'")
        print(f"✅ Direct LLM test passed: {response.content}")
    except Exception as e:
        print(f"❌ Direct LLM test failed: {e}")
        return
    
    # Test with browser-use Agent
    print("\nTest 2: Testing LLM with browser-use Agent...")
    try:
        # Create a simple browser session
        profile = BrowserProfile(headless=True, user_data_dir=None)
        session = BrowserSession(browser_profile=profile)
        await session.start()
        
        # Create agent with explicit tool_calling_method to avoid detection
        agent = Agent(
            task="Say hello",
            llm=llm,
            browser_session=session,
            enable_memory=False,
            tool_calling_method='function_calling'  # Explicitly set to avoid auto-detection
        )
        
        print("Agent created successfully")
        
        # Clean up
        await session.close()
        print("✅ Browser-use Agent test passed")
        
    except Exception as e:
        print(f"❌ Browser-use Agent test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_llm_connection()) 