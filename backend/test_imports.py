#!/usr/bin/env python3
"""
Test script to verify all required imports for voice_browser_server.py
This helps ensure the requirements.txt is complete before deployment
"""

def test_imports():
    """Test all imports used in voice_browser_server.py"""
    print("Testing imports for Voice Browser Server...")
    
    try:
        import asyncio
        print("✅ asyncio")
        
        import os
        print("✅ os")
        
        import time
        print("✅ time")
        
        import threading
        print("✅ threading")
        
        import subprocess
        print("✅ subprocess")
        
        import queue
        print("✅ queue")
        
        import json
        print("✅ json")
        
        import logging
        print("✅ logging")
        
        from typing import Dict, Any, Optional
        print("✅ typing")
        
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        print("✅ fastapi")
        
        from fastapi.middleware.cors import CORSMiddleware
        print("✅ fastapi.middleware.cors")
        
        from langchain_openai import ChatOpenAI
        print("✅ langchain_openai")
        
        from browser_use import Agent, Browser, BrowserConfig
        print("✅ browser_use")
        
        from dotenv import load_dotenv
        print("✅ dotenv")
        
        import speech_recognition as sr
        print("✅ speech_recognition")
        
        print("\n🎉 All imports successful! The requirements.txt is complete.")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Please install missing packages in requirements.txt")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    exit(0 if success else 1) 