# Voice Browser - Quick Start Guide

## ✅ Current Status
Your voice browser is now **fully integrated and working**! Both the backend and frontend are connected and ready to use.

## 🚀 How to Use

### 1. **Start the Backend Server**
```bash
cd /Users/sandeep/Downloads/NeuSpring/projects/browserVoice
source backend/venv/bin/activate
cd backend
python -m uvicorn voice_browser_server:app --host 0.0.0.0 --port 8000 --reload
```

### 2. **Start the Frontend**
```bash
cd "Cofounder-Labs interactive-browser-use main frontend"
npm run dev
```

### 3. **Access the Voice Browser**
1. Open your browser and go to: http://localhost:3000/voice-browser
2. You should see "Connected" (green dot) indicating the backend is working
3. Click **"Start Listening"** to begin voice control
4. Say commands like:
   - "Go to Google"
   - "Search for Python tutorials"
   - "Navigate to GitHub"
   - "Find the latest news about AI"

### 4. **Accessibility Features for Blind Users**
- **Screen Reader Support**: All status updates are announced automatically
- **Clear Voice Feedback**: The system speaks all results and actions
- **Simple Interface**: Minimal buttons, focused on voice control
- **Keyboard Navigation**: All controls are keyboard accessible
- **ARIA Labels**: Proper screen reader labels throughout

## 🎤 Voice Commands

### Basic Commands
- **"Go to [website]"** - Navigate to any website
- **"Search for [topic]"** - Perform web searches
- **"Click [element]"** - Click on page elements
- **"Scroll down/up"** - Navigate pages
- **"Fill form with [data]"** - Complete web forms

### Control Commands
- **"Cancel"** - Stop the current task
- **"Stop talking"** - Interrupt speech output

## 🔧 System Overview

### Backend (Port 8000)
- **Voice Browser Server**: Processes commands using browser-use + OpenAI
- **WebSocket API**: Real-time communication with frontend
- **Browser Automation**: Controls Chrome browser for web tasks
- **Text-to-Speech**: Reads results aloud (macOS `say` command)

### Frontend (Port 3000)
- **Voice Browser UI**: Simple, accessible interface
- **Speech Recognition**: Browser-based voice input
- **WebSocket Client**: Connects to backend server
- **Screen Reader Support**: Full accessibility features

## 🛠️ Troubleshooting

### If "Disconnected" appears:
1. Check if backend is running: `curl http://localhost:8000`
2. Verify your `.env` file has `OPENAI_API_KEY=your_key_here`
3. Restart the backend server

### If voice recognition doesn't work:
1. Ensure you're using Chrome or Safari
2. Allow microphone permissions when prompted
3. Check your system microphone settings

### If browser automation fails:
1. Verify Chrome is installed
2. Check your internet connection
3. Some sites may block automation

## 📋 Next Steps

Your voice browser is ready for use! The system will:
1. ✅ Accept voice commands through the web interface
2. ✅ Process them using AI (GPT-4o)
3. ✅ Control a Chrome browser automatically
4. ✅ Read results aloud for accessibility
5. ✅ Provide real-time status updates

## 🎯 Perfect for Blind Users

This setup is specifically designed for accessibility:
- **No complex interface** - Just voice commands
- **Full audio feedback** - Everything is spoken aloud
- **Screen reader compatible** - Proper ARIA labels
- **Simple controls** - Start/stop listening, cancel tasks
- **Automatic reconnection** - Robust connection handling

Enjoy your voice-controlled web browsing experience! 🎉 