# Voice Browser UI

A web interface for controlling a browser using voice commands. This project consists of a FastAPI backend that manages browser automation and a Next.js frontend that provides voice input and visual feedback.

## Features

- 🎤 Voice-controlled browser automation
- 🌐 Real-time WebSocket communication
- 🔊 Text-to-speech feedback
- 🎯 Interruptible commands ("stop talking", "cancel")
- 📊 Visual command history
- 🚀 Automated web browsing with AI

## Project Structure

```
browserVoice/
├── backend/
│   ├── voice_browser.py  # Modified browser automation script
│   ├── app.py           # FastAPI WebSocket server
│   ├── requirements.txt # Python dependencies
│   └── .env            # Environment variables (create this)
├── Cofounder-Labs interactive-browser-use main frontend/
│   ├── app/
│   │   └── voice-browser/
│   │       └── page.tsx # Main UI page
│   ├── components/      # UI components (already included)
│   └── package.json     # Node dependencies
└── README.md
```

## Prerequisites

- Python 3.8+
- Node.js 18+
- Chrome browser
- OpenAI API key

## Setup Instructions

### 1. Backend Setup

```bash
cd backend

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file and add your OpenAI API key
echo "OPENAI_API_KEY=your-openai-api-key-here" > .env
```

### 2. Frontend Setup

```bash
cd "Cofounder-Labs interactive-browser-use main frontend"

# Install dependencies
npm install
# or
pnpm install
```

## Running the Application

You'll need two terminal windows:

### Terminal 1 - Start Backend Server

```bash
cd backend
uvicorn app:app --reload --port 8000
```

The backend will be available at `http://localhost:8000`

### Terminal 2 - Start Frontend

```bash
cd web
npm run dev
# or
pnpm dev
```

The frontend will be available at `http://localhost:3000`

## Usage

1. Open `http://localhost:3000/voice-browser` in Chrome
2. Click "Start Listening" to enable voice input
3. Speak your command clearly (e.g., "Search for weather in Seattle")
4. The browser window will open separately and execute your command
5. Voice feedback will be provided through your speakers

### Voice Commands

- **Any browsing command**: "Search for Python tutorials", "Go to GitHub", etc.
- **Cancel current task**: Say "Cancel"
- **Stop speech output**: Say "Stop talking" or "Next"
- **Exit**: Close the browser window or stop the backend server

## Troubleshooting

### Backend Issues

1. **"OPENAI_API_KEY not found"**: Make sure you've created the `.env` file in the backend directory
2. **Connection refused**: Ensure the backend server is running on port 8000
3. **Module not found**: Install all requirements with `pip install -r requirements.txt`

### Frontend Issues

1. **WebSocket connection failed**: Check that the backend is running
2. **Microphone not working**: 
   - Ensure Chrome has microphone permissions
   - Check your system microphone settings
3. **No speech output**: Check your browser's audio settings

### Browser Automation Issues

1. **Browser doesn't open**: The browser-use package may need additional setup
2. **Commands fail**: Check the console for error messages

## Development Tips

- The browser window opens separately (not embedded) for simplicity
- Voice recognition works best in Chrome
- Speak clearly and wait for the "Hearing:" indicator
- The system uses your default microphone and speakers

## Environment Variables

Create a `.env` file in the backend directory:

```env
OPENAI_API_KEY=your-openai-api-key-here
```

## Tech Stack

- **Backend**: FastAPI, LangChain, browser-use, WebSockets
- **Frontend**: Next.js, React, Tailwind CSS, Web Speech API
- **AI**: OpenAI GPT-4

## License

This project is for educational purposes. 



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