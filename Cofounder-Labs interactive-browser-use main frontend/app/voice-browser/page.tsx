"use client"

import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export default function VoiceBrowserUI() {
  const [isConnected, setIsConnected] = useState(false)
  const [isListening, setIsListening] = useState(false)
  const [transcript, setTranscript] = useState("")
  const [messages, setMessages] = useState<Array<{type: string, text: string, timestamp: number}>>([])
  const [isProcessing, setIsProcessing] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const recognitionRef = useRef<any>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const reconnectAttemptsRef = useRef(0)

  // Connect to backend WebSocket
  useEffect(() => {
    connectToBackend()
    setupSpeechRecognition()
    
    return () => {
      // Clean up on unmount
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      if (wsRef.current) {
        wsRef.current.close()
      }
      if (recognitionRef.current) {
        recognitionRef.current.stop()
      }
    }
  }, [])

  const connectToBackend = () => {
    try {
      // Clear any existing connection
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.close()
      }

      const ws = new WebSocket('ws://localhost:8000/ws')
      wsRef.current = ws

      ws.onopen = () => {
        setIsConnected(true)
        reconnectAttemptsRef.current = 0
        addMessage('system', 'Connected to voice browser backend')
        // Announce connection for screen readers
        speak('Connected to voice browser. You can now give commands.')
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          
          if (data.type === 'speak') {
            speak(data.data.text)
            addMessage('assistant', data.data.text)
          } else if (data.type === 'status') {
            addMessage('status', data.data.message)
            if (data.data.message.includes('Processing:')) {
              setIsProcessing(true)
            } else if (data.data.message.includes('Ready')) {
              setIsProcessing(false)
            }
          } else if (data.type === 'error') {
            addMessage('error', data.data.message)
            setIsProcessing(false)
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error)
        }
      }

      ws.onclose = (event) => {
        setIsConnected(false)
        addMessage('system', 'Disconnected from backend')
        
        // Only announce and retry if it wasn't a clean close
        if (event.code !== 1000) {
          speak('Disconnected from voice browser backend')
          
          // Exponential backoff for reconnection
          const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 30000)
          reconnectAttemptsRef.current++
          
          addMessage('system', `Attempting to reconnect in ${delay/1000} seconds...`)
          
          reconnectTimeoutRef.current = setTimeout(() => {
            connectToBackend()
          }, delay)
        }
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        addMessage('error', 'Connection error. Please check if backend is running on port 8000.')
      }

    } catch (error) {
      console.error('Failed to connect to backend:', error)
      addMessage('error', 'Failed to connect to backend. Please ensure the backend is running with: python -m uvicorn voice_browser_server:app --host 0.0.0.0 --port 8000')
    }
  }

  const setupSpeechRecognition = () => {
    if (typeof window !== 'undefined' && 'webkitSpeechRecognition' in window) {
      const recognition = new (window as any).webkitSpeechRecognition()
      recognition.continuous = true
      recognition.interimResults = true
      recognition.lang = 'en-US'
      
      recognition.onresult = (event: any) => {
        const current = event.resultIndex
        const transcript = event.results[current][0].transcript
        
        setTranscript(transcript)
        
        // Check for final result
        if (event.results[current].isFinal) {
          const finalTranscript = transcript.trim()
          
          if (finalTranscript) {
            // Send command to backend
            sendCommand(finalTranscript)
            addMessage('user', finalTranscript)
          }
          
          setTranscript('')
        }
      }
      
      recognition.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error)
        addMessage('error', `Speech recognition error: ${event.error}`)
        setIsListening(false)
      }
      
      recognition.onend = () => {
        if (isListening && isConnected) {
          // Restart recognition if it was supposed to be listening
          recognition.start()
        }
      }
      
      recognitionRef.current = recognition
    } else {
      addMessage('error', 'Speech recognition not supported in this browser')
    }
  }

  const speak = (text: string) => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel() // Cancel any ongoing speech
      const utterance = new SpeechSynthesisUtterance(text)
      utterance.rate = 0.9
      utterance.pitch = 1
      window.speechSynthesis.speak(utterance)
    }
  }

  const sendCommand = (command: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ 
        type: 'command', 
        text: command 
      }))
    } else {
      addMessage('error', 'Not connected to backend')
    }
  }

  const addMessage = (type: string, text: string) => {
    setMessages(prev => [...prev, { 
      type, 
      text, 
      timestamp: Date.now() 
    }])
  }

  const toggleListening = () => {
    if (!recognitionRef.current) {
      speak('Speech recognition not available')
      return
    }

    if (!isConnected) {
      speak('Not connected to backend')
      return
    }

    if (isListening) {
      recognitionRef.current.stop()
      setIsListening(false)
      speak('Stopped listening')
    } else {
      try {
        recognitionRef.current.start()
        setIsListening(true)
        speak('Started listening for commands')
      } catch (error) {
        console.error('Failed to start recognition:', error)
        speak('Failed to start listening')
      }
    }
  }

  const cancelCurrentTask = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ 
        type: 'cancel', 
        text: '' 
      }))
      speak('Cancelling current task')
    }
  }

  // Get latest message for screen readers
  const latestMessage = messages.length > 0 ? messages[messages.length - 1] : null

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-4xl mx-auto">
        <main role="main" aria-live="polite">
          <Card>
            <CardHeader>
              <CardTitle>
                Voice Browser Control
                <span className="sr-only">
                  {isConnected ? 'Connected to backend' : 'Disconnected from backend'}
                </span>
              </CardTitle>
              <div className="flex items-center gap-2">
                <div 
                  className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}
                  aria-hidden="true"
                />
                <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
                {!isConnected && (
                  <Button
                    onClick={connectToBackend}
                    size="sm"
                    variant="outline"
                  >
                    Retry Connection
                  </Button>
                )}
              </div>
            </CardHeader>
            
            <CardContent className="space-y-4">
              {/* Main Control Buttons */}
              <div className="flex gap-4">
                <Button
                  onClick={toggleListening}
                  disabled={!isConnected}
                  className={`${isListening ? 'bg-red-600 hover:bg-red-700' : 'bg-blue-600 hover:bg-blue-700'} text-white px-6 py-3 text-lg`}
                  aria-describedby="listening-status"
                >
                  {isListening ? 'Stop Listening' : 'Start Listening'}
                </Button>
                
                {isProcessing && (
                  <Button
                    onClick={cancelCurrentTask}
                    className="bg-orange-600 hover:bg-orange-700 text-white px-6 py-3 text-lg"
                  >
                    Cancel Task
                  </Button>
                )}
              </div>

              <div id="listening-status" className="sr-only">
                {isListening ? 'Currently listening for voice commands' : 'Not listening'}
                {transcript && `, current transcript: ${transcript}`}
              </div>

              {/* Current Transcript */}
              {transcript && (
                <div className="p-4 bg-blue-50 border border-blue-200 rounded">
                  <p><strong>Hearing:</strong> {transcript}</p>
                </div>
              )}

              {/* Status Messages */}
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {messages.slice(-10).map((msg, idx) => (
                  <div
                    key={`${msg.timestamp}-${idx}`}
                    className={`p-3 rounded ${
                      msg.type === 'user' ? 'bg-blue-100 border-l-4 border-blue-500' :
                      msg.type === 'assistant' ? 'bg-green-100 border-l-4 border-green-500' :
                      msg.type === 'error' ? 'bg-red-100 border-l-4 border-red-500' :
                      msg.type === 'system' ? 'bg-gray-100 border-l-4 border-gray-500' :
                      'bg-yellow-100 border-l-4 border-yellow-500'
                    }`}
                    role={msg.type === 'error' ? 'alert' : undefined}
                  >
                    <p>
                      <strong className="capitalize">{msg.type}:</strong> {msg.text}
                    </p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Instructions */}
          <Card className="mt-6">
            <CardHeader>
              <CardTitle>How to Use Voice Browser</CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2">
                <li>• Click "Start Listening" to begin voice control</li>
                <li>• Say commands like "Search for Python tutorials" or "Go to Google"</li>
                <li>• Say "Cancel" to stop the current task</li>
                <li>• The browser window will open separately for automation</li>
                <li>• All actions and results will be read aloud</li>
              </ul>
            </CardContent>
          </Card>

          {/* Hidden live region for screen reader announcements */}
          <div aria-live="assertive" aria-atomic="true" className="sr-only">
            {latestMessage && latestMessage.type === 'assistant' && latestMessage.text}
          </div>
        </main>
      </div>
    </div>
  )
} 