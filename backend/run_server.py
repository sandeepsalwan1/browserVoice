#!/usr/bin/env python3

import uvicorn
from app import app

if __name__ == "__main__":
    print("Starting Voice Browser Backend Server...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        log_level="info",
        reload=False
    ) 