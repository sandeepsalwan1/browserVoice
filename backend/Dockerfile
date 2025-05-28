FROM python:3.12-slim

# Install system dependencies for headless Chrome
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    curl \
    xvfb \
    && wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for headless operation
ENV DISPLAY=:99
ENV HEADLESS=true
ENV CHROME_BIN=/usr/bin/google-chrome
ENV CHROME_PATH=/usr/bin/google-chrome

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install playwright browsers
RUN playwright install chromium
RUN playwright install-deps

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Start the application
CMD ["python", "-m", "uvicorn", "voice_browser_server:app", "--host", "0.0.0.0", "--port", "8000"] 