# Voice Browser Deployment Guide

This guide explains how to deploy the Voice Browser application to Render using the included `render.yaml` blueprint.

## Prerequisites

1. **Render Account**: Sign up at [render.com](https://render.com)
2. **GitHub Repository**: Your code must be in a GitHub repository
3. **OpenAI API Key**: Required for voice processing and browser automation

## Deployment Steps

### 1. Prepare Your Repository

1. Push your code to GitHub
2. Update the `repo` URLs in `render.yaml` to point to your GitHub repository:
   ```yaml
   repo: https://github.com/YOUR-USERNAME/browserVoice.git
   ```

### 2. Deploy via Render Dashboard

1. Log into your Render Dashboard
2. Click "New" → "Blueprint"
3. Connect your GitHub repository
4. Select the repository containing your `render.yaml` file
5. Click "Apply"

### 3. Configure Environment Variables

During deployment, Render will prompt you for the `OPENAI_API_KEY`. This is required for the application to function.

**Required Environment Variables:**
- `OPENAI_API_KEY`: Your OpenAI API key

**Auto-configured Variables:**
- `PORT`: Set automatically by Render
- `PYTHONUNBUFFERED`: Configured for Python logging
- `NODE_ENV`: Set to production for frontend
- `NEXT_PUBLIC_WS_URL`: Configured to connect frontend to backend

### 4. Post-Deployment Configuration

1. **Custom Domains** (Optional): Configure custom domains in the Render Dashboard
2. **SSL Certificates**: Automatically provided by Render for all services
3. **Monitoring**: Set up monitoring and alerts in the Render Dashboard

## Architecture

The deployment creates two services:

### Backend Service (`voice-browser-backend`)
- **Runtime**: Python 3.11+
- **Framework**: FastAPI with WebSocket support
- **Features**: Voice recognition, browser automation, OpenAI integration
- **Instance Type**: Starter (upgrade to Standard/Pro for production)
- **Health Check**: `/` endpoint

### Frontend Service (`voice-browser-frontend`)
- **Runtime**: Node.js 18+
- **Framework**: Next.js 15
- **Features**: Voice interface, real-time WebSocket communication
- **Instance Type**: Starter (upgrade to Standard/Pro for production)
- **Health Check**: `/` endpoint

## Environment-Specific Settings

### Development
```bash
# Local development uses these defaults:
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
NODE_ENV=development
```

### Production (Render)
```bash
# Production automatically configures:
NEXT_PUBLIC_WS_URL=wss://voice-browser-backend.onrender.com/ws
NODE_ENV=production
```

## Scaling and Performance

### Instance Types
- **Free**: Limited, not recommended for production
- **Starter**: Good for testing and light usage ($7/month per service)
- **Standard**: Recommended for production ($25/month per service)
- **Pro**: High-performance production workloads ($85/month per service)

### Auto-scaling (Optional)
Add to `render.yaml` for automatic scaling:
```yaml
scaling:
  minInstances: 1
  maxInstances: 3
  targetCPUPercent: 70
```

## Monitoring and Debugging

### Logs Access
- View real-time logs in the Render Dashboard
- Both services provide structured logging
- WebSocket connections are logged for debugging

### Health Checks
- Backend: `https://your-backend.onrender.com/`
- Frontend: `https://your-frontend.onrender.com/`

### Common Issues

1. **WebSocket Connection Failed**
   - Check backend service is running
   - Verify `NEXT_PUBLIC_WS_URL` is correct
   - Ensure both services are in the same region

2. **OpenAI API Errors**
   - Verify `OPENAI_API_KEY` is set correctly
   - Check API key has sufficient credits
   - Review backend logs for specific error messages

3. **Browser Automation Issues**
   - Playwright browsers are installed during build
   - Instance needs sufficient memory (Standard+ recommended)
   - Check backend logs for Playwright errors

## Security Considerations

1. **Environment Variables**: Never commit API keys to version control
2. **CORS**: Backend is configured to accept connections from frontend
3. **HTTPS**: All Render services use HTTPS by default
4. **IP Restrictions**: Configure if needed in Render Dashboard

## Cost Optimization

1. **Preview Environments**: Set to manual generation to control costs
2. **Instance Sizing**: Start with Starter, upgrade as needed
3. **Auto-sleep**: Free services sleep after 15 minutes of inactivity
4. **Monitoring**: Set up alerts for unexpected usage

## Backup and Recovery

1. **Code**: Stored in GitHub repository
2. **Configuration**: Defined in `render.yaml`
3. **Environment Variables**: Document separately and store securely
4. **No Persistent Data**: Application is stateless by design

## Support and Troubleshooting

1. **Render Documentation**: [render.com/docs](https://render.com/docs)
2. **Application Logs**: Available in Render Dashboard
3. **GitHub Issues**: Report application-specific issues in your repository
4. **Render Support**: Available for paid plans

## Local Development

For local development, use:
```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn voice_browser_server:app --host 0.0.0.0 --port 8000 --reload

# Frontend
cd web
npm install
npm run dev
```

Create a `.env` file in the root directory with:
```bash
OPENAI_API_KEY=your_openai_api_key_here
``` 