#!/bin/bash

# Voice Browser Deployment Script for Render
# This script helps prepare your repository for deployment to Render

set -e  # Exit on any error

echo "🚀 Voice Browser Deployment Preparation"
echo "======================================"

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Error: This script must be run from within a git repository"
    exit 1
fi

# Get the current remote URL
REMOTE_URL=$(git config --get remote.origin.url 2>/dev/null || echo "")

if [ -z "$REMOTE_URL" ]; then
    echo "❌ Error: No git remote origin found. Please add your GitHub repository as origin first:"
    echo "   git remote add origin https://github.com/your-username/browserVoice.git"
    echo "   git push -u origin main"
    exit 1
fi

echo "📋 Current repository: $REMOTE_URL"

# Extract repository URL for render.yaml
if [[ $REMOTE_URL == *"github.com"* ]]; then
    # Convert SSH to HTTPS if needed
    if [[ $REMOTE_URL == git@* ]]; then
        HTTPS_URL=$(echo $REMOTE_URL | sed 's/git@github.com:/https:\/\/github.com\//' | sed 's/\.git$//')
    else
        HTTPS_URL=$(echo $REMOTE_URL | sed 's/\.git$//')
    fi
    
    echo "🔧 Updating render.yaml with your repository URL..."
    
    # Update render.yaml with actual repository URL
    if [ -f "render.yaml" ]; then
        sed -i.bak "s|https://github.com/your-username/browserVoice.git|$HTTPS_URL.git|g" render.yaml
        echo "✅ Updated render.yaml"
    else
        echo "❌ Error: render.yaml not found in current directory"
        exit 1
    fi
    
    # Update production render.yaml if it exists
    if [ -f "render-production.yaml" ]; then
        sed -i.bak "s|https://github.com/your-username/browserVoice.git|$HTTPS_URL.git|g" render-production.yaml
        echo "✅ Updated render-production.yaml"
    fi
    
else
    echo "❌ Error: Repository must be hosted on GitHub for Render deployment"
    exit 1
fi

# Check if required files exist
echo "🔍 Checking required files..."

REQUIRED_FILES=(
    "backend/requirements.txt"
    "backend/voice_browser_server.py"
    "web/package.json"
    "web/app/voice-browser/page.tsx"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ Found: $file"
    else
        echo "❌ Missing: $file"
        exit 1
    fi
done

# Check if .env is accidentally committed
if [ -f ".env" ]; then
    echo "⚠️  Warning: .env file found. Make sure it's not committed to git!"
    echo "   Check .gitignore includes .env"
fi

# Validate package.json has required scripts
echo "🔍 Validating package.json scripts..."
if grep -q '"build"' web/package.json && grep -q '"start"' web/package.json; then
    echo "✅ Next.js build and start scripts found"
else
    echo "❌ Error: Missing build or start scripts in web/package.json"
    exit 1
fi

# Check if changes need to be committed
if [ -n "$(git status --porcelain)" ]; then
    echo "📝 Uncommitted changes detected. Committing updated configuration..."
    git add render.yaml
    [ -f "render-production.yaml" ] && git add render-production.yaml
    git commit -m "Update render.yaml with repository URL for deployment"
fi

# Push to remote
echo "📤 Pushing changes to remote repository..."
git push origin $(git branch --show-current)

echo ""
echo "🎉 Deployment preparation complete!"
echo ""
echo "Next steps:"
echo "1. 🌐 Go to https://dashboard.render.com"
echo "2. 🆕 Click 'New' → 'Blueprint'"
echo "3. 🔗 Connect your GitHub repository: $HTTPS_URL"
echo "4. 📁 Select this repository"
echo "5. ⚡ Click 'Apply' to deploy"
echo ""
echo "During deployment, you'll be prompted for:"
echo "• OPENAI_API_KEY - Your OpenAI API key"
echo ""
echo "Files ready for deployment:"
echo "• render.yaml - Development/staging configuration"
echo "• render-production.yaml - Production configuration (optional)"
echo "• DEPLOY.md - Detailed deployment guide"
echo ""
echo "📖 For detailed instructions, see DEPLOY.md" 