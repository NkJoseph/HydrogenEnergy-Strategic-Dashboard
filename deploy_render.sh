#!/bin/bash

# Render.com Deployment Script
# This script helps prepare and deploy to Render.com

echo "🚀 Hydrogen Dashboard - Render Deployment Script"
echo "=================================================="

# Check if git is available
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install git first."
    exit 1
fi

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Not in a git repository. Please initialize git first."
    exit 1
fi

echo "📋 Checking required files..."

# Check for required files
required_files=("api.py" "dashboard.py" "requirements.txt" "render.yaml" "runtime.txt")

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file is missing"
        exit 1
    fi
done

echo ""
echo "🔧 Preparing for deployment..."

# Add all files to git
git add .

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo "ℹ️  No changes to commit"
else
    echo "📝 Committing changes..."
    git commit -m "Prepare for Render deployment - $(date)"
fi

# Push to origin
echo "📤 Pushing to GitHub..."
git push origin main

echo ""
echo "✅ Code pushed to GitHub successfully!"
echo ""
echo "🌐 Next steps:"
echo "1. Go to https://render.com"
echo "2. Sign up/Login with GitHub"
echo "3. Click 'New +' → 'Blueprint'"
echo "4. Connect your GitHub repository"
echo "5. Render will automatically detect render.yaml"
echo "6. Review and deploy your services"
echo ""
echo "📊 Your services will be available at:"
echo "   API: https://hydrogen-api.onrender.com"
echo "   Dashboard: https://hydrogen-dashboard.onrender.com"
echo ""
echo "🔍 Monitor deployment in Render dashboard"
echo "📝 Check logs if deployment fails"
