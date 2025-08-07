#!/usr/bin/env python3
"""
Deployment Helper Script for Hydrogen Dashboard
Checks project readiness and provides deployment instructions.
"""

import os
import sys
from pathlib import Path

def check_requirements():
    """Check if all required files exist."""
    required_files = [
        "api.py",
        "dashboard.py", 
        "requirements.txt",
        "render.yaml",
        "Procfile"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    return missing_files

def check_environment():
    """Check environment variables."""
    env_vars = {
        "PORT": os.environ.get("PORT", "Not set"),
        "HYDROGEN_API_URL": os.environ.get("HYDROGEN_API_URL", "Not set"),
        "DEBUG": os.environ.get("DEBUG", "Not set")
    }
    return env_vars

def main():
    print("🚀 Hydrogen Dashboard Deployment Checker")
    print("=" * 50)
    
    # Check required files
    print("\n📁 Checking required files...")
    missing = check_requirements()
    if missing:
        print(f"❌ Missing files: {', '.join(missing)}")
        return False
    else:
        print("✅ All required files present")
    
    # Check environment
    print("\n🔧 Environment variables:")
    env_vars = check_environment()
    for key, value in env_vars.items():
        status = "✅" if value != "Not set" else "❌"
        print(f"   {status} {key}: {value}")
    
    # Deployment options
    print("\n🎯 Deployment Options:")
    print("1. Render.com (Recommended)")
    print("   - Free: 750 hours/month")
    print("   - Steps: render.yaml already configured")
    print("   - URL: https://render.com")
    
    print("\n2. Railway.app")
    print("   - Free: $5 credit/month")
    print("   - Steps: Connect GitHub repo")
    print("   - URL: https://railway.app")
    
    print("\n3. Heroku")
    print("   - Free: Basic dynos")
    print("   - Steps: Procfile configured")
    print("   - URL: https://heroku.com")
    
    print("\n📋 Next Steps:")
    print("1. Push your code to GitHub")
    print("2. Choose a deployment platform")
    print("3. Follow the platform-specific steps")
    print("4. Set environment variables")
    print("5. Deploy!")
    
    print("\n📖 For detailed instructions, see: DEPLOYMENT.md")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 