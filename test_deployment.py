#!/usr/bin/env python3
"""
Deployment Test Script for Hydrogen Dashboard
Tests deployment readiness without requiring installed packages.
"""

import os
import sys
from pathlib import Path

def test_files():
    """Test if all required files exist."""
    print("🔍 Testing required files...")
    
    required_files = [
        "api.py",
        "dashboard.py",
        "requirements.txt",
        "render.yaml",
        "Procfile",
        "runtime.txt",
        ".gitignore"
    ]
    
    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
    
    if missing:
        print(f"❌ Missing files: {missing}")
        return False
    
    print("✅ All required files present")
    return True

def test_requirements():
    """Test if requirements.txt is valid."""
    print("\n🔍 Testing requirements.txt...")
    
    req_file = Path("requirements.txt")
    if not req_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        with open(req_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Check for essential packages
        essential = ['fastapi', 'uvicorn', 'dash', 'plotly']
        found = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                package = line.split('==')[0].split('>=')[0].split('<=')[0]
                found.append(package)
        
        missing = [pkg for pkg in essential if pkg not in found]
        if missing:
            print(f"❌ Missing packages: {missing}")
            return False
        
        print("✅ requirements.txt is valid")
        return True
    except Exception as e:
        print(f"❌ Requirements test failed: {e}")
        return False

def test_code_syntax():
    """Test if Python files have valid syntax."""
    print("\n🔍 Testing code syntax...")
    
    python_files = ["api.py", "dashboard.py"]
    syntax_errors = []
    
    for file in python_files:
        if not Path(file).exists():
            syntax_errors.append(f"{file} not found")
            continue
            
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic syntax check
            compile(content, file, 'exec')
            print(f"✅ {file} syntax is valid")
        except SyntaxError as e:
            syntax_errors.append(f"{file}: {e}")
        except Exception as e:
            syntax_errors.append(f"{file}: {e}")
    
    if syntax_errors:
        print(f"❌ Syntax errors found: {syntax_errors}")
        return False
    
    print("✅ All Python files have valid syntax")
    return True

def test_render_config():
    """Test if render.yaml is valid."""
    print("\n🔍 Testing render.yaml...")
    
    render_file = Path("render.yaml")
    if not render_file.exists():
        print("❌ render.yaml not found")
        return False
    
    try:
        with open(render_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for essential components
        required_sections = ['services:', 'type: web', 'name: hydrogen-api', 'name: hydrogen-dashboard']
        missing = []
        
        for section in required_sections:
            if section not in content:
                missing.append(section)
        
        if missing:
            print(f"❌ Missing sections in render.yaml: {missing}")
            return False
        
        print("✅ render.yaml is valid")
        return True
    except Exception as e:
        print(f"❌ render.yaml test failed: {e}")
        return False

def test_environment():
    """Test environment variables."""
    print("\n🔍 Testing environment...")
    
    # These will be set by Render
    env_vars = {
        "PORT": os.environ.get("PORT", "Not set"),
        "HYDROGEN_API_URL": os.environ.get("HYDROGEN_API_URL", "Not set"),
    }
    
    print("Environment variables (will be set by Render):")
    for key, value in env_vars.items():
        status = "⚠️" if value == "Not set" else "✅"
        print(f"   {status} {key}: {value}")
    
    return True

def test_git_status():
    """Test if this is a git repository."""
    print("\n🔍 Testing git status...")
    
    git_dir = Path(".git")
    if not git_dir.exists():
        print("⚠️ Not a git repository - you'll need to initialize git before deployment")
        return True  # Not critical for deployment
    
    print("✅ Git repository detected")
    return True

def main():
    """Run all tests."""
    print("🚀 Hydrogen Dashboard Deployment Tests")
    print("=" * 50)
    
    tests = [
        test_files,
        test_requirements,
        test_code_syntax,
        test_render_config,
        test_environment,
        test_git_status
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print("🎉 All tests passed! Ready for deployment.")
        print("\n📋 Next Steps:")
        print("1. Push code to GitHub")
        print("2. Go to https://render.com")
        print("3. Create new Blueprint")
        print("4. Connect your GitHub repo")
        print("5. Deploy!")
        return True
    else:
        print(f"❌ {total - passed} test(s) failed. Please fix issues before deploying.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 