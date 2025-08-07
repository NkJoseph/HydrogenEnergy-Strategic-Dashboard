import requests

def check_service(url, name):
    try:
        response = requests.get(url, timeout=3)
        status = "✅ Running" if response.status_code == 200 else f"❌ Error {response.status_code}"
        print(f"{name}: {status}")
        return response.status_code == 200
    except Exception as e:
        print(f"{name}: ❌ Not responding ({str(e)[:50]})")
        return False

print("🔍 Quick Status Check")
print("-" * 30)

api_ok = check_service("http://localhost:8000", "API Server    ")
dashboard_ok = check_service("http://localhost:8050", "Dashboard     ")

print("-" * 30)
if api_ok and dashboard_ok:
    print("🎉 Both services are running successfully!")
    print("📊 Dashboard: http://localhost:8050")
    print("🔧 API: http://localhost:8000")
elif api_ok:
    print("⚠️  API is running but dashboard needs to be started")
    print("💡 Try: hydrogenDashboard\\Scripts\\python.exe dashboard.py")
else:
    print("❌ Services need to be started") 