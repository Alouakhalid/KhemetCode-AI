"""
𓂀  KhemetCode AI — API Demo
    Uses a real API key to generate code from a natural language prompt
"""

import requests
import json

# ── Config ────────────────────────────────────────────────────
API_KEY  = "0e834251-c9cf-4356-8735-092c2dbf3e94"
BASE_URL = "http://127.0.0.1:5050"
PROMPT   = "print hello world with python"

# ── Send Request ──────────────────────────────────────────────
print("\n𓂀  KhemetCode AI — Code Generation")
print("=" * 45)
print(f"  🔑 Key     : {API_KEY[:8]}...{API_KEY[-4:]}")
print(f"  📝 Prompt  : {PROMPT}")
print(f"  🌡  Temp    : 0.7")
print("=" * 45)
print("  ⏳ Sending request to model…\n")

response = requests.post(
    f"{BASE_URL}/api/generate",
    headers={
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    },
    json={
        "text":        PROMPT,
        "temperature": 0.7,
        "max_len":     120,
        "top_k":       0
    }
)

# ── Show Response ─────────────────────────────────────────────
if response.status_code == 200:
    data    = response.json()
    code    = data.get("code", "")
    metrics = data.get("metrics", {})

    print("  ✅ Response received!")
    print()
    print("┌─ Generated Code " + "─" * 27)
    for line in code.split():
        print(f"│  {line}", end=" ")
    print(f"\n└" + "─" * 44)
    print()
    print("  📊 Code Metrics:")
    print(f"     Lines      → {metrics.get('lines', '—')}")
    print(f"     Tokens     → {metrics.get('tokens', '—')}")
    print(f"     Max Depth  → {metrics.get('max_depth', '—')}")
    print(f"     Complexity → CC{metrics.get('complexity', '—')}")
    print(f"     Timestamp  → {data.get('timestamp','—')}")

elif response.status_code == 401:
    print("  ❌ Invalid or revoked API key.")
    print(f"     Response: {response.json()}")

elif response.status_code == 503:
    print("  ⏳ Model still loading — wait a moment and retry.")

else:
    print(f"  ❌ Error {response.status_code}: {response.text}")

print()
