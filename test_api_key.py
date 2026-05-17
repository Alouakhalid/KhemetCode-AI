"""
KhemetCode API Key Test Script
Tests: create key → generate with header → generate with body → check usage
"""

import requests
import json

BASE = "http://127.0.0.1:5050"

def sep(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print('─'*55)

# ─── 1. Create an API key ───────────────────────────────────
sep("1. Creating API Key")
r = requests.post(f"{BASE}/api/keys",
    json={"name": "Python Test Key"},
    headers={"Content-Type": "application/json"})

if r.status_code != 200:
    print(f"❌ Failed: {r.text}")
    exit(1)

key_data = r.json()
API_KEY  = key_data["key"]
print(f"✅ Key created!")
print(f"   Name    : {key_data['name']}")
print(f"   Key     : {API_KEY}")
print(f"   Created : {key_data['created']}")

# ─── 2. List all keys ───────────────────────────────────────
sep("2. Listing All Keys")
r = requests.get(f"{BASE}/api/keys")
keys = r.json()["keys"]
print(f"   Total keys: {len(keys)}")
for k in keys:
    status = "✅ Active" if k["active"] else "🔴 Revoked"
    print(f"   [{status}] {k['name']}  —  {k['key_display']}  (usage: {k['usage']})")

# ─── 3. Generate code via X-API-Key header ──────────────────
sep("3. Generate via X-API-Key Header")
prompt = "check if a list is sorted in ascending order"
print(f"   Prompt : {prompt}")

r = requests.post(f"{BASE}/api/generate",
    headers={
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    },
    json={
        "text": prompt,
        "temperature": 0.8,
        "max_len": 100,
        "top_k": 0
    })

if r.status_code == 200:
    d = r.json()
    print(f"\n   ✅ Code generated!")
    print(f"   ┌─ Generated Code {'─'*20}")
    for line in d['code'].split():
        print(f"   │  {line}", end=' ')
    print(f"\n   └{'─'*37}")
    m = d.get('metrics', {})
    print(f"\n   📊 Metrics:")
    print(f"      Lines      : {m.get('lines','—')}")
    print(f"      Tokens     : {m.get('tokens','—')}")
    print(f"      Max Depth  : {m.get('max_depth','—')}")
    print(f"      Complexity : CC{m.get('complexity','—')}")
    print(f"   🌡  Temperature: {d['temperature']}")
else:
    print(f"   ❌ Error {r.status_code}: {r.json()}")

# ─── 4. Generate via body field ─────────────────────────────
sep("4. Generate via api_key Body Field")
prompt2 = "return the size of a list"
print(f"   Prompt : {prompt2}")

r = requests.post(f"{BASE}/api/generate",
    json={
        "text": prompt2,
        "temperature": 1.0,
        "max_len": 80,
        "top_k": 10,
        "api_key": API_KEY
    })

if r.status_code == 200:
    d = r.json()
    print(f"\n   ✅ Code generated!")
    print(f"   Code   : {d['code']}")
    m = d.get('metrics', {})
    print(f"   📊 {m.get('lines','?')}L · {m.get('tokens','?')}T · CC{m.get('complexity','?')}")
else:
    print(f"   ❌ Error {r.status_code}: {r.json()}")

# ─── 5. Test with invalid key ────────────────────────────────
sep("5. Testing Invalid Key (should be rejected if keys exist)")
r = requests.post(f"{BASE}/api/generate",
    headers={"X-API-Key": "invalid-key-00000000"},
    json={"text": "test"})

if r.status_code == 401:
    print(f"   ✅ Correctly rejected invalid key (401 Unauthorized)")
    print(f"   Response: {r.json()}")
elif r.status_code == 200:
    print(f"   ⚠️  Accepted (keys not enforced when no keys in store, or key matched)")
else:
    print(f"   Status: {r.status_code} — {r.json()}")

# ─── 6. Check usage incremented ─────────────────────────────
sep("6. Verify Usage Counter")
r = requests.get(f"{BASE}/api/keys")
keys = r.json()["keys"]
for k in keys:
    if "Python Test Key" in k["name"]:
        print(f"   Key     : {k['key_display']}")
        print(f"   Usage   : {k['usage']} call(s)")
        print(f"   Last    : {k.get('last_used','—')}")
        if k['usage'] >= 1:
            print(f"   ✅ Usage counter incremented correctly!")
        else:
            print(f"   ⚠️  Usage still 0 — check increment logic")

sep("✅ All Tests Complete")
print("  Visit http://127.0.0.1:5050/keys to see keys in the UI\n")
