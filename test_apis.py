import json
import requests
import time

# ===== Test Gemini API keys =====
print("=" * 60)
print("TESTING GEMINI API KEYS")
print("=" * 60)

with open('/scratch/ctoxtli/moexp/working_Gemini_API_keys.json') as f:
    keys = json.load(f)

# Test each key with a simple request
# First, list available models
for i, key in enumerate(keys):
    print(f"\n--- Key {i+1}: {key[:20]}... ---")
    
    # List models to see what's available
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            models = resp.json().get('models', [])
            gemini_models = [m['name'] for m in models if 'gemini' in m['name'].lower()]
            print(f"  Available Gemini models: {len(gemini_models)}")
            for m in gemini_models[:10]:
                print(f"    {m}")
        else:
            print(f"  List models failed: {resp.status_code} - {resp.text[:200]}")
    except Exception as e:
        print(f"  Error listing models: {e}")
    
    # Test generation with gemini-2.0-flash (cheapest)
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}"
        payload = {
            "contents": [{"parts": [{"text": "What is 2+2? Reply with just the number."}]}]
        }
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code == 200:
            result = resp.json()
            text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'No text')
            print(f"  Test generation (flash): {text.strip()[:100]}")
        else:
            print(f"  Generation failed: {resp.status_code} - {resp.text[:200]}")
    except Exception as e:
        print(f"  Error generating: {e}")
    
    time.sleep(1)  # Rate limit safety

# Check quotas for first working key
working_key = keys[0]
print(f"\n--- Checking rate limits with key 1 ---")
# Make rapid requests to see when we hit limits
for model_name in ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro']:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={working_key}"
    payload = {
        "contents": [{"parts": [{"text": "Say 'ok'"}]}]
    }
    try:
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code == 200:
            print(f"  {model_name}: AVAILABLE")
            # Check response headers for rate limit info
            for h in ['x-ratelimit-limit', 'x-ratelimit-remaining', 'x-ratelimit-reset', 'retry-after']:
                if h in resp.headers:
                    print(f"    {h}: {resp.headers[h]}")
        else:
            print(f"  {model_name}: {resp.status_code} - {resp.text[:150]}")
    except Exception as e:
        print(f"  {model_name}: Error - {e}")
    time.sleep(0.5)

# ===== Test HPC LLM API =====
print("\n" + "=" * 60)
print("TESTING HPC LLM API")
print("=" * 60)

hpc_base = "https://llm.rcd.clemson.edu/v1"

# Test without auth first
try:
    resp = requests.get(f"{hpc_base}/models", timeout=15)
    print(f"Models endpoint (no auth): {resp.status_code}")
    if resp.status_code == 200:
        models = resp.json()
        print(f"  Models: {json.dumps(models, indent=2)[:500]}")
except Exception as e:
    print(f"  Error: {e}")

# Try a test completion
for model in ['gptoss-20b', 'qwen3-30b-a3b-instruct-fp8']:
    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
            "max_tokens": 50
        }
        resp = requests.post(f"{hpc_base}/chat/completions", json=payload, timeout=30)
        print(f"\n  {model}: {resp.status_code}")
        if resp.status_code == 200:
            result = resp.json()
            text = result.get('choices', [{}])[0].get('message', {}).get('content', 'No content')
            print(f"    Response: {text[:200]}")
        else:
            print(f"    Error: {resp.text[:300]}")
    except Exception as e:
        print(f"  {model}: Error - {e}")

print("\n=== DONE ===")
