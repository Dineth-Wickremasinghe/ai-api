# test_api_key.py

import requests
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("❌ No API key found in .env file")
    print("Please add: GEMINI_API_KEY=your_key_here")
    exit(1)

print(f"✅ API key found (first 10 chars): {API_KEY[:10]}...")

# Test the API
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"

payload = {
    "contents": [
        {
            "parts": [
                {"text": "Say 'Hello, my API key is working!'"}
            ]
        }
    ]
}

headers = {
    "Content-Type": "application/json"
}

print("📡 Sending test request to Gemini API...")

try:
    response = requests.post(url, json=payload, headers=headers, timeout=30)

    print(f"\n📊 Response Status Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print("\n✅ SUCCESS! Your API key is VALID!")
        print(f"Response: {result['candidates'][0]['content']['parts'][0]['text']}")
    else:
        print("\n❌ FAILED! Your API key is INVALID or there's an issue.")
        print(f"Error: {response.text}")

except Exception as e:
    print(f"\n❌ Error: {e}")