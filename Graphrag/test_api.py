import requests
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv('OPENROUTER_API_KEY')
MODEL = 'google/gemini-2.0-flash-exp:free'  # Test this
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Say 'API works!'"}]
}

response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)
print(f"Status: {response.status_code}")
print(response.json())
