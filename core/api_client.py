# core/api_client.py (Unchanged)
import requests
from config.constants import API_KEY, BASE_URL, MODEL

def get_completion(messages, temperature=0.3, top_p=0.8, max_tokens=512):
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://openrouter.ai",
        "X-Title": "Streamlit-Mythalion",
    }

    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=300,
    )
    if response.status_code != 200:
        print("OpenRouter Error:", response.text)
        raise Exception(f"OpenRouter Error: {response.status_code} - {response.text}")

    return response.json()["choices"][0]["message"]["content"]
