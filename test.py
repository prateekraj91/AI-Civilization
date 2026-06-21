import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "qwen3:8b",
        "prompt": "/no_think Say hello",
        "stream": False
    }
)

print(response.json()["response"])