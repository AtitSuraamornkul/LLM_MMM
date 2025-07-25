import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
import requests

# Load environment variables from .env file
load_dotenv()

# Get environment variables
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

def check_ollama_status():
    try:
        if not OLLAMA_BASE_URL:
            return False, "OLLAMA_BASE_URL not configured"
            
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return True, [model["name"] for model in models]
        else:
            return False, []
    except Exception as e:
        return False, str(e)

def get_available_ollama_models():
    try:
        if not OLLAMA_BASE_URL:
            return []
            
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        else:
            return []
    except Exception:
        return []

