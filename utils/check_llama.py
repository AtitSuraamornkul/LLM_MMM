from langchain_ollama import ChatOllama
import requests

def check_ollama_status():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return True, [model["name"] for model in models]
        else:
            return False, []
    except Exception as e:
        return False, str(e)
    
def get_available_ollama_models():
    """Get list of available Ollama models"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        else:
            return []
    except Exception:
        return []