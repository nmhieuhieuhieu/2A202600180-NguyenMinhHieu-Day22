import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_config():
    config = {
        "langchain_project": os.getenv("LANGCHAIN_PROJECT", "not-set"),
        "openai_base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        "api_key_set": bool(os.getenv("OPENAI_API_KEY")) and bool(os.getenv("LANGCHAIN_API_KEY"))
    }
    return config

if __name__ == "__main__":
    conf = get_config()
    if conf["api_key_set"]:
        print("✅ Config loaded successfully")
    else:
        print("⚠️ Warning: API Keys are not fully set in .env")
        
    print(f"   LangSmith project : {conf['langchain_project']}")
    print(f"   OpenAI endpoint   : {conf['openai_base_url']}")
    print(f"   Default LLM model : {conf['openai_model']}")
    print(f"   Embedding model   : {conf['embedding_model']}")
