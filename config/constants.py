# config/constants.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 🔐 API-related constants
API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "meta-llama/llama-3.3-70b-instruct:free"
MAX_TURNS = 6

# 📚 FAISS-related constants
TOP_K = 3                 # Number of relevant chunks to retrieve
VECTOR_DIM = 768          # Depends on the model (e.g., BGE-M3)
MEMORY_FILE = Path("data/memory.jsonl")  # Path to persistent FAISS memory

# 🧠 FAISS Memory State (for mood/intent tuning)
FAISS_MEMORY_JSON = str(Path("persona/faiss_memory_state.json"))

# 🔥 Safety check
if not API_KEY:
    print("⚠️ Warning: OPENROUTER_API_KEY not found in .env file")
