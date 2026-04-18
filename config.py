# config.py

import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = "gemini-2.5-flash"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    COLLECTION_NAME: str = "cutting_room_kb"

settings = Settings()

# ===== DEBUG: Add these lines =====
print("=" * 50)
print(f"📁 Current directory: {os.getcwd()}")
print(f"🔑 API Key loaded: {'✅ YES' if settings.GEMINI_API_KEY else '❌ NO'}")
print(f"📝 First 10 chars: {settings.GEMINI_API_KEY[:10] if settings.GEMINI_API_KEY else 'None'}")
print(f"🤖 Model: {settings.GEMINI_MODEL}")
print("=" * 50)
# =================================