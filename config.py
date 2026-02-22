import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"
PORT = int(os.getenv("PORT", "5000"))
