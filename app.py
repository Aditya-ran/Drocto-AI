import os
import base64
import hashlib
import html
import inspect
import json
import re
import time
from datetime import datetime

import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

# ============================================================
# DROCTO AI — GLASSY MULTILINGUAL FIRST-AID ASSISTANT
#
# Routing:
#   TEXT      -> Groq chat model
#   VOICE     -> Groq Whisper -> Groq chat model -> spoken reply
#   IMAGE     -> Groq vision -> Groq conversational reply
#   FOLLOW-UP -> Groq using stored Groq image observations
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="Drocto AI - The First Aid Helper",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------
# API / MODEL CONFIG
# --------------------------
try:
    GROQ_API_KEY = str(st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))).strip()
except Exception:
    # Local development usually uses .env; Streamlit Community Cloud uses Secrets.
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

CHAT_MODEL = os.getenv("CHAT_MODEL", "openai/gpt-oss-20b")
CHAT_FALLBACK_MODEL = os.getenv("CHAT_FALLBACK_MODEL", "llama-3.3-70b-versatile")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen3.8-27b")
STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3-turbo")

LANGUAGE_OPTIONS = {
    "auto": {"label": "🌐 Auto-detect", "whisper": None, "speech": "en-IN"},
    "english": {"label": "🇬🇧 English", "whisper": "en", "speech": "en-IN"},
    "hinglish": {"label": "🇮🇳 Hinglish", "whisper": None, "speech": "hi-IN"},
    "hindi": {"label": "🇮🇳 Hindi", "whisper": "hi", "speech": "hi-IN"},
    "assamese": {"label": "🇮🇳 Assamese (অসমীয়া)", "whisper": "as", "speech": "as-IN"},
    "bengali": {"label": "🇮🇳 Bengali (বাংলা)", "whisper": "bn", "speech": "bn-IN"},
    "bodo": {"label": "🇮🇳 Bodo (बर')", "whisper": None, "speech": "brx-IN"},
    "dogri": {"label": "🇮🇳 Dogri (डोगरी)", "whisper": None, "speech": "doi-IN"},
    "gujarati": {"label": "🇮🇳 Gujarati (ગુજરાતી)", "whisper": "gu", "speech": "gu-IN"},
    "kannada": {"label": "🇮🇳 Kannada (ಕನ್ನಡ)", "whisper": "kn", "speech": "kn-IN"},
    "kashmiri": {"label": "🇮🇳 Kashmiri (کٲشُر)", "whisper": None, "speech": "ks-IN"},
    "konkani": {"label": "🇮🇳 Konkani (कोंकणी)", "whisper": None, "speech": "kok-IN"},
    "maithili": {"label": "🇮🇳 Maithili (मैथिली)", "whisper": None, "speech": "mai-IN"},
    "malayalam": {"label": "🇮🇳 Malayalam (മലയാളം)", "whisper": "ml", "speech": "ml-IN"},
    "manipuri": {"label": "🇮🇳 Manipuri (মৈতৈলোন)", "whisper": None, "speech": "mni-IN"},
    "marathi": {"label": "🇮🇳 Marathi (मराठी)", "whisper": "mr", "speech": "mr-IN"},
    "nepali": {"label": "🇮🇳 Nepali (नेपाली)", "whisper": "ne", "speech": "ne-IN"},
    "odia": {"label": "🇮🇳 Odia (ଓଡ଼ିଆ)", "whisper": "or", "speech": "or-IN"},
    "punjabi": {"label": "🇮🇳 Punjabi (ਪੰਜਾਬੀ)", "whisper": "pa", "speech": "pa-IN"},
    "sanskrit": {"label": "🇮🇳 Sanskrit (संस्कृतम्)", "whisper": None, "speech": "sa-IN"},
    "santali": {"label": "🇮🇳 Santali (ᱥᱟᱱᱛᱟᱲᱤ)", "whisper": None, "speech": "sat-IN"},
    "sindhi": {"label": "🇮🇳 Sindhi (सिन्धी)", "whisper": None, "speech": "sd-IN"},
    "tamil": {"label": "🇮🇳 Tamil (தமிழ்)",
