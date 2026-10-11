import os
import base64
import hashlib
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
    "tamil": {"label": "🇮🇳 Tamil (தமிழ்)", "whisper": "ta", "speech": "ta-IN"},
    "telugu": {"label": "🇮🇳 Telugu (తెలుగు)", "whisper": "te", "speech": "te-IN"},
    "urdu": {"label": "🇮🇳 Urdu (اردو)", "whisper": "ur", "speech": "ur-IN"},
}

WHISPER_LANGUAGE_MAP = {
    "en": "english", "english": "english",
    "hi": "hindi", "hindi": "hindi",
    "as": "assamese", "assamese": "assamese",
    "bn": "bengali", "bengali": "bengali",
    "gu": "gujarati", "gujarati": "gujarati",
    "kn": "kannada", "kannada": "kannada",
    "ml": "malayalam", "malayalam": "malayalam",
    "mr": "marathi", "marathi": "marathi",
    "ne": "nepali", "nepali": "nepali",
    "or": "odia", "odia": "odia", "oriya": "odia",
    "pa": "punjabi", "punjabi": "punjabi",
    "ta": "tamil", "tamil": "tamil",
    "te": "telugu", "telugu": "telugu",
    "ur": "urdu", "urdu": "urdu",
}

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

MAX_HISTORY_MESSAGES = 12
REQUEST_TIMEOUT = 45

# ============================================================
# THEME DEFINITIONS
# ============================================================
THEMES = {
    "dark": {
        "TEXT": "#e8eefb",
        "MUTED": "#94a7c4",
        "GLASS": "rgba(255, 255, 255, 0.055)",
        "GLASS_STRONG": "rgba(255, 255, 255, 0.095)",
        "GLASS_CARD": "rgba(255, 255, 255, 0.045)",
        "BORDER": "rgba(255, 255, 255, 0.16)",
        "BORDER_INPUT": "rgba(255, 255, 255, 0.20)",
        "BORDER_SOFT": "rgba(255, 255, 255, 0.09)",
        "SHADOW": "0 10px 34px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.07)",
        "SHADOW_INPUT": (
            "0 14px 40px rgba(0,0,0,0.35), "
            "0 2px 8px rgba(0,0,0,0.18), "
            "inset 0 1px 0 rgba(255,255,255,0.14), "
            "inset 0 -1px 0 rgba(0,0,0,0.10)"
        ),
        "SHADOW_CARD": "0 8px 26px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.05)",
        "SHEEN": (
            "linear-gradient(115deg, "
            "rgba(255,255,255,0.16) 0%, "
            "rgba(255,255,255,0.06) 22%, "
            "rgba(255,255,255,0) 46%, "
            "rgba(255,255,255,0) 60%, "
            "rgba(255,255,255,0.05) 78%, "
            "rgba(255,255,255,0.12) 100%)"
        ),
        "TOPLINE": (
            "linear-gradient(90deg, "
            "transparent 0%, rgba(255,255,255,0.55) 50%, transparent 100%)"
        ),
        "ACCENT": "#5eead4",
        "ACCENT2": "#a78bfa",
        "ACCENT_SOFT": "rgba(94,234,212,0.14)",
        "ACCENT2_SOFT": "rgba(167,139,250,0.14)",
        "PILL_BG": "rgba(255,255,255,0.055)",
        "PILL_HOVER": "rgba(255,255,255,0.12)",
        "BG": (
            "radial-gradient(1100px 620px at 8% -12%, rgba(56,189,248,0.22), transparent 62%), "
            "radial-gradient(980px 700px at 92% -6%, rgba(167,139,250,0.22), transparent 62%), "
            "radial-gradient(900px 820px at 50% 112%, rgba(45,212,191,0.18), transparent 62%), "
            "linear-gradient(160deg, #070b14 0%, #0b1220 48%, #050810 100%)"
        ),
    },
    "light": {
        "TEXT": "#0f1b2d",
        "MUTED": "#5a6b86",
        "GLASS": "rgba(255, 255, 255, 0.55)",
        "GLASS_STRONG": "rgba(255, 255, 255, 0.78)",
        "GLASS_CARD": "rgba(255, 255, 255, 0.62)",
        "BORDER": "rgba(15, 27, 45, 0.10)",
        "BORDER_INPUT": "rgba(15, 27, 45, 0.14)",
        "BORDER_SOFT": "rgba(15, 27, 45, 0.06)",
        "SHADOW": "0 10px 34px rgba(31,54,90,0.14), inset 0 1px 0 rgba(255,255,255,0.85)",
        "SHADOW_INPUT": (
            "0 14px 40px rgba(31,54,90,0.18), "
            "0 2px 8px rgba(31,54,90,0.10), "
            "inset 0 1px 0 rgba(255,255,255,0.95), "
            "inset 0 -1px 0 rgba(15,27,45,0.05)"
        ),
        "SHADOW_CARD": "0 8px 26px rgba(31,54,90,0.10), inset 0 1px 0 rgba(255,255,255,0.90)",
        "SHEEN": (
            "linear-gradient(115deg, "
            "rgba(255,255,255,0.75) 0%, "
            "rgba(255,255,255,0.35) 22%, "
            "rgba(255,255,255,0) 46%, "
            "rgba(255,255,255,0) 60%, "
            "rgba(255,255,255,0.35) 78%, "
            "rgba(255,255,255,0.70) 100%)"
        ),
        "TOPLINE": (
            "linear-gradient(90deg, "
            "transparent 0%, rgba(255,255,255,1) 50%, transparent 100%)"
        ),
        "ACCENT": "#0d9488",
        "ACCENT2": "#6d28d9",
        "ACCENT_SOFT": "rgba(13,148,136,0.12)",
        "ACCENT2_SOFT": "rgba(109,40,217,0.12)",
        "PILL_BG": "rgba(255,255,255,0.72)",
        "PILL_HOVER": "rgba(255,255,255,0.95)",
        "BG": (
            "radial-gradient(1100px 620px at 8% -12%, rgba(56,189,248,0.35), transparent 62%), "
            "radial-gradient(980px 700px at 92% -6%, rgba(167,139,250,0.30), transparent 62%), "
            "radial-gradient(900px 820px at 50% 112%, rgba(45,212,191,0.28), transparent 62%), "
            "linear-gradient(160deg, #eef4ff 0%, #f8faff 48%, #e9f1ff 100%)"
        ),
    },
}

CSS_TEMPLATE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root{
  --fa-text: __TEXT__;
  --fa-muted: __MUTED__;
  --fa-glass: __GLASS__;
  --fa-glass-strong: __GLASS_STRONG__;
  --fa-glass-card: __GLASS_CARD__;
  --fa-border: __BORDER__;
  --fa-border-input: __BORDER_INPUT__;
  --fa-border-soft: __BORDER_SOFT__;
  --fa-shadow: __SHADOW__;
  --fa-shadow-input: __SHADOW_INPUT__;
  --fa-shadow-card: __SHADOW_CARD__;
  --fa-sheen: __SHEEN__;
  --fa-topline: __TOPLINE__;
  --fa-accent: __ACCENT__;
  --fa-accent2: __ACCENT2__;
  --fa-accent-soft: __ACCENT_SOFT__;
  --fa-accent2-soft: __ACCENT2_SOFT__;
  --fa-pill-bg: __PILL_BG__;
  --fa-pill-hover: __PILL_HOVER__;
  --fa-bg: __BG__;
}

/* ══════════════ APP SHELL ══════════════ */
html, body, [class*="css"]{
  font-family: "Inter","Segoe UI",-apple-system,BlinkMacSystemFont,sans-serif;
}
.stApp{
  background: var(--fa-bg) !important;
  background-attachment: fixed !important;
  color: var(--fa-text) !important;
  min-height: 100vh;
}
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
section.main{ background: transparent !important; }
[data-testid="stHeader"],
[data-testid="stToolbar"]{
  background: transparent !important;
  background-color: transparent !important;
  box-shadow: none !important;
}
[data-testid="stDecoration"]{ display:none !important; }
#MainMenu, footer{ visibility: hidden; }

.block-container{
  max-width: 1100px;
  padding-top: 1.05rem;
  padding-bottom: 6rem;
}

/* ══════════════ KILL BOTTOM BLACK BAND ══════════════ */
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottom"] > div > div,
[data-testid="stBottomBlockContainer"],
[data-testid="stBottomBlockContainer"] > div,
[data-testid="stBottomBlockContainer"] > div > div,
.stBottom,
.stBottomBlockContainer{
  background: transparent !important;
  background-color: transparent !important;
  background-image: none !important;
  border: none !important;
  box-shadow: none !important;
}
[data-testid="stBottom"]::before,
[data-testid="stBottom"]::after,
[data-testid="stBottomBlockContainer"]::before,
[data-testid="stBottomBlockContainer"]::after{
  display: none !important;
  background: transparent !important;
  content: none !important;
}

/* ══════════════ TYPOGRAPHY ══════════════ */
.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp h5,.stApp h6{
  color: var(--fa-text) !important;
  letter-spacing: -0.02em;
}
.stApp p,.stApp li,.stApp label,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stWidgetLabel"] p{
  color: var(--fa-text) !important;
}
[data-testid="stCaptionContainer"] p, .stCaption, small{
  color: var(--fa-muted) !important;
}
.stApp a{ color: var(--fa-accent) !important; }

/* ══════════════ HERO ══════════════ */
.fa-hero{
  position: relative;
  padding: 34px 34px 30px;
  border-radius: 26px;
  background: linear-gradient(
    135deg,
    var(--fa-accent-soft) 0%,
    var(--fa-glass-card) 45%,
    var(--fa-accent2-soft) 100%
  );
  border: 1px solid var(--fa-border);
  backdrop-filter: blur(24px) saturate(160%);
  -webkit-backdrop-filter: blur(24px) saturate(160%);
  box-shadow: var(--fa-shadow);
  overflow: hidden;
  margin-bottom: 22px;
}
.fa-hero::before{
  content: "";
  position: absolute;
  inset: 0;
  background: var(--fa-sheen);
  pointer-events: none;
  border-radius: inherit;
  opacity: 0.7;
}
.fa-hero::after{
  content: "";
  position: absolute;
  top: 0; left: 18px; right: 18px;
  height: 1px;
  background: var(--fa-topline);
  pointer-events: none;
}
.fa-hero-inner{ position: relative; z-index: 2; }
.fa-hero-badge{
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 5px 12px;
  border-radius: 999px;
  background: var(--fa-glass-strong);
  border: 1px solid var(--fa-border);
  font-size: 12px;
  color: var(--fa-text);
  margin-bottom: 14px;
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
}
.fa-hero-badge .dot{
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--fa-accent);
  box-shadow: 0 0 10px var(--fa-accent);
}
.fa-hero-title{
  font-size: 42px;
  font-weight: 800;
  letter-spacing: -0.03em;
  line-height: 1.08;
  margin: 0 0 10px 0;
  background: linear-gradient(100deg, var(--fa-accent), var(--fa-accent2));
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}
.fa-hero-sub{
  color: var(--fa-muted);
  font-size: 15px;
  line-height: 1.55;
  max-width: 620px;
  margin: 0 0 20px 0;
}
.fa-hero-tags{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.fa-tag{
  padding: 6px 12px;
  border-radius: 999px;
  background: var(--fa-pill-bg);
  border: 1px solid var(--fa-border-soft);
  font-size: 12.5px;
  color: var(--fa-text);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  transition: all 0.18s ease;
}
.fa-tag:hover{
  background: var(--fa-pill-hover);
  border-color: var(--fa-accent);
  color: var(--fa-accent);
  transform: translateY(-1px);
}

/* ══════════════ INFO STRIP ══════════════ */
.fa-info{
  margin: 0 0 22px 0;
  padding: 14px 17px;
  border-radius: 18px;
  background: var(--fa-glass-card);
  border: 1px solid var(--fa-border-soft);
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  box-shadow: var(--fa-shadow-card);
  color: var(--fa-text);
  font-size: 13.5px;
  line-height: 1.5;
}
.fa-info .dot{
  color: var(--fa-accent);
  text-shadow: 0 0 14px var(--fa-accent);
  margin-right: 7px;
}

/* ══════════════ SECTION LABELS ══════════════ */
.fa-section-label{
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12.5px;
  font-weight: 700;
  letter-spacing: 0.10em;
  text-transform: uppercase;
  color: var(--fa-muted);
  margin: 4px 0 12px 2px;
}
.fa-section-label::after{
  content: "";
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, var(--fa-border), transparent);
}

/* ══════════════ CHIP BUTTONS ══════════════ */
[class*="st-key-chip_"] button,
div[class*="st-key-chip_"] > .stButton > button{
  width: 100% !important;
  min-height: 92px !important;
  padding: 16px 18px !important;
  border-radius: 16px !important;
  background: var(--fa-glass-card) !important;
  border: 1px solid var(--fa-border-soft) !important;
  backdrop-filter: blur(16px) saturate(150%) !important;
  -webkit-backdrop-filter: blur(16px) saturate(150%) !important;
  box-shadow: var(--fa-shadow-card) !important;
  color: var(--fa-text) !important;
  font-weight: 700 !important;
  font-size: 14px !important;
  text-align: left !important;
  justify-content: flex-start !important;
  white-space: normal !important;
  line-height: 1.5 !important;
  transition: all 0.2s ease !important;
}
[class*="st-key-chip_"] button p,
[class*="st-key-chip_"] button div{
  text-align: left !important;
  width: 100% !important;
  color: inherit !important;
}
[class*="st-key-chip_"] button:hover{
  transform: translateY(-2px) !important;
  border-color: color-mix(in srgb, var(--fa-accent) 55%, var(--fa-border)) !important;
  box-shadow: 0 12px 30px -10px var(--fa-accent), var(--fa-shadow-card) !important;
  color: var(--fa-accent) !important;
}
[class*="st-key-chip_"] button:focus:not(:active){
  color: var(--fa-accent) !important;
  border-color: var(--fa-accent) !important;
}

/* ══════════════ FEATURE CARDS ══════════════ */
.fa-features{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 14px;
  margin-bottom: 26px;
}
.fa-feature{
  padding: 18px;
  border-radius: 18px;
  background: var(--fa-glass-card);
  border: 1px solid var(--fa-border-soft);
  backdrop-filter: blur(18px);
  -webkit-backdrop-filter: blur(18px);
  box-shadow: var(--fa-shadow-card);
  position: relative;
  overflow: hidden;
  transition: all 0.2s ease;
}
.fa-feature::before{
  content: "";
  position: absolute;
  inset: 0;
  background: var(--fa-sheen);
  opacity: 0.28;
  pointer-events: none;
}
.fa-feature:hover{
  border-color: color-mix(in srgb, var(--fa-accent2) 45%, var(--fa-border));
  transform: translateY(-2px);
}
.fa-feature-ico{
  width: 38px; height: 38px;
  border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, var(--fa-accent-soft), var(--fa-accent2-soft));
  border: 1px solid var(--fa-border);
  font-size: 18px;
  margin-bottom: 10px;
  position: relative;
  z-index: 1;
}
.fa-feature-title{
  font-size: 14px;
  font-weight: 700;
  margin: 0 0 5px 0;
  color: var(--fa-text);
  position: relative;
  z-index: 1;
}
.fa-feature-text{
  font-size: 12.5px;
  color: var(--fa-muted);
  line-height: 1.5;
  margin: 0;
  position: relative;
  z-index: 1;
}

/* ══════════════ SIDEBAR ══════════════ */
[data-testid="stSidebar"]{
  background: var(--fa-glass) !important;
  backdrop-filter: blur(22px) saturate(160%);
  -webkit-backdrop-filter: blur(22px) saturate(160%);
  border-right: 1px solid var(--fa-border) !important;
  box-shadow: var(--fa-shadow);
}
[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"]{ background: transparent !important; }

.fa-side-card{
  border: 1px solid var(--fa-border);
  border-radius: 20px;
  padding: 16px 17px;
  background: var(--fa-glass-card);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  box-shadow: var(--fa-shadow-card);
  margin-bottom: 14px;
}
.fa-side-title{
  font-size: 17px;
  font-weight: 800;
  color: var(--fa-text);
  letter-spacing: -0.01em;
}
.fa-side-copy{
  margin-top: 4px;
  font-size: 12.5px;
  line-height: 1.45;
  color: var(--fa-muted);
}
.fa-side-note{
  margin-top: 10px;
  padding-top: 12px;
  border-top: 1px solid var(--fa-border-soft);
  color: var(--fa-muted);
  font-size: 11.5px;
  line-height: 1.55;
}

/* ══════════════ CHAT MESSAGES ══════════════ */
[data-testid="stChatMessage"]{
  background: var(--fa-glass) !important;
  border: 1px solid var(--fa-border) !important;
  border-radius: 18px !important;
  padding: 14px 18px !important;
  backdrop-filter: blur(18px) saturate(150%);
  -webkit-backdrop-filter: blur(18px) saturate(150%);
  box-shadow: var(--fa-shadow);
  margin-bottom: 12px;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]){
  border-color: color-mix(in srgb, var(--fa-accent) 45%, var(--fa-border)) !important;
}
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"]{
  background: var(--fa-glass-strong) !important;
  border: 1px solid var(--fa-border) !important;
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
}
[data-testid="stChatMessage"] p{ line-height: 1.62; }

/* ══════════════ CAP CHAT IMAGE SIZE ══════════════ */
[data-testid="stChatMessage"] [data-testid="stImage"],
[data-testid="stChatMessage"] [data-testid="stImageContainer"],
[data-testid="stChatMessage"] [data-testid="stImage"] > img,
[data-testid="stChatMessage"] img{
  max-width: 240px !important;
  max-height: 240px !important;
  width: auto !important;
  height: auto !important;
  display: block !important;
}
[data-testid="stChatMessage"] [data-testid="stImage"] img{
  border-radius: 14px !important;
  border: 1px solid var(--fa-border) !important;
  box-shadow: var(--fa-shadow) !important;
}
[data-testid="stChatMessage"] [data-testid="stImage"] figcaption,
[data-testid="stChatMessage"] [data-testid="stImageCaption"]{
  font-size: 11.5px !important;
  color: var(--fa-muted) !important;
  margin-top: 4px !important;
}

/* ══════════════ GLOSSY CHAT INPUT ══════════════ */
[data-testid="stChatInput"],
[data-testid="stChatInputContainer"],
.stChatInputContainer{
  background: linear-gradient(
    180deg,
    var(--fa-glass-strong) 0%,
    var(--fa-glass) 100%
  ) !important;
  border: 1px solid var(--fa-border-input) !important;
  border-radius: 20px !important;
  backdrop-filter: blur(26px) saturate(180%);
  -webkit-backdrop-filter: blur(26px) saturate(180%);
  box-shadow: var(--fa-shadow-input) !important;
  position: relative;
  overflow: hidden;
  transition: border-color 0.25s ease, box-shadow 0.25s ease;
}
[data-testid="stChatInput"]:focus-within,
[data-testid="stChatInputContainer"]:focus-within,
.stChatInputContainer:focus-within{
  border-color: color-mix(in srgb, var(--fa-accent) 55%, var(--fa-border-input)) !important;
  box-shadow:
    var(--fa-shadow-input),
    0 0 0 4px color-mix(in srgb, var(--fa-accent) 18%, transparent) !important;
}
[data-testid="stChatInput"]::before,
[data-testid="stChatInputContainer"]::before,
.stChatInputContainer::before{
  content: "";
  position: absolute;
  inset: 0;
  background: var(--fa-sheen);
  pointer-events: none;
  border-radius: inherit;
  z-index: 0;
}
[data-testid="stChatInput"]::after,
[data-testid="stChatInputContainer"]::after,
.stChatInputContainer::after{
  content: "";
  position: absolute;
  top: 0; left: 14px; right: 14px;
  height: 1px;
  background: var(--fa-topline);
  pointer-events: none;
  z-index: 1;
}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInputContainer"] textarea,
.stChatInputContainer textarea,
[data-testid="stChatInput"] > div,
[data-testid="stChatInputContainer"] > div{
  background: transparent !important;
  color: var(--fa-text) !important;
  position: relative;
  z-index: 2;
}
[data-testid="stChatInput"] textarea::placeholder,
[data-testid="stChatInputContainer"] textarea::placeholder,
.stChatInputContainer textarea::placeholder{
  color: var(--fa-muted) !important;
}
[data-testid="stChatInput"] button,
[data-testid="stChatInputContainer"] button,
.stChatInputContainer button{
  background: linear-gradient(
    135deg,
    color-mix(in srgb, var(--fa-accent) 22%, var(--fa-glass-strong)) 0%,
    color-mix(in srgb, var(--fa-accent2) 22%, var(--fa-glass-strong)) 100%
  ) !important;
  color: var(--fa-text) !important;
  border: 1px solid color-mix(in srgb, var(--fa-accent) 35%, var(--fa-border-input)) !important;
  border-radius: 12px !important;
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.25),
              inset 0 1px 0 rgba(255,255,255,0.18);
  transition: all 0.18s ease;
  position: relative;
  z-index: 2;
}
[data-testid="stChatInput"] button:hover,
[data-testid="stChatInputContainer"] button:hover,
.stChatInputContainer button:hover{
  color: var(--fa-accent) !important;
  border-color: var(--fa-accent) !important;
  transform: translateY(-1px);
}
[data-testid="stChatInput"] button svg,
[data-testid="stChatInputContainer"] button svg,
.stChatInputContainer button svg{
  fill: currentColor !important;
}

/* ══════════════ GENERAL BUTTONS ══════════════ */
.stButton > button,
.stDownloadButton > button,
.stFormSubmitButton > button{
  background: var(--fa-glass) !important;
  color: var(--fa-text) !important;
  border: 1px solid var(--fa-border) !important;
  border-radius: 14px !important;
  backdrop-filter: blur(14px) saturate(150%);
  -webkit-backdrop-filter: blur(14px) saturate(150%);
  box-shadow: var(--fa-shadow);
  transition: all .18s ease;
  font-weight: 600;
}
.stButton > button:hover,
.stDownloadButton > button:hover,
.stFormSubmitButton > button:hover{
  border-color: var(--fa-accent) !important;
  color: var(--fa-accent) !important;
  transform: translateY(-1px);
  box-shadow: 0 10px 28px -10px var(--fa-accent);
}
.stButton > button:focus:not(:active),
.stButton > button:active{
  border-color: var(--fa-accent) !important;
  color: var(--fa-accent) !important;
}

/* ══════════════ INPUTS / SELECT ══════════════ */
.stTextInput input, .stTextArea textarea{
  color: var(--fa-text) !important;
  background: var(--fa-glass-strong) !important;
  border: 1px solid var(--fa-border) !important;
  border-radius: 14px !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder{
  color: var(--fa-muted) !important;
}

[data-testid="stSelectbox"] > div > div{
  background: var(--fa-glass-strong) !important;
  border: 1px solid var(--fa-border) !important;
  border-radius: 12px !important;
  color: var(--fa-text) !important;
}
[data-testid="stSelectbox"] *{
  color: var(--fa-text) !important;
}

/* ══════════════ TOGGLE ══════════════ */
[data-testid="stToggle"]{ background: transparent !important; }

/* ══════════════ ALERTS / CODE / DIVIDER ══════════════ */
[data-testid="stAlert"]{
  background: var(--fa-glass) !important;
  border: 1px solid var(--fa-border) !important;
  border-radius: 16px !important;
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
}
[data-testid="stAlert"] *{ color: var(--fa-text) !important; }

pre, code, [data-testid="stCode"], [data-testid="stCodeBlock"]{
  background: var(--fa-glass-strong) !important;
  color: var(--fa-text) !important;
  border: 1px solid var(--fa-border) !important;
  border-radius: 14px !important;
}
hr, [data-testid="stDivider"]{ border-color: var(--fa-border) !important; }

/* ══════════════ FILE UPLOADER / AUDIO ══════════════ */
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"]{
  background: var(--fa-glass) !important;
  border: 1px dashed var(--fa-border) !important;
  border-radius: 16px !important;
}
[data-testid="stFileUploaderDropzone"] *{ color: var(--fa-text) !important; }

[data-testid="stAudioInput"] button{
  background: var(--fa-glass-strong) !important;
  border: 1px solid var(--fa-border) !important;
  border-radius: 14px !important;
  color: var(--fa-text) !important;
}

/* ══════════════ SPINNER / STATUS ══════════════ */
.stSpinner > div{ border-top-color: var(--fa-accent) !important; }
.stSpinner p{ color: var(--fa-muted) !important; }

/* ══════════════ FOOTER ══════════════ */
.fa-footer{
  margin-top: 18px;
  padding-top: 12px;
  border-top: 1px solid var(--fa-border-soft);
  color: var(--fa-muted);
  font-size: 11.5px;
  display: flex;
  justify-content: space-between;
}

/* ══════════════ SCROLLBAR ══════════════ */
::-webkit-scrollbar{ width: 10px; height: 10px; }
::-webkit-scrollbar-track{ background: transparent; }
::-webkit-scrollbar-thumb{
  background: var(--fa-border);
  border-radius: 999px;
  border: 2px solid transparent;
  background-clip: content-box;
}
::-webkit-scrollbar-thumb:hover{
  background: var(--fa-accent);
  background-clip: content-box;
}
</style>
"""


def inject_css(mode: str) -> None:
    palette = THEMES.get(mode, THEMES["dark"])
    css = CSS_TEMPLATE
    for key, value in palette.items():
        css = css.replace(f"__{key}__", value)
    st.markdown(css, unsafe_allow_html=True)


# ============================================================
# ERROR HELPERS
# ============================================================
def status_code(response_or_exc):
    for attr in ("status_code", "code", "status", "http_status"):
        value = getattr(response_or_exc, attr, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def api_error_text(response):
    try:
        data = response.json()
        return data.get("error", {}).get("message", response.text)
    except Exception:
        return response.text


# ============================================================
# ENVIRONMENT CONTEXT
# ============================================================
@st.cache_data(ttl=1800, show_spinner=False)
def geocode(location):
    response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": location, "count": 5, "language": "en", "format": "json"},
        timeout=10,
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    if not results:
        return None
    wanted = location.strip().lower()
    chosen = next((x for x in results if x.get("name", "").lower() == wanted), results[0])
    return {
        "name": chosen.get("name", location),
        "country": chosen.get("country", ""),
        "latitude": chosen["latitude"],
        "longitude": chosen["longitude"],
    }


@st.cache_data(ttl=900, show_spinner=False)
def environment_context(lat, lon):
    weather = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,wind_speed_10m,uv_index",
            "forecast_days": 1,
            "timezone": "auto",
        },
        timeout=10,
    )
    weather.raise_for_status()

    air = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "pm10,pm2_5,nitrogen_dioxide,ozone",
            "timezone": "auto",
        },
        timeout=10,
    )
    air.raise_for_status()

    return {
        "weather": weather.json().get("current", {}),
        "air_quality": air.json().get("current", {}),
    }


def compact_environment(env):
    if not env:
        return "No environmental context available."
    w = env.get("weather", {})
    a = env.get("air_quality", {})
    return (
        f"Temperature {w.get('temperature_2m', 'unknown')} C; "
        f"feels like {w.get('apparent_temperature', 'unknown')} C; "
        f"humidity {w.get('relative_humidity_2m', 'unknown')}%; "
        f"wind {w.get('wind_speed_10m', 'unknown')} km/h; "
        f"UV {w.get('uv_index', 'unknown')}; "
        f"PM2.5 {a.get('pm2_5', 'unknown')} ug/m3; "
        f"PM10 {a.get('pm10', 'unknown')} ug/m3."
    )


def should_use_environment(text):
    terms = [
        "weather", "heat", "hot", "cold", "humidity", "humid", "dust",
        "pollution", "air quality", "air pollution", "sun", "uv", "outdoor",
        "outside", "environment", "pollen", "smoke", "travel", "travelling",
        "traveling",
    ]
    lowered = text.lower()
    return any(term in lowered for term in terms)


# ============================================================
# SAFETY
# ============================================================
EMERGENCY_PHRASES = [
    "difficulty breathing", "trouble breathing", "shortness of breath",
    "can't breathe", "cannot breathe", "throat swelling", "swelling of the throat",
    "tongue swelling", "face swelling", "fainting", "passed out", "blue lips",
    "severe chest pain", "uncontrolled bleeding", "heavy bleeding", "severe confusion",
]


def has_emergency_signal(text):
    lowered = (text or "").lower()
    return [phrase for phrase in EMERGENCY_PHRASES if phrase in lowered]


def emergency_reply(language="english"):
    if language in ("hindi", "hinglish"):
        return (
            "Yeh aise symptoms hain jinhe chat ke through handle nahi karna chahiye. Abhi urgent medical help lo "
            "aur kisi trusted adult ya paas ke person ko batao ki kya ho raha hai. Possible emergency mein AI par depend mat karo."
        )
    return (
        "That sounds like something that should be dealt with urgently rather than through a chat. "
        "Please get urgent medical help now and let a trusted adult or someone nearby know what is happening. "
        "Don't rely on an AI response to manage a possible emergency."
    )


# ============================================================
# PROMPTS
# ============================================================
CHAT_SYSTEM = r"""
You are Drocto AI, a friendly conversational first-aid and everyday-health assistant.

Your job is to talk WITH the user, not write a medical pamphlet AT them.

VOICE / PERSONALITY
- Sound calm, warm, observant, and genuinely helpful.
- Respond to the user's latest message first.
- Use natural paragraphs and conversational transitions by default.
- Do not turn every answer into headings, numbered sections, or bullet lists.
- Use a short list only when it clearly helps, such as several warning signs.
- Avoid robotic phrases such as "Based on the information provided", "It is important to note", and "As an AI".
- Do not restate the user's entire question.
- You may say things like "Yeah, that can happen", "Hmm, there are a couple of possibilities", or "The main thing I'd focus on is..." when natural.
- Match the user's tone without becoming silly or unprofessional.
- Keep normal replies concise enough for a live school demonstration.
- Ask at most ONE useful follow-up question when it would meaningfully change the guidance.
- Don't ask a question just to keep the conversation going.

LANGUAGE / STYLE MATCHING
- Reply in the same language and writing style as the user's latest message unless they explicitly ask for another language.
- English input → natural English.
- Hindi written in Devanagari → natural Hindi in Devanagari.
- Hinglish / Roman Hindi → natural Hinglish using Roman script, keeping the same casual Hindi + English mix. Do not silently translate it into formal Hindi or pure English.
- For spoken input, preserve the language detected from the speech. Hindi speech should get Hindi; Hinglish speech should get Hinglish.
- For mixed-language input, follow the dominant style of the user's message and keep common English medical terms when that feels natural.

OFF-TOPIC HANDLING
If the user's message is clearly unrelated to health, first aid, safety, or wellbeing —
such as coding, math, history, politics, entertainment, sports, homework, gossip,
trivia, or finance — do NOT answer it. Reply instead with ONE short, warm, playful,
humorous sentence that gently declines and steers back to health topics. Never be
rude, sarcastic, or condescending. Two sentences maximum.

Examples of good off-topic replies (do not copy verbatim):
- "My stethoscope only picks up health questions — but if that code ever gives you a paper cut, I'm your AI."
- "Wrong clinic, friend! Great with bandages, terrible with algebra. Got a scrape or burn I can help with?"
- "That's well outside my first-aid kit, but it sounds like it could use a bandage too. Anything medical I can help with?"

If the message is borderline (touches on wellbeing, lifestyle, sleep, stress, diet,
exercise, posture, eye strain, screen fatigue, etc.), treat it as health-related.

HEALTH SAFETY
- This is educational information and first-aid guidance, not diagnosis or emergency care.
- Never say the user definitely has a disease, allergy, infection, or other condition.
- A photo cannot establish a diagnosis.
- Don't prescribe prescription medicines or give medication doses.
- Suggest only conservative, low-risk steps.
- If the situation sounds urgent, make the need for professional help clear.
- Don't invent symptoms, test results, causes, or details.
- Environmental/weather information is context, not proof of cause.

IMAGE CONTEXT
When image observations are provided by the vision model, treat them as uncertain
visual observations. Never mention internal model routing, Groq, prompts, API keys,
or software internals to the user unless the user explicitly asks about the implementation.

RESPONSE STYLE EXAMPLE
Instead of:
"### What you can do now\n- Wash the area...\n- Apply pressure..."
write something like:
"Yeah, for a small cut the first thing I'd focus on is getting the bleeding under control.
Hold a clean cloth or gauze against it with steady pressure for a few minutes. Once it stops,
gently rinse the area with clean running water and cover it. If the bleeding won't stop,
the cut is deep or widely open, or movement/feeling is affected, get medical help.
How deep does it look?"

Do not copy that example word-for-word. It shows the conversational style only.
"""

VISION_SYSTEM = r"""
You are Drocto AI's image-observation component.

Look at the image only for visible, relevant observations. Do not diagnose.
Do not prescribe medication. A photo alone cannot determine the medical cause.
If the image is blurry or insufficient, say so. If anything looks potentially
serious, recommend professional assessment rather than trying to identify a diagnosis.

Return concise internal notes:
VISIBLE:
- ...

POSSIBLE GENERAL CATEGORIES (NOT A DIAGNOSIS):
- ...

LIMITATIONS:
- ...

USEFUL FOLLOW-UP:
- ...
"""


def build_chat_messages(current_text, env_text="", image_notes="", response_language="auto"):
    messages = [{"role": "system", "content": CHAT_SYSTEM}]

    for item in st.session_state.messages[-MAX_HISTORY_MESSAGES:]:
        role = item.get("role")
        if role not in ("user", "assistant"):
            continue
        content = item.get("content", "")
        if content:
            messages.append({"role": role, "content": content})

    context = current_text or "The user sent an attachment without a typed question."

    if image_notes:
        context = (
            "The user is asking about a previously attached image. Here are the "
            "vision component's uncertain observations. Use them as context only; "
            "do not diagnose from them.\n\n"
            f"IMAGE OBSERVATIONS:\n{image_notes}\n\n"
            f"USER MESSAGE:\n{current_text or 'Please help me understand this.'}"
        )

    language_instruction = {
        "english": "Reply in natural English.",
        "hinglish": "Reply in natural Hinglish using Roman script, mixing Hindi and English naturally. Do not translate into formal Hindi or pure English.",
        "hindi": "Reply in natural Hindi using Devanagari script.",
        "assamese": "Reply in natural Assamese using Assamese script.",
        "bengali": "Reply in natural Bengali using Bengali script.",
        "bodo": "Reply in natural Bodo. Preserve the user's script when possible.",
        "dogri": "Reply in natural Dogri. Preserve the user's script when possible.",
        "gujarati": "Reply in natural Gujarati using Gujarati script.",
        "kannada": "Reply in natural Kannada using Kannada script.",
        "kashmiri": "Reply in natural Kashmiri. Preserve the user's script when possible.",
        "konkani": "Reply in natural Konkani. Preserve the user's script when possible.",
        "maithili": "Reply in natural Maithili. Preserve the user's script when possible.",
        "malayalam": "Reply in natural Malayalam using Malayalam script.",
        "manipuri": "Reply in natural Manipuri. Preserve the user's script when possible.",
        "marathi": "Reply in natural Marathi using Devanagari script.",
        "nepali": "Reply in natural Nepali using Devanagari script.",
        "odia": "Reply in natural Odia using Odia script.",
        "punjabi": "Reply in natural Punjabi using Gurmukhi script when the user uses Gurmukhi.",
        "sanskrit": "Reply in natural Sanskrit using Devanagari script when appropriate.",
        "santali": "Reply in natural Santali. Preserve the user's script when possible.",
        "sindhi": "Reply in natural Sindhi. Preserve the user's script when possible.",
        "tamil": "Reply in natural Tamil using Tamil script.",
        "telugu": "Reply in natural Telugu using Telugu script.",
        "urdu": "Reply in natural Urdu using Urdu script.",
        "auto": "Infer the user's language and writing style from the latest message and match it exactly.",
    }.get(response_language, "Infer the user's language and writing style from the latest message and match it exactly.")

    context += f"\n\nLANGUAGE MODE: {response_language.upper()}\n{language_instruction}"

    if env_text:
        context += f"\n\nOptional environmental context: {env_text}"

    messages.append({"role": "user", "content": context})
    return messages


# ============================================================
# LANGUAGE DETECTION
# ============================================================
ROMAN_HINDI_WORDS = {
    "aap", "ap", "aur", "bahut", "bhi", "bol", "chahiye", "gaya", "gayi",
    "hai", "hain", "ho", "hoga", "hoon", "hua", "hui", "ka", "kab", "kaha",
    "kaise", "kar", "karo", "karna", "kya", "kyu", "kyun", "lag", "mera",
    "meri", "mere", "mujhe", "mujh", "nahi", "nahin", "par", "pe", "raha",
    "rahi", "sakta", "sakti", "se", "tha", "the", "thi", "toh", "tum", "tumhe",
    "tujhe", "wala", "wali", "waise", "ya", "yeh", "ye", "zaroor", "sirf",
    "dard", "khujli", "bukhar", "sardi", "garmi", "pet", "gala", "sar", "haath",
    "pair", "chot", "jal", "jala", "jalan", "sujan", "daane", "dane", "khansi",
}

MARATHI_WORDS = {
    "mala", "majha", "majhi", "majhe", "kay", "kaay", "ahe", "aahe", "nahi",
    "tumhi", "aapan", "mhanje", "kasa", "kashi", "kuthe", "ka", "mi", "mala",
    "dukhat", "dukh", "tap", "ang", "ghasa", "khokla", "suj",
}

NEPALI_WORDS = {
    "malai", "mero", "merai", "tapai", "tapaai", "cha", "chha", "ho", "haina",
    "kaha", "kasari", "kina", "dukhai", "dukha", "joro", "khoki", "ghati",
}

DEVANAGARI_LANG_HINTS = {
    "marathi": {"मला", "माझा", "माझी", "माझे", "आहे", "नाही", "काय", "तुम्ही", "म्हणजे", "दुखत", "ताप"},
    "nepali": {"मलाई", "मेरो", "तपाईं", "छ", "छैन", "कसरी", "किन", "दुखाइ", "ज्वरो"},
    "hindi": {"मुझे", "मेरा", "मेरी", "है", "नहीं", "क्या", "आप", "कैसे", "दर्द", "बुखार", "खांसी"},
}


def infer_language(text, whisper_language=""):
    text = (text or "").strip()
    whisper_key = str(whisper_language or "").strip().lower()
    whisper_detected = WHISPER_LANGUAGE_MAP.get(whisper_key)
    if whisper_detected:
        roman_tokens = re.findall(r"[a-zA-Z]+", text.lower())
        roman_hindi_hits = sum(1 for token in roman_tokens if token in ROMAN_HINDI_WORDS)
        if whisper_detected == "hindi" and roman_hindi_hits >= 1 and not re.search(r"[\u0900-\u097F]", text):
            return "hinglish", "hi-IN"
        return whisper_detected, LANGUAGE_OPTIONS[whisper_detected]["speech"]

    if re.search(r"[\u0980-\u09FF]", text):
        if re.search(r"[ৰৱ]", text):
            return "assamese", "as-IN"
        return "bengali", "bn-IN"
    if re.search(r"[\u0A80-\u0AFF]", text):
        return "gujarati", "gu-IN"
    if re.search(r"[\u0A00-\u0A7F]", text):
        return "punjabi", "pa-IN"
    if re.search(r"[\u0B00-\u0B7F]", text):
        return "odia", "or-IN"
    if re.search(r"[\u0B80-\u0BFF]", text):
        return "tamil", "ta-IN"
    if re.search(r"[\u0C00-\u0C7F]", text):
        return "telugu", "te-IN"
    if re.search(r"[\u0C80-\u0CFF]", text):
        return "kannada", "kn-IN"
    if re.search(r"[\u0D00-\u0D7F]", text):
        return "malayalam", "ml-IN"
    if re.search(r"[\uABC0-\uABFF]", text):
        return "manipuri", "mni-IN"
    if re.search(r"[\u1C50-\u1C7F]", text):
        return "santali", "sat-IN"

    if re.search(r"[\u0900-\u097F]", text):
        if any(word in text for word in DEVANAGARI_LANG_HINTS["marathi"]):
            return "marathi", "mr-IN"
        if any(word in text for word in DEVANAGARI_LANG_HINTS["nepali"]):
            return "nepali", "ne-IN"
        return "hindi", "hi-IN"

    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    roman_hindi_hits = sum(1 for token in tokens if token in ROMAN_HINDI_WORDS)
    roman_marathi_hits = sum(1 for token in tokens if token in MARATHI_WORDS)
    roman_nepali_hits = sum(1 for token in tokens if token in NEPALI_WORDS)

    if roman_marathi_hits >= 2 and roman_marathi_hits >= roman_hindi_hits:
        return "marathi", "mr-IN"
    if roman_nepali_hits >= 2 and roman_nepali_hits >= roman_hindi_hits:
        return "nepali", "ne-IN"
    if roman_hindi_hits >= 1:
        return "hinglish", "hi-IN"

    if re.search(r"[\u0600-\u06FF]", text):
        return "urdu", "ur-IN"

    return "english", "en-IN"


def language_label(language):
    return LANGUAGE_OPTIONS.get(language, LANGUAGE_OPTIONS["english"]).get("label", "English")


def selected_or_detected_language(text, whisper_language=""):
    selected = st.session_state.get("response_language", "auto")
    if selected != "auto" and selected in LANGUAGE_OPTIONS:
        return selected, LANGUAGE_OPTIONS[selected]["speech"]
    return infer_language(text, whisper_language)


# ============================================================
# GROQ CHAT
# ============================================================
def groq_chat(current_text, env_text="", image_notes="", response_language="auto"):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing from your .env file.")

    messages = build_chat_messages(current_text, env_text=env_text, image_notes=image_notes, response_language=response_language)

    models = [CHAT_MODEL]
    if CHAT_FALLBACK_MODEL and CHAT_FALLBACK_MODEL != CHAT_MODEL:
        models.append(CHAT_FALLBACK_MODEL)

    last_error = ""

    for model in models:
        try:
            response = requests.post(
                GROQ_CHAT_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.65,
                    "max_tokens": 700,
                },
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code in (429, 500, 502, 503, 504):
                last_error = api_error_text(response)
                continue

            if not response.ok:
                raise RuntimeError(api_error_text(response))

            answer = response.json()["choices"][0]["message"]["content"].strip()
            if answer:
                return answer, model

        except requests.RequestException as exc:
            last_error = str(exc)

    raise RuntimeError(f"The chat service is unavailable right now. {last_error}")


# ============================================================
# GROQ SPEECH TO TEXT
# ============================================================
def groq_transcribe(audio_bytes, filename="voice.webm", mime_type="audio/webm", language_code=None):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing from your .env file.")

    response = requests.post(
        GROQ_STT_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        files={"file": (filename, audio_bytes, mime_type)},
        data={
            "model": STT_MODEL,
            "response_format": "verbose_json",
            "temperature": "0",
            **({"language": language_code} if language_code else {}),
        },
        timeout=60,
    )

    if response.status_code in (429, 500, 502, 503, 504):
        raise RuntimeError("Voice transcription is temporarily unavailable.")
    if not response.ok:
        raise RuntimeError(api_error_text(response))

    data = response.json()
    text = data.get("text", "").strip()
    if not text:
        raise RuntimeError("I couldn't make out the voice message. Try speaking a little more clearly.")

    return {"text": text, "language": data.get("language", "")}


# ============================================================
# GROQ IMAGE ANALYSIS
# ============================================================
def groq_image_analysis(image_bytes, mime_type, user_question):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing from your .env file.")

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    prompt = VISION_SYSTEM + "\n\nUSER'S QUESTION:\n" + (user_question or "Please look at this image.")

    payload = {
        "model": VISION_MODEL,
        "messages": [
            {"role": "system", "content": VISION_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                    },
                ],
            },
        ],
        "temperature": 0.2,
        "max_tokens": 500,
    }

    last_error = ""
    for attempt in range(3):
        try:
            response = requests.post(
                GROQ_CHAT_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code in (429, 500, 502, 503, 504):
                last_error = api_error_text(response)
                if attempt < 2:
                    time.sleep(1.0 * (attempt + 1))
                    continue

            if not response.ok:
                raise RuntimeError(api_error_text(response))

            answer = response.json()["choices"][0]["message"]["content"].strip()
            if answer:
                return answer
            raise RuntimeError("Groq returned no image observations.")

        except requests.RequestException as exc:
            last_error = str(exc)
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))

    raise RuntimeError(last_error or "Groq image analysis is temporarily unavailable.")


# ============================================================
# DEMO MODE
# ============================================================
def demo_response(text, has_image=False, has_audio=False, language="english"):
    lowered = (text or "").lower()

    if language in ("hindi", "hinglish"):
        if has_image:
            return "Photo aa gayi hai. Demo Mode mein main ek prepared example use kar raha hoon; live version image ke visible features dekhkar practical next steps batayega.\n\n*Demo Mode: yeh live medical assessment nahi hai.*"
        if any(x in lowered for x in ["cut", "wound", "bleeding", "scrape", "कट", "घाव"]):
            return "Agar chhota cut hai, pehle clean cloth ya gauze se steady pressure dekar bleeding control karo. Bleeding ruk na rahi ho, cut deep ya bahut khula ho, ya movement/feeling normal na ho, toh medical help lo.\n\n*Demo Mode: yeh live medical assessment nahi hai.*"
        if any(x in lowered for x in ["burn", "scald", "जल", "जलन"]):
            return "Chhote recent burn mein cool running water helpful hota hai. Direct ice mat lagao aur blisters mat phodo. Burn bada ya deep-looking ho, ya face, eyes, hands jaise sensitive area par ho, toh medical professional se check karao.\n\n*Demo Mode: yeh live medical assessment nahi hai.*"
        if any(x in lowered for x in ["rash", "itch", "itchy", "allergy", "bumps", "खुजली", "रैश"]):
            return "Red ya itchy skin ke kaafi possible reasons ho sakte hain, isliye sirf symptom dekhkar cause confirm nahi hota. Abhi koi naya irritating product avoid karo, area ko scratch mat karo, aur cool clean compress try kar sakte ho.\n\n*Demo Mode: yeh live medical assessment nahi hai.*"
        return "Haan, batao kya hua aur abhi kya symptoms notice ho rahe hain. Main possible reasons, safe next steps aur kab medical help leni chahiye, woh samjha dunga.\n\n*Demo Mode: yeh live medical assessment nahi hai.*"

    if has_image:
        return (
            "I can see the photo has come through. In Demo Mode I'm using a prepared example rather than actually "
            "judging the picture, but in the live version the image-analysis step would look at visible features and "
            "then we'd continue the conversation from there.\n\n"
            "*Demo Mode: this is a prepared educational response, not a live medical assessment.*"
        )

    if any(x in lowered for x in ["cut", "wound", "bleeding", "scrape"]):
        return (
            "Yeah, for a small cut the main thing is to get the bleeding under control first. Hold a clean cloth or "
            "gauze against it with steady pressure for a few minutes. Once it stops, gently rinse the area with clean "
            "running water and cover it. If the bleeding won't stop, the cut is deep or widely open, or you can't move "
            "or feel the area normally, get medical help.\n\n"
            "*Demo Mode: this is a prepared educational response, not a live medical assessment.*"
        )

    if any(x in lowered for x in ["burn", "scald"]):
        return (
            "Ouch — for a small recent burn, cool running water is the useful first step. Avoid putting ice directly on "
            "the skin or popping blisters. If the burn is large, deep-looking, or involves the face, eyes, hands, or "
            "another sensitive area, get it assessed by a medical professional.\n\n"
            "*Demo Mode: this is a prepared educational response, not a live medical assessment.*"
        )

    if any(x in lowered for x in ["rash", "itch", "itchy", "allergy", "bumps"]):
        return (
            "Hmm, itchy or red skin can come from a few different things, including irritation or a reaction to "
            "something that touched the skin, so the symptom by itself doesn't tell us the cause. For now, avoid any "
            "new product that might be irritating it, don't scratch the area, and a cool clean compress may feel more "
            "comfortable.\n\n"
            "*Demo Mode: this is a prepared educational response, not a live medical assessment.*"
        )

    return (
        "Sure — tell me what happened and what you're noticing right now. I'll help you work through the likely "
        "possibilities, the simple things you can safely do next, and anything that would make it worth getting medical "
        "help.\n\n*Demo Mode: this is a prepared educational response, not a live medical assessment.*"
    )


# ============================================================
# READ ALOUD
# ============================================================
def render_read_aloud(text, key, language="english", auto=False):
    """Render the existing read-aloud control, with a reliable Hindi/multilingual voice path."""
    clean = re.sub(r"\*+", "", text)
    clean = re.sub(r"#+\s*", "", clean)
    clean = re.sub(r"\n\s*[-•]\s*", ". ", clean)
    clean = re.sub(r"\n+", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()

    payload = json.dumps(clean, ensure_ascii=False).replace("</script>", "<" + "\\/" + "script>")
    lang = LANGUAGE_OPTIONS.get(language, LANGUAGE_OPTIONS["english"])["speech"]
    auto_flag = "true" if auto else "false"

    # Google Translate TTS is used only as a browser-side fallback when the
    # user's device does not expose a local voice for the requested language.
    tts_code = {
        "english": "en",
        "hinglish": "hi",
        "hindi": "hi",
        "assamese": "as",
        "bengali": "bn",
        "gujarati": "gu",
        "kannada": "kn",
        "malayalam": "ml",
        "marathi": "mr",
        "nepali": "ne",
        "odia": "or",
        "punjabi": "pa",
        "tamil": "ta",
        "telugu": "te",
        "urdu": "ur",
    }.get(language, lang.split("-")[0])

    components.html(
        f"""
        <div style="display:flex;justify-content:flex-start;align-items:center;gap:7px;margin:2px 0 4px 0;">
          <button id="speak-{key}"
            style="border:1px solid rgba(165,215,235,.22);background:linear-gradient(145deg,rgba(45,78,107,.7),rgba(14,30,53,.8));border-radius:999px;padding:6px 11px;font-size:11px;cursor:pointer;color:#eaf6ff;box-shadow:inset 0 1px 0 rgba(255,255,255,.08);">
            🔊 Read aloud
          </button>
          <button id="stop-{key}"
            style="border:1px solid rgba(165,215,235,.22);background:linear-gradient(145deg,rgba(45,78,107,.7),rgba(14,30,53,.8));border-radius:999px;padding:6px 11px;font-size:11px;cursor:pointer;color:#eaf6ff;box-shadow:inset 0 1px 0 rgba(255,255,255,.08);">
            ⏹ Stop speaking
          </button>
        </div>
        <script>
          const text = {payload};
          const speakBtn = document.getElementById("speak-{key}");
          const stopBtn = document.getElementById("stop-{key}");
          const language = {json.dumps(lang)};
          const ttsCode = {json.dumps(tts_code)};
          const autoSpeak = {auto_flag};
          let activeUtterance = null;
          let autoTimer = null;
          let remoteAudio = null;
          let remoteQueue = [];
          let remoteIndex = 0;
          let stoppedByUser = false;

          function resetSpeakButton() {{
            speakBtn.textContent = '🔊 Read aloud';
          }}

          function splitForTTS(value) {{
            const sentences = value.match(/[^.!?।]+[.!?।]?/g) || [value];
            const chunks = [];
            let current = '';
            for (const sentence of sentences) {{
              const s = sentence.trim();
              if (!s) continue;
              if ((current + ' ' + s).trim().length <= 180) {{
                current = (current + ' ' + s).trim();
              }} else {{
                if (current) chunks.push(current);
                if (s.length <= 180) current = s;
                else {{
                  const words = s.split(/\\s+/);
                  current = '';
                  for (const word of words) {{
                    if ((current + ' ' + word).trim().length <= 180) current = (current + ' ' + word).trim();
                    else {{
                      if (current) chunks.push(current);
                      current = word;
                    }}
                  }}
                }}
              }}
            }}
            if (current) chunks.push(current);
            return chunks.length ? chunks : [value.slice(0, 180)];
          }}

          function stopRemoteAudio() {{
            remoteQueue = [];
            remoteIndex = 0;
            if (remoteAudio) {{
              try {{
                remoteAudio.pause();
                remoteAudio.currentTime = 0;
                remoteAudio.removeAttribute('src');
                remoteAudio.load();
              }} catch (e) {{}}
              remoteAudio = null;
            }}
          }}

          function stopSpeaking(broadcast=true) {{
            stoppedByUser = true;
            if (autoTimer) {{
              clearTimeout(autoTimer);
              autoTimer = null;
            }}
            if ('speechSynthesis' in window) {{
              try {{
                window.speechSynthesis.cancel();
                setTimeout(() => {{
                  try {{ window.speechSynthesis.cancel(); }} catch (e) {{}}
                }}, 0);
              }} catch (e) {{}}
            }}
            activeUtterance = null;
            stopRemoteAudio();
            resetSpeakButton();

            if (broadcast && window.parent && window.parent !== window) {{
              try {{ window.parent.postMessage({{type: 'drocto-stop-speech'}}, '*'); }} catch (e) {{}}
            }}
          }}

          function chooseLocalVoice() {{
            if (!('speechSynthesis' in window)) return null;
            const voices = window.speechSynthesis.getVoices();
            const wanted = language.toLowerCase();
            const base = wanted.split('-')[0];
            const languageNames = {{
              'hi': ['hindi'], 'bn': ['bengali'], 'mr': ['marathi'], 'ta': ['tamil'],
              'te': ['telugu'], 'ml': ['malayalam'], 'kn': ['kannada'], 'gu': ['gujarati'],
              'pa': ['punjabi'], 'or': ['odia', 'oriya'], 'as': ['assamese'], 'ur': ['urdu'],
              'ne': ['nepali'], 'en': ['english']
            }};
            const names = languageNames[base] || [];
            return voices.find(v => v.lang && v.lang.toLowerCase() === wanted)
              || voices.find(v => v.lang && v.lang.toLowerCase().startsWith(base + '-'))
              || voices.find(v => v.lang && v.lang.toLowerCase() === base)
              || voices.find(v => names.some(n => (v.name || '').toLowerCase().includes(n)));
          }}

          function speakWithBrowserVoice() {{
            if (!('speechSynthesis' in window)) return false;
            const voice = chooseLocalVoice();
            if (!voice) return false;

            const utterance = new SpeechSynthesisUtterance(text);
            activeUtterance = utterance;
            utterance.voice = voice;
            utterance.lang = voice.lang || language;
            utterance.rate = ['hi-IN','bn-IN','mr-IN','ta-IN','te-IN','ml-IN','kn-IN','gu-IN','pa-IN','or-IN','as-IN','ur-IN','ne-IN'].includes(language) ? 0.94 : 0.98;
            utterance.pitch = 1.0;
            utterance.onstart = () => {{
              speakBtn.textContent = '🔊 Speaking…';
            }};
            utterance.onend = () => {{
              if (activeUtterance === utterance) activeUtterance = null;
              resetSpeakButton();
            }};
            utterance.onerror = () => {{
              if (activeUtterance === utterance) activeUtterance = null;
              resetSpeakButton();
            }};

            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(utterance);
            speakBtn.textContent = '🔊 Speaking…';
            return true;
          }}

          function playRemoteChunk() {{
            if (stoppedByUser || remoteIndex >= remoteQueue.length) {{
              resetSpeakButton();
              remoteAudio = null;
              return;
            }}

            const chunk = remoteQueue[remoteIndex++];
            const url = 'https://translate.google.com/translate_tts?ie=UTF-8&client=tw-ob&tl='
              + encodeURIComponent(ttsCode) + '&q=' + encodeURIComponent(chunk);
            remoteAudio = new Audio(url);
            remoteAudio.preload = 'auto';
            remoteAudio.onplay = () => speakBtn.textContent = '🔊 Speaking…';
            remoteAudio.onended = () => playRemoteChunk();
            remoteAudio.onerror = () => {{
              // If remote TTS is unavailable, try the browser voice once.
              remoteAudio = null;
              if (!stoppedByUser && speakWithBrowserVoice()) return;
              resetSpeakButton();
            }};
            remoteAudio.play().catch(() => {{
              remoteAudio = null;
              if (!stoppedByUser && speakWithBrowserVoice()) return;
              resetSpeakButton();
            }});
          }}

          function speakRemote() {{
            remoteQueue = splitForTTS(text);
            remoteIndex = 0;
            speakBtn.textContent = '🔊 Speaking…';
            playRemoteChunk();
          }}

          function speakNow() {{
            stoppedByUser = false;
            stopSpeaking(false);
            stoppedByUser = false;

            if (language === 'hi-IN') {{
              // Hindi: wait briefly for Chrome/Edge voices to populate. If the
              // machine has no Hindi voice, use the same button with a Hindi TTS fallback.
              const tryHindi = () => {{
                if (stoppedByUser) return;
                if (!speakWithBrowserVoice()) speakRemote();
              }};
              if ('speechSynthesis' in window) {{
                window.speechSynthesis.getVoices();
                setTimeout(tryHindi, 120);
                if ('onvoiceschanged' in window.speechSynthesis) {{
                  window.speechSynthesis.addEventListener('voiceschanged', tryHindi, {{once:true}});
                }}
              }} else {{
                speakRemote();
              }}
              return;
            }}

            if ('speechSynthesis' in window) {{
              const tryLocal = () => {{
                if (stoppedByUser) return;
                if (!speakWithBrowserVoice()) {{
                  // Fall back for supported regional languages when no local voice exists.
                  speakRemote();
                }}
              }};
              window.speechSynthesis.getVoices();
              setTimeout(tryLocal, 80);
              if ('onvoiceschanged' in window.speechSynthesis) {{
                window.speechSynthesis.addEventListener('voiceschanged', tryLocal, {{once:true}});
              }}
            }} else {{
              speakRemote();
            }}
          }}

          speakBtn.onclick = speakNow;
          stopBtn.onclick = () => stopSpeaking(true);

          window.addEventListener('message', (event) => {{
            if (event && event.data && event.data.type === 'drocto-stop-speech') {{
              stopSpeaking(false);
            }}
          }});

          if (autoSpeak) {{
            autoTimer = setTimeout(() => {{
              autoTimer = null;
              speakNow();
            }}, 220);
          }}
        </script>
        """,
        height=41,
        scrolling=False,
    )


# ============================================================
# SESSION STATE
# ============================================================
DEFAULTS = {
    "messages": [],
    "demo_mode": False,
    "location": "",
    "last_image_notes": "",
    "voice_fallback_processed": "",
    "last_response_language": "english",
    "response_language": "auto",
    "auto_speak_voice": True,
    "theme": "dark",
}

for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def reset_chat():
    st.session_state.messages = []
    st.session_state.last_image_notes = ""


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div class="fa-side-card">
            <div class="fa-side-title">💊 Drocto AI</div>
            <div class="fa-side-copy">Friendly first-aid guidance for everyday health questions.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### 🎨 Appearance")
    dark_mode = st.toggle(
        "🌙 Dark mode",
        value=(st.session_state.theme == "dark"),
        key="dark_mode_toggle",
    )
    st.session_state.theme = "dark" if dark_mode else "light"
    st.caption(f"Current theme: **{'Dark' if dark_mode else 'Light'}**")

    st.divider()

    if st.button("🗑️  New conversation", use_container_width=True):
        reset_chat()
        st.rerun()

    st.session_state.demo_mode = st.toggle(
        "🎬  Demo Mode",
        value=st.session_state.demo_mode,
        help="Uses prepared answers so you can demonstrate the interface even when APIs are unavailable.",
    )

    language_keys = list(LANGUAGE_OPTIONS.keys())
    current_index = language_keys.index(st.session_state.response_language) if st.session_state.response_language in language_keys else 0
    selected_language = st.selectbox(
        "Response language",
        language_keys,
        index=current_index,
        format_func=lambda key: LANGUAGE_OPTIONS[key]["label"],
        help="Auto-detect follows the language and writing style you use. Choose a language manually for a guaranteed output language.",
    )
    st.session_state.response_language = selected_language

    st.session_state.auto_speak_voice = st.toggle(
        "🔊 Auto-speak voice replies",
        value=st.session_state.auto_speak_voice,
        help="When you send a voice message, Drocto AI automatically speaks its reply aloud.",
    )

    st.session_state.location = st.text_input(
        "Location (optional)",
        value=st.session_state.location,
        placeholder="e.g. Dhanbad",
        help="Used only to add simple weather/air-quality context when a question makes it relevant.",
    )

    st.markdown(
        """
        <div class="fa-side-note">
            Normal chat → fast chat model<br>
            Photos → Groq vision + chat<br>
            Voice → speech-to-text + spoken reply<br>
            Languages → Indian multilingual replies + Hinglish<br><br>
            General information and first-aid guidance — not a medical diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )

inject_css(st.session_state.theme)


# ============================================================
# HERO + CHIPS + FEATURES
# ============================================================
HERO_HTML = """
<div class="fa-hero">
  <div class="fa-hero-inner">
    <div class="fa-hero-badge">
      <span class="dot"></span> Online · Educational assistant
    </div>
    <h1 class="fa-hero-title">💊 Drocto AI</h1>
    <p class="fa-hero-sub">
      Describe it, show it, or say it — get calm, practical first-aid guidance. Fast chat for
      text, Groq vision for photos, and voice notes transcribed on the fly, in your language.
    </p>
    <div class="fa-hero-tags">
      <span class="fa-tag">🩹 Wound care</span>
      <span class="fa-tag">🔥 Burns</span>
      <span class="fa-tag">🐝 Bites & stings</span>
      <span class="fa-tag">🌡️ Fever</span>
      <span class="fa-tag">🫁 Choking</span>
      <span class="fa-tag">🗣️ Hindi · Hinglish · 22+ languages</span>
    </div>
  </div>
</div>
"""

INFO_HTML = """
<div class="fa-info">
  <span class="dot">●</span>
  <b>How it works:</b> describe what happened → get possible explanations and practical next steps.
  Drocto AI doesn't diagnose.
</div>
"""

FEATURES_HTML = """
<div class="fa-section-label">How it works</div>
<div class="fa-features">
  <div class="fa-feature">
    <div class="fa-feature-ico">💬</div>
    <p class="fa-feature-title">Ask a question</p>
    <p class="fa-feature-text">"What should I do for a minor cut?" — answered with calm, practical steps.</p>
  </div>
  <div class="fa-feature">
    <div class="fa-feature-ico">📷</div>
    <p class="fa-feature-title">Attach a photo</p>
    <p class="fa-feature-text">Show a rash, wound or swelling — Groq vision describes visible observations.</p>
  </div>
  <div class="fa-feature">
    <div class="fa-feature-ico">🎙️</div>
    <p class="fa-feature-title">Voice chat</p>
    <p class="fa-feature-text">Speak naturally and Drocto AI talks back in the same language.</p>
  </div>
</div>
"""

CHIPS = [
    ("🩹  Minor cut care", "How do I clean and dress a small cut at home?"),
    ("🔥  Small burn", "What should I do for a minor kitchen burn?"),
    ("🐝  Insect bite", "My arm is swollen after a bee sting. What should I do?"),
    ("🫁  Choking", "What are the basic steps to help someone who is choking?"),
]


if not st.session_state.messages:
    st.markdown(HERO_HTML, unsafe_allow_html=True)
    st.markdown(INFO_HTML, unsafe_allow_html=True)

    st.markdown('<div class="fa-section-label">Try asking</div>', unsafe_allow_html=True)
    chip_cols = st.columns(len(CHIPS))
    for i, (col, (label, question)) in enumerate(zip(chip_cols, CHIPS)):
        with col:
            if st.button(
                label,
                key=f"chip_{i}",
                use_container_width=True,
                help=question,
            ):
                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state._pending_chip = question
                st.rerun()

    st.markdown(FEATURES_HTML, unsafe_allow_html=True)
else:
    st.markdown(
        "<h2 style='margin: 4px 0 18px 0; font-size: 24px;'>💊 Drocto AI</h2>",
        unsafe_allow_html=True,
    )

if st.session_state.demo_mode:
    st.info("🎬 Demo Mode is on — prepared replies are used and no live AI request is required.")


# ============================================================
# CHAT HISTORY
# ============================================================
for idx, message in enumerate(st.session_state.messages):
    role = message["role"]
    avatar = "🧑" if role == "user" else "💊"
    with st.chat_message(role, avatar=avatar):
        if message.get("image_bytes"):
            st.image(message["image_bytes"], caption="Photo attached", width=220)
        if message.get("audio_bytes"):
            st.audio(message["audio_bytes"], format=message.get("audio_mime", "audio/webm"))
        if message.get("content"):
            st.markdown(message["content"])
            if role == "assistant":
                render_read_aloud(
                    message["content"],
                    f"history-{idx}",
                    language=message.get("language", "english"),
                )
        if message.get("demo"):
            st.caption("🎬 Prepared demonstration response")


# ============================================================
# CHAT INPUT COMPATIBILITY
# ============================================================
chat_signature = inspect.signature(st.chat_input)
chat_accepts_audio = "accept_audio" in chat_signature.parameters
submission = None
legacy_audio = None

if chat_accepts_audio:
    submission = st.chat_input(
        "Tell me what happened…",
        key="care_chat_input",
        accept_file=True,
        file_type=["jpg", "jpeg", "png", "webp"],
        accept_audio=True,
        audio_sample_rate=16000,
        submit_mode="disable",
    )
else:
    legacy_audio = st.audio_input("🎙️ Voice message", sample_rate=16000, key="legacy_audio")
    submission = st.chat_input(
        "Tell me what happened…",
        key="care_chat_input_legacy",
        accept_file=True,
        file_type=["jpg", "jpeg", "png", "webp"],
    )


def get_submission_parts(value):
    text = ""
    files = []
    audio = None

    if value is None:
        return text, files, audio
    if isinstance(value, str):
        return value, files, audio

    if hasattr(value, "get"):
        text = value.get("text", "") or ""
        files = value.get("files", []) or []
        audio = value.get("audio")
    else:
        text = getattr(value, "text", "") or ""
        files = getattr(value, "files", []) or []
        audio = getattr(value, "audio", None)

    return text, files, audio


def attachment_bytes(uploaded):
    return uploaded.getvalue(), uploaded.type or "application/octet-stream"


if legacy_audio is not None and submission is None:
    audio_hash = hashlib.sha256(legacy_audio.getvalue()).hexdigest()
    if audio_hash != st.session_state.voice_fallback_processed:
        st.session_state.voice_fallback_processed = audio_hash
        submission = {"text": "", "files": [], "audio": legacy_audio}


# ============================================================
# HANDLE CHIP-TRIGGERED PROMPT
# ============================================================
chip_prompt = st.session_state.pop("_pending_chip", None)

if chip_prompt and submission is None:
    submission = {"text": chip_prompt, "files": [], "audio": None}


# ============================================================
# PROCESS NEW TURN
# ============================================================
if submission is not None:
    text, files, audio = get_submission_parts(submission)
    image = files[0] if files else None

    image_bytes = None
    image_mime = ""
    audio_bytes = None
    audio_mime = ""
    whisper_language = ""

    if image is not None:
        try:
            image_bytes, image_mime = attachment_bytes(image)
        except Exception:
            image_bytes = None

    if audio is not None:
        try:
            audio_bytes, audio_mime = attachment_bytes(audio)
        except Exception:
            audio_bytes = None

    transcribed_text = text.strip()
    if audio_bytes:
        with st.status("🎙️ Transcribing your voice…", expanded=False) as status:
            try:
                selected_for_audio = st.session_state.get("response_language", "auto")
                audio_language_hint = LANGUAGE_OPTIONS.get(selected_for_audio, {}).get("whisper") if selected_for_audio != "auto" else None
                transcription = groq_transcribe(
                    audio_bytes,
                    filename="voice.webm",
                    mime_type=audio_mime or "audio/webm",
                    language_code=audio_language_hint,
                )
                transcribed_text = transcription["text"]
                whisper_language = transcription.get("language", "")
                status.update(label="🎙️ Voice understood", state="complete")
            except Exception as exc:
                status.update(label="Voice transcription failed", state="error")
                st.error(str(exc))

    response_language, _speech_lang = selected_or_detected_language(transcribed_text, whisper_language)
    st.session_state.last_response_language = response_language

    display_text = transcribed_text
    if not display_text:
        if image is not None:
            display_text = "Take a look at this photo and help me understand what I should do next."
        elif audio_bytes:
            display_text = "🎙️ Voice message"
        else:
            display_text = "Can you help me with this?"

    st.session_state.messages.append(
        {
            "role": "user",
            "content": transcribed_text,
            "image_bytes": image_bytes,
            "audio_bytes": audio_bytes,
            "audio_mime": audio_mime,
            "language": response_language,
        }
    )

    with st.chat_message("user", avatar="🧑"):
        if image_bytes:
            st.image(image_bytes, caption="Photo attached", width=220)
        if audio_bytes:
            st.audio(audio_bytes, format=audio_mime or "audio/webm")
        if display_text:
            st.markdown(display_text)

    emergency = has_emergency_signal(transcribed_text)

    env_text = ""
    location = st.session_state.location.strip()
    if location and should_use_environment(transcribed_text):
        try:
            loc = geocode(location)
            if loc:
                env = environment_context(loc["latitude"], loc["longitude"])
                env_text = compact_environment(env)
        except requests.RequestException:
            env_text = ""

    with st.chat_message("assistant", avatar="💊"):
        model_used = ""

        if emergency:
            answer = emergency_reply(response_language)
            model_used = "SAFETY"

        elif st.session_state.demo_mode:
            answer = demo_response(
                transcribed_text,
                has_image=bool(image_bytes),
                has_audio=bool(audio_bytes),
                language=response_language,
            )
            model_used = "DEMO"

        elif image_bytes:
            try:
                with st.spinner("🔎 Taking a closer look…"):
                    image_notes = groq_image_analysis(
                        image_bytes,
                        image_mime or "image/jpeg",
                        transcribed_text or display_text,
                    )
                st.session_state.last_image_notes = image_notes

                with st.spinner("💬 Putting that into plain English…"):
                    answer, model_used = groq_chat(
                        transcribed_text or "What should I know about this photo?",
                        env_text=env_text,
                        image_notes=image_notes,
                        response_language=response_language,
                    )
            except Exception as exc:
                answer = (
                    "I couldn't analyze that photo right now. You can still ask me about what you're noticing in "
                    "words, or try the photo again in a moment.\n\n"
                    f"*Technical detail: {exc}*"
                )
                model_used = "IMAGE ERROR"

        else:
            try:
                with st.spinner("💬 Thinking…"):
                    answer, model_used = groq_chat(
                        transcribed_text or display_text,
                        env_text=env_text,
                        image_notes=st.session_state.last_image_notes if st.session_state.last_image_notes else "",
                        response_language=response_language,
                    )
            except Exception as exc:
                answer = (
                    "I'm having trouble reaching the chat service right now. Try again in a moment, or switch on "
                    "Demo Mode if you need the presentation to keep running.\n\n"
                    f"*Technical detail: {exc}*"
                )
                model_used = "CHAT ERROR"

        st.markdown(answer)
        render_read_aloud(
            answer,
            f"new-{len(st.session_state.messages)}",
            language=response_language,
            auto=bool(audio_bytes) and st.session_state.auto_speak_voice,
        )
        if model_used == "DEMO":
            st.caption("🎬 Prepared demonstration response")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "language": response_language,
            "demo": model_used == "DEMO",
        }
    )


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    f"""
    <div class="fa-footer">
      <span>💊 Drocto AI • General first-aid information — not a diagnosis</span>
      <span>{datetime.now():%d %b %Y}</span>
    </div>
    """,
    unsafe_allow_html=True,
)