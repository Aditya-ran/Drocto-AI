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
    "tamil": {"label": "🇮🇳 Tamil (தமிழ்)", "whisper": "ta", "speech": "ta-IN"},
    "telugu": {"label": "🇮🇳 Telugu (తెలుగు)", "whisper": "te", "speech": "te-IN"},
    "urdu": {"label": "🇮🇳 Urdu (اردو)", "whisper": "ur", "speech": "ur-IN"},
    "arabic": {"label": "🌍 Arabic (العربية)", "whisper": "ar", "speech": "ar-SA"},
    "chinese": {"label": "🌍 Chinese (中文)", "whisper": "zh", "speech": "zh-CN"},
    "dutch": {"label": "🌍 Dutch (Nederlands)", "whisper": "nl", "speech": "nl-NL"},
    "french": {"label": "🌍 French (Français)", "whisper": "fr", "speech": "fr-FR"},
    "german": {"label": "🌍 German (Deutsch)", "whisper": "de", "speech": "de-DE"},
    "greek": {"label": "🌍 Greek (Ελληνικά)", "whisper": "el", "speech": "el-GR"},
    "hebrew": {"label": "🌍 Hebrew (עברית)", "whisper": "he", "speech": "he-IL"},
    "indonesian": {"label": "🌍 Indonesian (Bahasa Indonesia)", "whisper": "id", "speech": "id-ID"},
    "italian": {"label": "🌍 Italian (Italiano)", "whisper": "it", "speech": "it-IT"},
    "japanese": {"label": "🌍 Japanese (日本語)", "whisper": "ja", "speech": "ja-JP"},
    "korean": {"label": "🌍 Korean (한국어)", "whisper": "ko", "speech": "ko-KR"},
    "persian": {"label": "🌍 Persian (فارسی)", "whisper": "fa", "speech": "fa-IR"},
    "polish": {"label": "🌍 Polish (Polski)", "whisper": "pl", "speech": "pl-PL"},
    "portuguese": {"label": "🌍 Portuguese (Português)", "whisper": "pt", "speech": "pt-BR"},
    "russian": {"label": "🌍 Russian (Русский)", "whisper": "ru", "speech": "ru-RU"},
    "spanish": {"label": "🌍 Spanish (Español)", "whisper": "es", "speech": "es-ES"},
    "swahili": {"label": "🌍 Swahili (Kiswahili)", "whisper": "sw", "speech": "sw-KE"},
    "thai": {"label": "🌍 Thai (ไทย)", "whisper": "th", "speech": "th-TH"},
    "turkish": {"label": "🌍 Turkish (Türkçe)", "whisper": "tr", "speech": "tr-TR"},
    "ukrainian": {"label": "🌍 Ukrainian (Українська)", "whisper": "uk", "speech": "uk-UA"},
    "vietnamese": {"label": "🌍 Vietnamese (Tiếng Việt)", "whisper": "vi", "speech": "vi-VN"},
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
    "ar": "arabic", "arabic": "arabic",
    "zh": "chinese", "zh-cn": "chinese", "chinese": "chinese",
    "nl": "dutch", "dutch": "dutch",
    "fr": "french", "french": "french",
    "de": "german", "german": "german",
    "el": "greek", "greek": "greek",
    "he": "hebrew", "hebrew": "hebrew",
    "id": "indonesian", "indonesian": "indonesian",
    "it": "italian", "italian": "italian",
    "ja": "japanese", "japanese": "japanese",
    "ko": "korean", "korean": "korean",
    "fa": "persian", "persian": "persian",
    "pl": "polish", "polish": "polish",
    "pt": "portuguese", "portuguese": "portuguese",
    "ru": "russian", "russian": "russian",
    "es": "spanish", "spanish": "spanish",
    "sw": "swahili", "swahili": "swahili",
    "th": "thai", "thai": "thai",
    "tr": "turkish", "turkish": "turkish",
    "uk": "ukrainian", "ukrainian": "ukrainian",
    "vi": "vietnamese", "vietnamese": "vietnamese",
}

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

MAX_HISTORY_MESSAGES = 12
REQUEST_TIMEOUT = 45

# Every user-facing static string lives here so the complete interface can be
# translated in one cached Groq request without changing the layout or CSS.
UI_COPY = {
    "site_language": "Website language",
    "site_language_help": "Translate app labels and information. AI reply language is chosen separately.",
    "custom_language": "Enter a language name",
    "language_name_custom": "Other / custom language",
    "sidebar_subtitle": "Friendly first-aid guidance for everyday health questions.",
    "appearance": "Appearance",
    "dark_mode": "Dark mode",
    "light_mode": "Light mode",
    "current_theme": "Current theme",
    "new_conversation": "New conversation",
    "demo_mode": "Demo Mode",
    "demo_mode_help": "Uses prepared replies so you can demonstrate the interface even when the API is unavailable.",
    "response_language": "AI reply language",
    "response_language_help": "Choose the language used for Drocto AI's answers. Auto-detect follows the language of your message.",
    "auto_speak": "Auto-speak voice replies",
    "auto_speak_help": "When you send a voice message, Drocto AI automatically speaks its reply aloud.",
    "location": "Location (optional)",
    "location_placeholder": "e.g. Dhanbad",
    "location_help": "Used only to add simple weather and air-quality context when relevant.",
    "side_normal_chat": "Text chat → fast conversational model",
    "side_photos": "Photos → image observations + chat",
    "side_voice": "Voice → speech transcription + spoken reply",
    "side_languages": "Multilingual replies, including Hinglish",
    "medical_disclaimer": "General information and first-aid guidance — not a medical diagnosis.",
    "online_badge": "Online · Educational assistant",
    "hero_intro": "Describe it, show it, or say it — get calm, practical first-aid guidance. Fast chat for text, Groq vision for photos, and voice notes transcribed on the fly, in your language.",
    "tag_wounds": "Wound care",
    "tag_burns": "Burns",
    "tag_bites": "Bites & stings",
    "tag_fever": "Fever",
    "tag_choking": "Choking",
    "tag_languages": "Hindi · Hinglish · 22+ languages",
    "how_it_works": "How it works",
    "how_info": "Describe what happened → get possible explanations and practical next steps. Drocto AI doesn't diagnose.",
    "try_asking": "Try asking",
    "feature_ask_title": "Ask a question",
    "feature_ask_text": "\"What should I do for a minor cut?\" — answered with calm, practical steps.",
    "feature_photo_title": "Attach a photo",
    "feature_photo_text": "Show a rash, wound or swelling — Groq vision describes visible observations.",
    "feature_voice_title": "Voice chat",
    "feature_voice_text": "Speak naturally and Drocto AI talks back in the same language.",
    "chip_cut_label": "Minor cut care",
    "chip_cut_question": "How do I clean and dress a small cut at home?",
    "chip_burn_label": "Small burn",
    "chip_burn_question": "What should I do for a minor kitchen burn?",
    "chip_bite_label": "Insect bite",
    "chip_bite_question": "My arm is swollen after a bee sting. What should I do?",
    "chip_choking_label": "Choking",
    "chip_choking_question": "What are the basic steps to help someone who is choking?",
    "demo_banner": "Demo Mode is on — prepared replies are used and no live AI request is required.",
    "photo_attached": "Photo attached",
    "prepared_reply": "Prepared demonstration response",
    "chat_placeholder": "Tell me what happened…",
    "voice_message": "Voice message",
    "transcribing_voice": "Transcribing your voice…",
    "voice_understood": "Voice understood",
    "transcription_failed": "Voice transcription failed",
    "taking_closer_look": "Taking a closer look…",
    "putting_into_plain_language": "Preparing a clear response…",
    "thinking": "Thinking…",
    "image_error": "I couldn't analyze that photo right now. You can still ask me about what you're noticing in words, or try the photo again in a moment.",
    "chat_error": "I'm having trouble reaching the chat service right now. Try again in a moment, or switch on Demo Mode if you need the presentation to keep running.",
    "technical_detail": "Technical detail",
    "default_image_prompt": "Take a look at this photo and help me understand what I should do next.",
    "default_help_prompt": "Can you help me with this?",
    "footer_disclaimer": "General first-aid information — not a diagnosis",
    "read_aloud": "Read aloud",
    "speaking": "Speaking…",
    "voice_not_supported": "Voice playback not supported",
}

# Keep language names in the same translation bundle as the rest of the UI.
for _lang_key, _lang_data in LANGUAGE_OPTIONS.items():
    _label_parts = _lang_data["label"].split(" ", 1)
    UI_COPY[f"language_name_{_lang_key}"] = _label_parts[1] if len(_label_parts) > 1 else _label_parts[0]


@st.cache_data(ttl=300, show_spinner=False)
def translate_ui_copy(language, api_key):
    """Translate the UI as one JSON bundle; cache by language and key presence."""
    if language == "english":
        return {**UI_COPY, "_translation_ok": True}
    if not api_key:
        return {**UI_COPY, "_translation_ok": False}

    prompt = (
        "Translate the VALUES of this JSON object into the language named " + language + ". "
        "Return only a valid JSON object with exactly the same keys and string values translated. "
        "Keep product name Drocto AI, medical meaning, emojis, arrows, and placeholders unchanged. "
        "Use fluent, natural, locally appropriate wording and the normal script for the target language. "
        "Do not add advice, explanations, markdown fences, or extra keys. Input JSON:\n"
        + json.dumps(UI_COPY, ensure_ascii=False)
    )
    errors = []
    models = [CHAT_MODEL]
    if CHAT_FALLBACK_MODEL and CHAT_FALLBACK_MODEL not in models:
        models.append(CHAT_FALLBACK_MODEL)
    for model in models:
        try:
            response = requests.post(
                GROQ_CHAT_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a professional software-interface translator. Preserve JSON keys exactly."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0,
                    "max_tokens": 5000,
                },
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"].strip()
            # Defensive parsing in case a model adds a fence despite the instruction.
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
            translated = json.loads(raw)
            result = dict(UI_COPY)
            valid = True
            for key in UI_COPY:
                value = translated.get(key)
                if not isinstance(value, str) or not value.strip():
                    valid = False
                    break
                result[key] = value.strip()
            if not valid:
                raise RuntimeError("UI translator returned incomplete JSON.")
            result["_translation_ok"] = True
            return result
        except Exception as exc:
            errors.append(str(exc))
    # Cache a short-lived English fallback so a broken key/API does not stall every rerun.
    return {**UI_COPY, "_translation_ok": False}


def safe_ui(value):
    """Escape translated copy before inserting it into the existing HTML layout."""
    return html.escape(str(value), quote=True)

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
    "can't breathe", "cannot breathe", "stopped breathing", "not breathing",
    "isn't breathing", "is not breathing", "hasn't breathed", "has not breathed",
    "throat swelling", "swelling of the throat", "tongue swelling", "face swelling",
    "fainting", "passed out", "blue lips", "severe chest pain",
    "uncontrolled bleeding", "heavy bleeding", "severe confusion", "unresponsive",
    "not waking up", "won't wake up", "will not wake up",
]

NONLITERAL_EMERGENCY_PHRASES = (
    "dying of laughter", "can't breathe laughing", "cannot breathe laughing",
    "can't breathe from laughing", "i'm dead", "im dead", "i am dead",
    "dead lol", "dead 💀", "that killed me", "this killed me", "my brain is dead",
    "i died yesterday", "i died last night", "i died last week",
)

EXPLICIT_FICTION_OR_JOKE = re.compile(
    r"\b(jk|just kidding|i'm joking|im joking|i am joking|this is a joke|prank|"
    r"in my story|for my story|in this story|my character|fictional character|"
    r"for a screenplay|for a script|in a movie|in a game|roleplay|hypothetically)\b",
    re.IGNORECASE,
)


def clearly_nonliteral_emergency_context(text):
    """Avoid hard-coded emergency replies for strong figurative/fictional cues.

    Ambiguous but extreme claims are handed to the conversational model, whose
    instructions require a brief conditional safety caveat rather than dismissal.
    """
    lowered = (text or "").lower().strip()
    if not lowered:
        return False
    if re.search(r"\b(not joking|no joke|i'm serious|im serious|this is real|for real)\b", lowered):
        return False
    if any(phrase in lowered for phrase in NONLITERAL_EMERGENCY_PHRASES):
        return True
    if EXPLICIT_FICTION_OR_JOKE.search(lowered):
        return True
    # An alleged breathing pause of multiple days is likely exaggeration or a bit;
    # let the model acknowledge that while still advising what to do if current and real.
    breathing = r"(?:breathed|breathing|breathe|breath)"
    long_duration = r"\b(?:for\s+)?(?:[2-9]|[1-9]\d+|two|three|four|five|six|seven|eight|nine|ten|several|many)\s+(?:days?|weeks?|months?|years?)\b"
    if re.search(breathing, lowered) and re.search(long_duration, lowered):
        return True
    return False


def has_emergency_signal(text):
    lowered = (text or "").lower()
    if clearly_nonliteral_emergency_context(lowered):
        return []
    return [phrase for phrase in EMERGENCY_PHRASES if phrase in lowered]


def emergency_reply(language="english"):
    if isinstance(language, str) and language.startswith("custom:"):
        language = language.split(":", 1)[1].strip() or "English"
    if language in ("hindi", "hinglish"):
        return (
            "Yeh possible emergency ho sakti hai. Abhi local emergency services ko call karo aur paas ke kisi person ko turant batao. "
            "Agar koi saans nahi le raha hai, emergency help aane tak dispatcher ki instructions follow karo. AI reply ka wait mat karo."
        )
    base = (
        "This may be an emergency. Call your local emergency number now and alert someone nearby. "
        "If someone is not breathing, follow the emergency dispatcher's instructions while help is on the way. "
        "Don't wait for an AI reply."
    )
    if language == "english":
        return base
    # Use the configured multilingual model to render a short safety template in the chosen reply language.
    if not GROQ_API_KEY:
        return base
    try:
        response = requests.post(
            GROQ_CHAT_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": CHAT_MODEL,
                "messages": [
                    {"role": "system", "content": f"Translate the safety message into natural {language}. Preserve its urgent meaning. Output only the translated message."},
                    {"role": "user", "content": base},
                ],
                "temperature": 0,
                "max_tokens": 250,
            },
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        translated = response.json()["choices"][0]["message"]["content"].strip()
        return translated or base
    except Exception:
        return base


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

CONTEXT, HUMOUR, SARCASM, MEMES AND FIGURATIVE LANGUAGE
Do not interpret a message by matching one dramatic phrase in isolation. Infer intent from the full sentence, previous turns, tense, emojis, slang, idioms, exaggeration, plausibility, and whether the message describes a real person/event happening now.

Recognize these common nonliteral patterns:
- Exaggeration: "I'm starving", "this homework will be the death of me", "I ran a million miles", "my head is exploding", "I slept for 100 years".
- Internet slang/memes: "I'm dead 💀", "bro I'm cooked", "skill issue", "it's over for me", "my brain has left the chat", "I died yesterday 💀".
- Figurative distress: "this headache is killing me 😂", "I'm dying of laughter", "I can't breathe laughing", "that joke murdered me". Do not confuse these with a plain statement of current breathing difficulty; if unclear, ask one short clarification and keep appropriate caution.
- Sarcasm/irony: "great, just what I needed 🙃", "love that for me", "sure, because that's totally normal". Respond to the likely intent rather than the literal words.
- Absurd or impossible claims: aliens stole my toes, I am texting from Mars, my pet has become prime minister, someone stopped breathing for two days. Acknowledge the bit playfully without affirming false facts.
- Fiction and quotation: stories, scripts, roleplay, game characters, hypotheticals, quoted messages and examples are not automatically reports of the user's own situation.
- Casual banter, playful teasing, typos, keyboard smash, rhetorical questions and emojis can signal playfulness; use them as clues, not absolute proof.
- Use conversation history. If the user is clearly testing whether you understand humour, do not suddenly switch into condolence/emergency boilerplate for an obvious bit.

SPECIFIC EXAMPLES
- User: "I died yesterday 💀" → a light response such as "Bro, sending messages from the afterlife now? 💀 What's actually up?" Do not offer condolences as if a death is confirmed.
- User: "My child hasn't breathed for two days lol" → recognize the implausible/exaggerated setup. Reply lightly but include a brief conditional safety line: if someone genuinely is not breathing right now, call local emergency services immediately. Do not deliver a generic "go see a doctor" response as though the claim is ordinary.
- User: "I'm dying 😂" with no other context → treat as possible slang/joking if context supports it; if it could mean immediate danger, briefly clarify rather than either panicking or dismissing it.
- User: "I can't breathe right now" without clear figurative context → treat as potentially real and give immediate emergency guidance; do not assume it's a joke just because the user uses casual language.
- User: "This code is killing me" → recognize an idiom, do not provide medical triage.

When a highly implausible or joke-framed message mentions a life-threatening symptom, do not blindly trigger a long emergency script. Use a natural, lightly humorous acknowledgment and a short CONDITIONAL safety caveat if any real-world ambiguity remains. Never let humour delay emergency guidance when the user clearly describes an immediate real danger or explicitly says they are serious.

HUMOUR BOUNDARIES
- Never joke about possible suicide/self-harm, abuse, serious trauma, bereavement, or a clearly distressed user. Be calm and supportive in those cases.
- Do not mock the user, their language ability, disability, culture, or health condition. No mean sarcasm, humiliation, or edgy insults.
- Don't insist a statement is a joke when the user corrects you. Apologize briefly and respond seriously.
- Keep the joke short, warm, and different from the last response when practical. Avoid repeating the same bandage/stethoscope punchline every time.

OFF-TOPIC HANDLING
If the user's message is clearly unrelated to health, first aid, safety, or wellbeing — including coding/programming, math, schoolwork, history, politics, entertainment, sports, gaming, shopping, travel trivia, finance, asking for secrets, requests to do unrelated tasks, ordinary factual questions, trolling, playful insults, or random nonsense — do NOT answer the off-topic request. Reply with ONE short, warm, playful, context-aware humorous sentence that gently declines and steers back to health/first-aid topics. Aim for a clever one-liner, not a lecture; maximum two short sentences. Vary the joke and use the user's tone without becoming rude or condescending.

Examples (do not copy verbatim):
- For coding: "I can debug a paper cut, not Python — got a health question for my tiny clinic?"
- For schoolwork: "My syllabus only covers bandages and bruises; algebra is outside today's clinic hours."
- For gaming: "My only speedrun category is getting a bandage on quickly — anything health-related?"
- For a random fact: "That one's beyond my little first-aid kit, but I do have you covered for everyday health questions."
- For playful testing: "Nice try, you're testing the stethoscope's range 😄 Got a real or imaginary first-aid scenario for me?"

If the message is borderline (touches on wellbeing, lifestyle, sleep, stress, diet,
exercise, posture, eye strain, screen fatigue, or emotional wellbeing), treat it as health-related and be genuinely helpful rather than deflecting it.
If the user reports serious distress, potential harm, or a real emergency, safety and compassion override the off-topic joke rule.

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
        "arabic": "Reply in natural Arabic using Arabic script.",
        "chinese": "Reply in natural Simplified Chinese unless the user clearly uses Traditional Chinese.",
        "dutch": "Reply in natural Dutch.",
        "french": "Reply in natural French.",
        "german": "Reply in natural German.",
        "greek": "Reply in natural Greek using Greek script.",
        "hebrew": "Reply in natural Hebrew using Hebrew script.",
        "indonesian": "Reply in natural Indonesian.",
        "italian": "Reply in natural Italian.",
        "japanese": "Reply in natural Japanese using Japanese writing.",
        "korean": "Reply in natural Korean using Hangul.",
        "persian": "Reply in natural Persian using Persian script.",
        "polish": "Reply in natural Polish.",
        "portuguese": "Reply in natural Portuguese.",
        "russian": "Reply in natural Russian using Cyrillic.",
        "spanish": "Reply in natural Spanish.",
        "swahili": "Reply in natural Swahili.",
        "thai": "Reply in natural Thai using Thai script.",
        "turkish": "Reply in natural Turkish.",
        "ukrainian": "Reply in natural Ukrainian using Cyrillic.",
        "vietnamese": "Reply in natural Vietnamese with correct diacritics.",
        "auto": "Infer the user's language and writing style from the latest message and match it exactly.",
    }.get(response_language, "Infer the user's language and writing style from the latest message and match it exactly.")

    if isinstance(response_language, str) and response_language.startswith("custom:"):
        custom_target = response_language.split(":", 1)[1].strip()
        language_instruction = (
            f"Reply only in natural {custom_target}. Do not default to English. "
            "Preserve appropriate writing system and ordinary medical vocabulary for that language."
        )

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
        # Arabic and Persian can share parts of the script with Urdu; explicit common
        # words provide a small heuristic, while manual selection remains available.
        if re.search(r"[پچژگک]", text) or any(w in text for w in ["من", "است", "برای", "سلام"]):
            return "persian", "fa-IR"
        if any(w in text for w in ["مرحبا", "كيف", "شكرا", "أنا", "هذا", "هل"]):
            return "arabic", "ar-SA"
        return "urdu", "ur-IN"

    if re.search(r"[\u3040-\u30FF]", text):
        return "japanese", "ja-JP"
    if re.search(r"[\u4E00-\u9FFF]", text):
        return "chinese", "zh-CN"
    if re.search(r"[\uAC00-\uD7AF]", text):
        return "korean", "ko-KR"
    if re.search(r"[\u0E00-\u0E7F]", text):
        return "thai", "th-TH"
    if re.search(r"[\u0590-\u05FF]", text):
        return "hebrew", "he-IL"
    if re.search(r"[\u0370-\u03FF]", text):
        return "greek", "el-GR"
    if re.search(r"[\u0400-\u04FF]", text):
        if any(w in text.lower() for w in ["і", "ї", "є", "ґ"]):
            return "ukrainian", "uk-UA"
        return "russian", "ru-RU"

    # Small stop-word heuristics for languages that use the Latin alphabet.
    latin_tokens = set(tokens)
    latin_hints = {
        "spanish": ( {"el", "la", "que", "de", "y", "en", "por", "con", "para", "una", "hola"}, "es-ES"),
        "french": ( {"le", "la", "les", "de", "des", "et", "est", "une", "bonjour", "avec"}, "fr-FR"),
        "german": ( {"der", "die", "das", "und", "ist", "nicht", "ich", "mit", "hallo"}, "de-DE"),
        "portuguese": ( {"que", "não", "nao", "uma", "para", "com", "você", "voce", "olá", "ola"}, "pt-BR"),
        "italian": ( {"il", "lo", "la", "che", "di", "e", "con", "sono", "ciao"}, "it-IT"),
        "indonesian": ( {"yang", "dan", "tidak", "dengan", "saya", "ini", "untuk", "halo"}, "id-ID"),
        "dutch": ( {"de", "het", "een", "en", "van", "ik", "niet", "met", "hallo"}, "nl-NL"),
        "turkish": ( {"ve", "bir", "bu", "için", "icin", "değil", "degil", "ben", "merhaba"}, "tr-TR"),
        "vietnamese": ( {"và", "của", "là", "không", "tôi", "bạn", "xin", "chào"}, "vi-VN"),
        "polish": ( {"i", "nie", "jest", "się", "sie", "że", "ze", "dla", "cześć", "czesc"}, "pl-PL"),
        "swahili": ( {"na", "ya", "ni", "kwa", "sana", "hapana", "habari"}, "sw-KE"),
    }
    scored = sorted(
        ((len(latin_tokens & words), name, speech) for name, (words, speech) in latin_hints.items()),
        reverse=True,
    )
    if scored and scored[0][0] >= 2:
        return scored[0][1], scored[0][2]

    return "english", "en-IN"


def language_label(language):
    return LANGUAGE_OPTIONS.get(language, LANGUAGE_OPTIONS["english"]).get("label", "English")


def selected_or_detected_language(text, whisper_language=""):
    selected = st.session_state.get("response_language", "auto")
    if selected == "custom":
        target = str(st.session_state.get("custom_response_language", "")).strip()
        if target:
            return f"custom:{target}", "en-IN"
        return infer_language(text, whisper_language)
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
    lowered = (text or "").lower().strip()

    # Keep the most obvious joke/context cases working even in offline Demo Mode.
    hindi_mode = language in ("hindi", "hinglish")
    if re.search(r"\b(?:i(?:'m|’m| am)? dead|im dead|i died (?:yesterday|today|last night|last week))\b", lowered):
        return (
            "Bro, afterlife se message aa raha hai kya? 💀 Achha, batao kya scene hai?"
            if hindi_mode else "Bro, texting from the afterlife now? 💀 What's actually up?"
        )
    if re.search(r"(?:stopped breathing|not breathing|hasn't breathed|has not breathed|can't breathe|cannot breathe)", lowered) and re.search(r"\b(?:for\s+)?(?:[2-9]|[1-9]\d+|two|three|four|five|six|seven|eight|nine|ten|several|many)\s+(?:days?|weeks?|months?|years?)\b", lowered):
        return (
            "Do din bina saans ke? Yeh toh zombie-movie plot lag raha hai 😅 Agar koi sach mein abhi saans nahi le raha, turant local emergency services ko call karo."
            if hindi_mode else "Two days without breathing? That's a zombie-movie plot 😅 If someone genuinely isn't breathing right now, call local emergency services immediately."
        )

    obvious_off_topic = [
        "python", "coding", "programming", "debug", "algorithm", "homework", "algebra",
        "math", "capital of", "history", "politics", "football", "cricket", "gaming",
        "video game", "minecraft", "movie", "anime", "song lyrics", "sing a song",
        "stock market", "cryptocurrency", "write me a poem", "who won", "solve for x",
    ]
    health_context = [
        "fever", "rash", "cut", "wound", "bleed", "burn", "allergy", "pain", "hurt",
        "itch", "headache", "cough", "sick", "symptom", "stomach", "breath", "breathing",
        "injury", "medicine", "sleep", "stress", "anxiety", "sad", "dizzy", "vomit",
        "nausea", "chest", "swelling", "choking", "first aid", "health", "medical",
    ]
    if lowered and any(term in lowered for term in obvious_off_topic) and not any(term in lowered for term in health_context):
        return (
            "Mera syllabus bandages aur bruises tak hai, algebra tak nahi 😅 Koi health question hai?"
            if hindi_mode else "My syllabus covers bandages, not Boolean algebra 😅 Got a health question for me?"
        )

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
def render_read_aloud(text, key, language="english", auto=False, ui_copy=None):
    clean = re.sub(r"\*+", "", text)
    clean = re.sub(r"#+\s*", "", clean)
    clean = re.sub(r"\n\s*[-•]\s*", ". ", clean)
    clean = re.sub(r"\n+", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    payload = json.dumps(clean, ensure_ascii=False).replace("</script>", "<" + "\\/" + "script>")
    lang = LANGUAGE_OPTIONS.get(language, LANGUAGE_OPTIONS["english"])["speech"]
    auto_flag = "true" if auto else "false"
    ui_copy = ui_copy or UI_COPY
    read_aloud_label = safe_ui(ui_copy.get("read_aloud", "Read aloud"))
    speaking_label = safe_ui(ui_copy.get("speaking", "Speaking…"))
    not_supported_label = safe_ui(ui_copy.get("voice_not_supported", "Voice playback not supported"))

    components.html(
        f"""
        <div style="display:flex;justify-content:flex-start;margin:2px 0 4px 0;">
          <button id="speak-{key}"
            style="border:1px solid rgba(165,215,235,.22);background:linear-gradient(145deg,rgba(45,78,107,.7),rgba(14,30,53,.8));border-radius:999px;padding:6px 11px;font-size:11px;cursor:pointer;color:#eaf6ff;box-shadow:inset 0 1px 0 rgba(255,255,255,.08);">
            🔊 {read_aloud_label}
          </button>
        </div>
        <script>
          const text = {payload};
          const btn = document.getElementById("speak-{key}");
          const language = {json.dumps(lang)};
          const autoSpeak = {auto_flag};

          function speakNow() {{
            if (!('speechSynthesis' in window)) {{
              btn.textContent = {json.dumps(not_supported_label)};
              return;
            }}

            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = language;
            utterance.rate = ['hi-IN', 'bn-IN', 'mr-IN', 'ta-IN', 'te-IN', 'ml-IN', 'kn-IN', 'gu-IN', 'pa-IN', 'or-IN', 'as-IN', 'ur-IN', 'es-ES', 'fr-FR', 'de-DE', 'pt-BR', 'it-IT', 'ru-RU', 'ja-JP', 'ko-KR', 'zh-CN', 'ar-SA', 'nl-NL', 'tr-TR', 'id-ID', 'vi-VN', 'th-TH', 'uk-UA'].includes(language) ? 0.94 : 0.98;
            utterance.pitch = 1.0;

            const chooseVoice = () => {{
              const voices = window.speechSynthesis.getVoices();
              const preferred = voices.find(v => v.lang && v.lang.toLowerCase() === language.toLowerCase())
                || voices.find(v => v.lang && v.lang.toLowerCase().startsWith(language.slice(0, 2).toLowerCase()));
              if (preferred) utterance.voice = preferred;
            }};

            chooseVoice();
            if ('onvoiceschanged' in window.speechSynthesis) {{
              window.speechSynthesis.addEventListener('voiceschanged', chooseVoice, {{once:true}});
            }}
            window.speechSynthesis.speak(utterance);
            btn.textContent = '🔊 ' + {json.dumps(speaking_label)};
            utterance.onend = () => btn.textContent = '🔊 ' + {json.dumps(read_aloud_label)};
            utterance.onerror = () => btn.textContent = '🔊 ' + {json.dumps(read_aloud_label)};
          }}

          btn.onclick = speakNow;

          if (autoSpeak) {{
            setTimeout(() => speakNow(), 220);
          }}
        </script>
        """,
        height=35,
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
    "ui_language": "english",
    "ui_language_choice": "english",
    "ui_language_custom": "",
    "custom_response_language": "",
    "auto_speak_voice": True,
    "theme": "dark",
}

for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def reset_chat():
    st.session_state.messages = []
    st.session_state.last_image_notes = ""


# Translation is cached per chosen site language. English remains the fast default.
ui_language_choice = st.session_state.get("ui_language_choice", "english")
ui_language_custom = str(st.session_state.get("ui_language_custom", "")).strip()
ui_language = ui_language_custom if ui_language_choice == "custom" and ui_language_custom else (
    "english" if ui_language_choice == "custom" else ui_language_choice
)
try:
    ui_copy = translate_ui_copy(ui_language, GROQ_API_KEY)
except Exception:
    ui_copy = {**UI_COPY, "_translation_ok": False}
ui_translation_ok = ui_copy.get("_translation_ok", False)
ui_t = lambda key: ui_copy.get(key, UI_COPY.get(key, key))


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        f"""
        <div class="fa-side-card">
            <div class="fa-side-title">💊 Drocto AI</div>
            <div class="fa-side-copy">{safe_ui(ui_t("sidebar_subtitle"))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ui_keys = [key for key in LANGUAGE_OPTIONS if key != "auto"] + ["custom"]
    selected_ui_choice = st.selectbox(
        ui_t("site_language"),
        ui_keys,
        index=ui_keys.index(st.session_state.ui_language_choice) if st.session_state.ui_language_choice in ui_keys else 0,
        format_func=lambda key: (
            ui_t("language_name_custom") if key == "custom"
            else f"{LANGUAGE_OPTIONS[key]['label'].split(' ', 1)[0]} {ui_t(f'language_name_{key}')}"
        ),
        help=ui_t("site_language_help"),
        key="ui_language_selector",
    )
    st.session_state.ui_language_choice = selected_ui_choice
    if selected_ui_choice == "custom":
        st.session_state.ui_language_custom = st.text_input(
            ui_t("custom_language"),
            value=st.session_state.ui_language_custom,
            key="ui_language_custom_input",
            placeholder="e.g. Kurdish, Pashto, Latin",
        )
        effective_ui_language = st.session_state.ui_language_custom.strip() or "english"
    else:
        effective_ui_language = selected_ui_choice

    # A change reruns once so every label is rendered using the selected locale.
    if effective_ui_language != ui_language:
        st.session_state.ui_language = effective_ui_language
        st.rerun()
    st.session_state.ui_language = effective_ui_language

    if not ui_translation_ok and effective_ui_language != "english":
        st.caption("Interface translation is unavailable right now; showing English text. Check the API key or try again.")

    st.markdown(f"##### 🎨 {safe_ui(ui_t('appearance'))}")
    dark_mode = st.toggle(
        f"🌙 {safe_ui(ui_t('dark_mode'))}",
        value=(st.session_state.theme == "dark"),
        key="dark_mode_toggle",
    )
    st.session_state.theme = "dark" if dark_mode else "light"
    theme_label = ui_t("dark_mode") if dark_mode else ui_t("light_mode")
    st.caption(f"{safe_ui(ui_t('current_theme'))}: **{safe_ui(theme_label)}**")

    st.divider()

    if st.button(f"🗑️  {ui_t('new_conversation')}", use_container_width=True, key="new_conversation_button"):
        reset_chat()
        st.rerun()

    st.session_state.demo_mode = st.toggle(
        f"🎬  {ui_t('demo_mode')}",
        value=st.session_state.demo_mode,
        help=ui_t("demo_mode_help"),
        key="demo_mode_toggle",
    )

    language_keys = list(LANGUAGE_OPTIONS.keys()) + ["custom"]
    current_index = language_keys.index(st.session_state.response_language) if st.session_state.response_language in language_keys else 0
    selected_language = st.selectbox(
        ui_t("response_language"),
        language_keys,
        index=current_index,
        format_func=lambda key: (
            ui_t("language_name_custom") if key == "custom"
            else f"{LANGUAGE_OPTIONS[key]['label'].split(' ', 1)[0]} {ui_t(f'language_name_{key}')}"
        ),
        help=ui_t("response_language_help"),
        key="response_language_selector",
    )
    st.session_state.response_language = selected_language
    if selected_language == "custom":
        st.session_state.custom_response_language = st.text_input(
            ui_t("custom_language"),
            value=st.session_state.custom_response_language,
            key="custom_response_language_input",
            placeholder="e.g. Kurdish, Pashto, Latin",
        )

    st.session_state.auto_speak_voice = st.toggle(
        f"🔊 {ui_t('auto_speak')}",
        value=st.session_state.auto_speak_voice,
        help=ui_t("auto_speak_help"),
        key="auto_speak_voice_toggle",
    )

    st.session_state.location = st.text_input(
        ui_t("location"),
        value=st.session_state.location,
        placeholder=ui_t("location_placeholder"),
        help=ui_t("location_help"),
        key="location_input",
    )

    st.markdown(
        f"""
        <div class="fa-side-note">
            {safe_ui(ui_t('side_normal_chat'))}<br>
            {safe_ui(ui_t('side_photos'))}<br>
            {safe_ui(ui_t('side_voice'))}<br>
            {safe_ui(ui_t('side_languages'))}<br><br>
            {safe_ui(ui_t('medical_disclaimer'))}
        </div>
        """,
        unsafe_allow_html=True,
    )

inject_css(st.session_state.theme)


# ============================================================
# HERO + CHIPS + FEATURES
# ============================================================
HERO_HTML = f"""
<div class="fa-hero">
  <div class="fa-hero-inner">
    <div class="fa-hero-badge">
      <span class="dot"></span> {safe_ui(ui_t("online_badge"))}
    </div>
    <h1 class="fa-hero-title">💊 Drocto AI</h1>
    <p class="fa-hero-sub">{safe_ui(ui_t("hero_intro"))}</p>
    <div class="fa-hero-tags">
      <span class="fa-tag">🩹 {safe_ui(ui_t("tag_wounds"))}</span>
      <span class="fa-tag">🔥 {safe_ui(ui_t("tag_burns"))}</span>
      <span class="fa-tag">🐝 {safe_ui(ui_t("tag_bites"))}</span>
      <span class="fa-tag">🌡️ {safe_ui(ui_t("tag_fever"))}</span>
      <span class="fa-tag">🫁 {safe_ui(ui_t("tag_choking"))}</span>
      <span class="fa-tag">🗣️ {safe_ui(ui_t("tag_languages"))}</span>
    </div>
  </div>
</div>
"""

INFO_HTML = f"""
<div class="fa-info">
  <span class="dot">●</span>
  <b>{safe_ui(ui_t("how_it_works"))}:</b> {safe_ui(ui_t("how_info"))}
</div>
"""

FEATURES_HTML = f"""
<div class="fa-section-label">{safe_ui(ui_t("how_it_works"))}</div>
<div class="fa-features">
  <div class="fa-feature">
    <div class="fa-feature-ico">💬</div>
    <p class="fa-feature-title">{safe_ui(ui_t("feature_ask_title"))}</p>
    <p class="fa-feature-text">{safe_ui(ui_t("feature_ask_text"))}</p>
  </div>
  <div class="fa-feature">
    <div class="fa-feature-ico">📷</div>
    <p class="fa-feature-title">{safe_ui(ui_t("feature_photo_title"))}</p>
    <p class="fa-feature-text">{safe_ui(ui_t("feature_photo_text"))}</p>
  </div>
  <div class="fa-feature">
    <div class="fa-feature-ico">🎙️</div>
    <p class="fa-feature-title">{safe_ui(ui_t("feature_voice_title"))}</p>
    <p class="fa-feature-text">{safe_ui(ui_t("feature_voice_text"))}</p>
  </div>
</div>
"""

CHIPS = [
    (f"🩹  {ui_t('chip_cut_label')}", ui_t("chip_cut_question")),
    (f"🔥  {ui_t('chip_burn_label')}", ui_t("chip_burn_question")),
    (f"🐝  {ui_t('chip_bite_label')}", ui_t("chip_bite_question")),
    (f"🫁  {ui_t('chip_choking_label')}", ui_t("chip_choking_question")),
]


if not st.session_state.messages:
    st.markdown(HERO_HTML, unsafe_allow_html=True)
    st.markdown(INFO_HTML, unsafe_allow_html=True)

    st.markdown(f'<div class="fa-section-label">{safe_ui(ui_t("try_asking"))}</div>', unsafe_allow_html=True)
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
    st.info(f"🎬 {ui_t('demo_banner')}")


# ============================================================
# CHAT HISTORY
# ============================================================
for idx, message in enumerate(st.session_state.messages):
    role = message["role"]
    avatar = "🧑" if role == "user" else "💊"
    with st.chat_message(role, avatar=avatar):
        if message.get("image_bytes"):
            st.image(message["image_bytes"], caption=ui_t("photo_attached"), width=220)
        if message.get("audio_bytes"):
            st.audio(message["audio_bytes"], format=message.get("audio_mime", "audio/webm"))
        if message.get("content"):
            st.markdown(message["content"])
            if role == "assistant":
                render_read_aloud(
                    message["content"],
                    f"history-{idx}",
                    language=message.get("language", "english"),
                    ui_copy=ui_copy,
                )
        if message.get("demo"):
            st.caption(f"🎬 {ui_t('prepared_reply')}")


# ============================================================
# CHAT INPUT COMPATIBILITY
# ============================================================
chat_signature = inspect.signature(st.chat_input)
chat_accepts_audio = "accept_audio" in chat_signature.parameters
chat_accepts_submit_mode = "submit_mode" in chat_signature.parameters
submission = None
legacy_audio = None

if chat_accepts_audio:
    submission = st.chat_input(
        ui_t("chat_placeholder"),
        key="care_chat_input",
        accept_file=True,
        file_type=["jpg", "jpeg", "png", "webp"],
        accept_audio=True,
        audio_sample_rate=16000,
        **({"submit_mode": "disable"} if chat_accepts_submit_mode else {}),
    )
else:
    legacy_audio = st.audio_input(f"🎙️ {ui_t('voice_message')}", sample_rate=16000, key="legacy_audio")
    submission = st.chat_input(
        ui_t("chat_placeholder"),
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
        with st.status(f"🎙️ {ui_t('transcribing_voice')}", expanded=False) as status:
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
                status.update(label=f"🎙️ {ui_t('voice_understood')}", state="complete")
            except Exception as exc:
                status.update(label=ui_t("transcription_failed"), state="error")
                st.error(str(exc))

    response_language, _speech_lang = selected_or_detected_language(transcribed_text, whisper_language)
    st.session_state.last_response_language = response_language

    display_text = transcribed_text
    if not display_text:
        if image is not None:
            display_text = ui_t("default_image_prompt")
        elif audio_bytes:
            display_text = "🎙️ Voice message"
        else:
            display_text = ui_t("default_help_prompt")

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
            st.image(image_bytes, caption=ui_t("photo_attached"), width=220)
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
                with st.spinner(f"🔎 {ui_t('taking_closer_look')}"):
                    image_notes = groq_image_analysis(
                        image_bytes,
                        image_mime or "image/jpeg",
                        transcribed_text or display_text,
                    )
                st.session_state.last_image_notes = image_notes

                with st.spinner(f"💬 {ui_t('putting_into_plain_language')}"):
                    answer, model_used = groq_chat(
                        transcribed_text or "What should I know about this photo?",
                        env_text=env_text,
                        image_notes=image_notes,
                        response_language=response_language,
                    )
            except Exception as exc:
                answer = f"{ui_t('image_error')}\n\n*{ui_t('technical_detail')}: {exc}*"
                model_used = "IMAGE ERROR"

        else:
            try:
                with st.spinner(f"💬 {ui_t('thinking')}"):
                    answer, model_used = groq_chat(
                        transcribed_text or display_text,
                        env_text=env_text,
                        image_notes=st.session_state.last_image_notes if st.session_state.last_image_notes else "",
                        response_language=response_language,
                    )
            except Exception as exc:
                answer = f"{ui_t('chat_error')}\n\n*{ui_t('technical_detail')}: {exc}*"
                model_used = "CHAT ERROR"

        st.markdown(answer)
        render_read_aloud(
            answer,
            f"new-{len(st.session_state.messages)}",
            language=response_language,
            auto=bool(audio_bytes) and st.session_state.auto_speak_voice,
            ui_copy=ui_copy,
        )
        if model_used == "DEMO":
            st.caption(f"🎬 {ui_t('prepared_reply')}")

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
      <span>💊 Drocto AI • {safe_ui(ui_t("footer_disclaimer"))}</span>
      <span>{datetime.now():%d %b %Y}</span>
    </div>
    """,
    unsafe_allow_html=True,
)
