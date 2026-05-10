
import streamlit as st
from google import genai
from google.genai import types
import json
from PIL import Image
import io
import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import numpy as np
import os
import hashlib
import re

# ── Auto-load API key from .env file (safe — never hardcoded) ─────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed — no problem, falls back to manual entry

_ENV_API_KEY = os.environ.get("GEMINI_API_KEY", "")
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ── TensorFlow (lazy-loaded to avoid slowing startup) ─────────────────────────
@st.cache_resource(show_spinner=False)
def load_tf_model():
    """Load EfficientNetB0 pretrained on ImageNet. Cached globally."""
    import tensorflow as tf
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
    model = EfficientNetB0(weights="imagenet", include_top=True)
    return model, preprocess_input, decode_predictions

# ── ImageNet class → Waste category mapping ───────────────────────────────────
# Maps top-1000 ImageNet synset labels to our 5 waste categories.
# Keys are substrings that appear in ImageNet class names (lowercased).
IMAGENET_TO_WASTE = {
    # Plastic
    "water bottle":     "Plastic", "bottle":           "Plastic",
    "plastic bag":      "Plastic", "milk can":         "Plastic",
    "pop bottle":       "Plastic", "soap dispenser":   "Plastic",
    "bucket":           "Plastic", "vase":             "Plastic",
    "pitcher":          "Plastic", "jug":              "Plastic",
    "cup":              "Plastic", "container":        "Plastic",
    "tray":             "Plastic", "packet":           "Plastic",

    # Organic
    "banana":           "Organic", "apple":            "Organic",
    "orange":           "Organic", "lemon":            "Organic",
    "strawberry":       "Organic", "pineapple":        "Organic",
    "broccoli":         "Organic", "cauliflower":      "Organic",
    "corn":             "Organic", "mushroom":         "Organic",
    "artichoke":        "Organic", "cucumber":         "Organic",
    "zucchini":         "Organic", "acorn":            "Organic",
    "fig":              "Organic", "pomegranate":      "Organic",
    "meat loaf":        "Organic", "burrito":          "Organic",
    "pizza":            "Organic", "pretzel":          "Organic",
    "bread":            "Organic", "bagel":            "Organic",
    "eggnog":           "Organic", "espresso":         "Organic",
    "carbonara":        "Organic", "guacamole":        "Organic",
    "head cabbage":     "Organic", "leaf":             "Organic",

    # Metal
    "can opener":       "Metal",   "tin can":          "Metal",
    "iron":             "Metal",   "nail":             "Metal",
    "screw":            "Metal",   "wrench":           "Metal",
    "hammer":           "Metal",   "padlock":          "Metal",
    "chain":            "Metal",   "hook":             "Metal",
    "shovel":           "Metal",   "ladle":            "Metal",
    "frying pan":       "Metal",   "wok":              "Metal",
    "safe":             "Metal",   "lock":             "Metal",
    "scissors":         "Metal",   "knife":            "Metal",
    "fork":             "Metal",   "spoon":            "Metal",
    "spatula":          "Metal",   "tongs":            "Metal",
    "file":             "Metal",   "steel drum":       "Metal",
    "barbell":          "Metal",   "dumbbell":         "Metal",
    "bicycle":          "Metal",   "car wheel":        "Metal",
    "hubcap":           "Metal",   "mailbox":          "Metal",

    # Paper
    "book":             "Paper",   "newspaper":        "Paper",
    "envelope":         "Paper",   "comic book":       "Paper",
    "notebook":         "Paper",   "binder":           "Paper",
    "menu":             "Paper",   "label":            "Paper",
    "cardboard":        "Paper",   "carton":           "Paper",
    "toilet tissue":    "Paper",   "paper towel":      "Paper",
    "tissue":           "Paper",   "letter opener":    "Paper",
    "ink cartridge":    "Paper",

    # E-waste
    "laptop":           "E-waste", "computer":         "E-waste",
    "desktop computer": "E-waste", "monitor":          "E-waste",
    "screen":           "E-waste", "television":       "E-waste",
    "remote control":   "E-waste", "mouse":            "E-waste",
    "keyboard":         "E-waste", "speaker":          "E-waste",
    "ipod":             "E-waste", "cellular telephone":"E-waste",
    "phone":            "E-waste", "modem":            "E-waste",
    "router":           "E-waste", "hard disc":        "E-waste",
    "disk brake":       "E-waste", "printer":          "E-waste",
    "electric fan":     "E-waste", "hand blender":     "E-waste",
    "mixer":            "E-waste", "microwave":        "E-waste",
    "washer":           "E-waste", "refrigerator":     "E-waste",
    "vacuum":           "E-waste", "battery":          "E-waste",
    "earphone":         "E-waste", "headphone":        "E-waste",
    "projector":        "E-waste", "camera":           "E-waste",
    "web site":         "E-waste",
}

def map_imagenet_to_waste(predictions):
    """
    Given decode_predictions output, map top-5 ImageNet classes
    to our waste categories and return category votes with confidence.
    Returns (category, confidence_pct, matched_label, all_scores_dict)
    """
    category_scores = {"Plastic": 0.0, "Organic": 0.0, "Metal": 0.0, "Paper": 0.0, "E-waste": 0.0}
    matched_label   = None

    for _, label, prob in predictions[0]:
        label_lower = label.lower().replace("_", " ")
        for keyword, category in IMAGENET_TO_WASTE.items():
            if keyword in label_lower:
                category_scores[category] += float(prob)
                if matched_label is None:
                    matched_label = label.replace("_", " ").title()
                break  # each prediction counts once

    top_cat   = max(category_scores, key=category_scores.get)
    top_score = category_scores[top_cat]

    # If nothing matched confidently, call it "Organic" as a safe fallback
    if top_score < 0.01:
        top_cat   = "Organic"
        top_score = 0.30

    # Normalise scores to 0-100 for display
    total = sum(category_scores.values()) or 1.0
    normalised = {k: int(v / total * 100) for k, v in category_scores.items()}

    confidence_pct = min(99, int(top_score * 400))  # scale for display
    return top_cat, confidence_pct, matched_label or "unknown object", normalised


def classify_with_tf(image: Image.Image) -> dict:
    """
    Run EfficientNetB0 (ImageNet pretrained) on the image.
    Returns a dict matching the same schema as classify_waste().
    """
    import tensorflow as tf
    model, preprocess_input, decode_predictions = load_tf_model()

    # Resize & preprocess for EfficientNetB0 (224×224)
    img_resized = image.resize((224, 224))
    img_array   = np.array(img_resized, dtype=np.float32)
    img_array   = np.expand_dims(img_array, axis=0)
    img_array   = preprocess_input(img_array)

    preds       = model.predict(img_array, verbose=0)
    decoded     = decode_predictions(preds, top=10)

    category, confidence, matched_label, scores = map_imagenet_to_waste(decoded)

    reason = (
        f"EfficientNetB0 (ImageNet) detected visual features resembling "
        f"'{matched_label}', mapped to the {category} waste category."
    )
    return {
        "category":   category,
        "confidence": confidence,
        "reason":     reason,
        "scores":     scores,
        "source":     "tensorflow",
    }

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Waste AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

/* ── DARK THEME (default) ── */
:root {
  --lime:#b8ff00; --lime-dim:#94cc00; --lime-soft:rgba(184,255,0,0.06);
  --teal:#00ffd5; --amber:#ffaa00; --rose:#ff4d6d; --violet:#9d4edd;
  --surface:#040804; --surface-2:#080f08; --surface-3:#0d150d;
  --glass:rgba(8,14,8,0.88); --glass-2:rgba(12,20,12,0.80);
  --border:rgba(184,255,0,0.09); --border-mid:rgba(184,255,0,0.20); --border-bright:rgba(184,255,0,0.38);
  --text:#edfce4; --text-2:#84a86c; --text-3:#3a5628;
  --plastic:#38bdf8; --organic:#4ade80; --metal:#fbbf24; --paper:#f472b6; --ewaste:#a78bfa;
  --r:20px;
}

/* ── LIGHT THEME ── */
body.light-mode {
  --lime:#2e7d00; --lime-dim:#245f00; --lime-soft:rgba(46,125,0,0.07);
  --teal:#007a6a; --amber:#b87700; --rose:#c0002a; --violet:#6b21d4;
  --surface:#f4f9f0; --surface-2:#eaf3e4; --surface-3:#d8eccc;
  --glass:rgba(255,255,255,0.90); --glass-2:rgba(240,248,234,0.92);
  --border:rgba(46,125,0,0.14); --border-mid:rgba(46,125,0,0.28); --border-bright:rgba(46,125,0,0.50);
  --text:#0f1f0a; --text-2:#3a5628; --text-3:#7aaa55;
  --plastic:#1a6fb5; --organic:#1e7a3a; --metal:#b87700; --paper:#b52070; --ewaste:#6b21d4;
}
body.light-mode .stApp{background:var(--surface)!important;}
body.light-mode .stApp::before{
  background:radial-gradient(ellipse 80% 60% at -5% -5%,rgba(46,125,0,0.07) 0%,transparent 55%),
  radial-gradient(ellipse 60% 45% at 105% 105%,rgba(0,122,106,0.05) 0%,transparent 50%)!important;}
body.light-mode .stApp::after{background-image:radial-gradient(circle,rgba(46,125,0,0.04) 1px,transparent 1px)!important;}
body.light-mode [data-testid="stSidebar"]{background:linear-gradient(180deg,#e8f5e0 0%,#f0f8e8 100%)!important;border-right:1px solid rgba(46,125,0,0.12)!important;box-shadow:none!important;}
body.light-mode .card,body.light-mode .result-card,body.light-mode .glass-card{background:rgba(255,255,255,0.94)!important;border-color:rgba(46,125,0,0.13)!important;box-shadow:0 4px 24px rgba(0,0,0,0.07)!important;}
body.light-mode .stButton>button{background:linear-gradient(135deg,rgba(46,125,0,0.09),rgba(46,125,0,0.04))!important;border-color:rgba(46,125,0,0.22)!important;color:var(--lime)!important;}
body.light-mode .stTextInput>div>div>input,body.light-mode .stTextArea textarea{background:rgba(255,255,255,0.97)!important;border-color:rgba(46,125,0,0.15)!important;color:var(--text)!important;}
body.light-mode .history-row{background:rgba(240,248,234,0.97)!important;border-color:rgba(46,125,0,0.10)!important;}
body.light-mode [data-testid="stFileUploadDropzone"]{background:rgba(240,248,234,0.9)!important;border-color:rgba(46,125,0,0.18)!important;}
body.light-mode .suggest-card{background:linear-gradient(145deg,rgba(255,255,255,0.97),rgba(240,248,234,0.99))!important;border-color:rgba(46,125,0,0.12)!important;}
body.light-mode .stTabs [data-baseweb="tab-list"]{background:rgba(232,245,224,0.97)!important;}
body.light-mode html,body.light-mode body,body.light-mode [class*="css"]{color:var(--text)!important;}

*,*::before,*::after{box-sizing:border-box;}
html,body,[class*="css"]{font-family:"Plus Jakarta Sans",sans-serif!important;color:var(--text)!important;-webkit-font-smoothing:antialiased!important;}
h1,h2,h3,h4,h5,h6{font-family:"Syne",sans-serif!important;}

/* ── APP BACKGROUND ── */
.stApp{background:var(--surface)!important;min-height:100vh!important;position:relative!important;overflow-x:hidden!important;}
.stApp::before{
  content:"";position:fixed;inset:0;z-index:0;pointer-events:none;
  background:
    radial-gradient(ellipse 80% 60% at -5% -5%,rgba(184,255,0,0.09) 0%,transparent 55%),
    radial-gradient(ellipse 60% 45% at 105% 105%,rgba(0,255,213,0.07) 0%,transparent 50%),
    radial-gradient(ellipse 45% 35% at 55% 45%,rgba(184,255,0,0.022) 0%,transparent 60%),
    radial-gradient(ellipse 70% 50% at 90% 5%,rgba(157,78,221,0.045) 0%,transparent 45%);
  animation:orbDrift 22s ease-in-out infinite alternate;
}
@keyframes orbDrift{0%{transform:scale(1) translate(0,0);}33%{transform:scale(1.03) translate(-8px,6px);}66%{transform:scale(0.98) translate(5px,-4px);}100%{transform:scale(1.02) translate(-3px,8px);}}
.stApp::after{
  content:"";position:fixed;inset:0;z-index:0;pointer-events:none;
  background-image:radial-gradient(circle,rgba(184,255,0,0.055) 1px,transparent 1px);
  background-size:34px 34px;
  mask-image:radial-gradient(ellipse 90% 90% at 50% 50%,black 15%,transparent 100%);
  -webkit-mask-image:radial-gradient(ellipse 90% 90% at 50% 50%,black 15%,transparent 100%);
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"]{background:linear-gradient(180deg,#020602 0%,#050d05 50%,#030803 100%)!important;border-right:1px solid rgba(184,255,0,0.065)!important;box-shadow:6px 0 50px rgba(0,0,0,0.7)!important;}
[data-testid="stSidebar"]>div{padding-top:0;}
[data-testid="stSidebar"] .stMarkdown p{color:var(--text-2)!important;font-size:0.8rem!important;}
[data-testid="stSidebar"] hr{border-color:rgba(184,255,0,0.055)!important;margin:0.7rem 0!important;}

.sidebar-logo{display:flex;align-items:center;gap:0.65rem;padding:1.1rem 1.1rem 1rem;border-bottom:1px solid rgba(184,255,0,0.055);margin-bottom:0.9rem;background:linear-gradient(180deg,rgba(184,255,0,0.022) 0%,transparent 100%);}
.sidebar-logo-icon{width:36px;height:36px;border-radius:11px;flex-shrink:0;background:linear-gradient(135deg,rgba(184,255,0,0.18),rgba(0,255,213,0.08));border:1px solid rgba(184,255,0,0.22);display:flex;align-items:center;justify-content:center;font-size:1.15rem;box-shadow:0 0 20px rgba(184,255,0,0.12);}
.sidebar-logo-text{font-family:"Syne",sans-serif;font-size:0.92rem;font-weight:800;color:var(--lime);letter-spacing:-0.02em;line-height:1.1;}
.sidebar-logo-sub{font-size:0.58rem;color:var(--text-3);letter-spacing:0.1em;font-family:"Space Mono",monospace;margin-top:0.1rem;}

/* ── HERO ── */
.hero-wrap{padding:2.8rem 0 1.2rem;position:relative;}
.hero-eyebrow{display:inline-flex;align-items:center;gap:0.5rem;font-family:"Space Mono",monospace;font-size:0.62rem;color:var(--lime-dim);letter-spacing:0.24em;text-transform:uppercase;background:var(--lime-soft);border:1px solid rgba(184,255,0,0.16);padding:0.26rem 0.9rem;border-radius:100px;margin-bottom:1.1rem;animation:heroIn 0.7s cubic-bezier(0.16,1,0.3,1) both;}
.hero-eyebrow-dot{width:5px;height:5px;border-radius:50%;background:var(--lime);box-shadow:0 0 8px var(--lime);animation:blinkDot 2.4s ease-in-out infinite;flex-shrink:0;}
@keyframes blinkDot{0%,100%{opacity:1;}50%{opacity:0.2;}}
.hero-title{font-family:"Syne",sans-serif;font-size:clamp(2.2rem,4.5vw,3.8rem);font-weight:800;line-height:0.94;letter-spacing:-0.05em;margin-bottom:0.75rem;background:linear-gradient(130deg,#b8ff00 0%,#d4ff66 28%,#00ffd5 60%,#66ffee 80%,#b8ff00 100%);background-size:300% 300%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;filter:drop-shadow(0 0 50px rgba(184,255,0,0.18));animation:shimmer 8s ease infinite,heroIn 0.7s cubic-bezier(0.16,1,0.3,1) 0.08s both;}
@keyframes shimmer{0%,100%{background-position:0% 50%;}50%{background-position:100% 50%;}}
@keyframes heroIn{from{opacity:0;transform:translateY(22px);}to{opacity:1;transform:translateY(0);}}
.hero-sub{font-size:0.72rem;color:var(--text-3);letter-spacing:0.16em;text-transform:uppercase;font-weight:700;font-family:"Space Mono",monospace;animation:heroIn 0.7s cubic-bezier(0.16,1,0.3,1) 0.16s both;}
.hero-divider{width:64px;height:2px;border-radius:2px;background:linear-gradient(90deg,var(--lime),var(--teal),rgba(0,255,213,0));margin:1.1rem 0 1.8rem;animation:heroIn 0.7s cubic-bezier(0.16,1,0.3,1) 0.22s both;}

/* ── CARDS ── */
.card,.result-card,.glass-card{background:var(--glass);border:1px solid var(--border);border-radius:var(--r);padding:1.5rem;margin:0.6rem 0;position:relative;overflow:hidden;backdrop-filter:blur(28px) saturate(1.5);-webkit-backdrop-filter:blur(28px) saturate(1.5);box-shadow:inset 0 1px 0 rgba(184,255,0,0.055),inset 0 0 0 1px rgba(255,255,255,0.012),0 4px 30px rgba(0,0,0,0.55);transition:border-color 0.3s,box-shadow 0.3s,transform 0.25s;}
.card::before,.result-card::before,.glass-card::before{content:"";position:absolute;top:0;left:8%;right:8%;height:1px;background:linear-gradient(90deg,transparent,rgba(184,255,0,0.26),transparent);pointer-events:none;}
.card:hover,.glass-card:hover{border-color:var(--border-mid);box-shadow:inset 0 1px 0 rgba(184,255,0,0.09),0 6px 50px rgba(0,0,0,0.65),0 0 30px rgba(184,255,0,0.04);transform:translateY(-2px);}
.result-card{border-color:rgba(184,255,0,0.11);animation:resultIn 0.55s cubic-bezier(0.16,1,0.3,1) both;}
@keyframes resultIn{from{opacity:0;transform:translateY(28px) scale(0.975);filter:blur(4px);}to{opacity:1;transform:translateY(0) scale(1);filter:blur(0);}}

/* ── CONFIDENCE RING ── */
.conf-ring-wrap{display:flex;flex-direction:column;align-items:center;gap:0.3rem;padding:0.5rem 0;}
.conf-ring{position:relative;width:110px;height:110px;}
.conf-ring svg{transform:rotate(-90deg);}
.conf-ring-track{fill:none;stroke:rgba(184,255,0,0.08);stroke-width:8;}
.conf-ring-fill{fill:none;stroke-width:8;stroke-linecap:round;transition:stroke-dashoffset 1.2s cubic-bezier(0.4,0,0.2,1);}
.conf-ring-label{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;}
.conf-ring-number{font-family:"Syne",sans-serif;font-size:1.75rem;font-weight:800;line-height:1;}
.conf-ring-text{font-family:"Space Mono",monospace;font-size:0.5rem;color:var(--text-3);letter-spacing:0.1em;text-transform:uppercase;}

/* ── CATEGORY BADGE ── */
.category-badge{display:inline-flex;align-items:center;gap:0.5rem;font-family:"Syne",sans-serif;font-size:1rem;font-weight:800;padding:0.5rem 1.3rem;border-radius:100px;border:1px solid currentColor;margin-bottom:1rem;letter-spacing:0.02em;background:rgba(255,255,255,0.025);text-shadow:0 0 24px currentColor;box-shadow:0 0 20px rgba(0,0,0,0.3),inset 0 0 20px rgba(255,255,255,0.02);animation:badgePop 0.5s cubic-bezier(0.34,1.56,0.64,1) 0.15s both;}
@keyframes badgePop{from{opacity:0;transform:scale(0.7);}to{opacity:1;transform:scale(1);}}

/* ── SCORE BARS ── */
.score-bar-wrap{margin:0.32rem 0;}
.score-bar-label{display:flex;justify-content:space-between;font-size:0.78rem;color:var(--text-2);margin-bottom:4px;}
.score-bar-track{height:7px;border-radius:100px;background:rgba(184,255,0,0.055);overflow:hidden;}
.score-bar-fill{height:100%;border-radius:100px;transition:width 1s cubic-bezier(0.4,0,0.2,1);position:relative;}
.score-bar-fill::after{content:"";position:absolute;top:0;left:0;right:0;bottom:0;background:linear-gradient(90deg,transparent 0%,rgba(255,255,255,0.22) 50%,transparent 100%);animation:barSheen 2.8s ease-in-out infinite;}
@keyframes barSheen{0%{transform:translateX(-100%);}100%{transform:translateX(400%);}}

/* ── TIP BOX ── */
.tip-box{background:linear-gradient(135deg,rgba(184,255,0,0.044),rgba(0,255,213,0.022));border:1px solid rgba(184,255,0,0.10);border-left:3px solid var(--lime-dim);border-radius:0 14px 14px 0;padding:1rem 1.2rem;margin-top:1rem;font-size:0.84rem;color:#aed190;line-height:1.7;}

/* ── STAT CARDS ── */
.stat-card{background:var(--glass-2);border:1px solid var(--border);border-radius:18px;padding:1.4rem 1rem;text-align:center;position:relative;overflow:hidden;transition:transform 0.25s,box-shadow 0.25s,border-color 0.25s;backdrop-filter:blur(18px);}
.stat-card::before{content:"";position:absolute;top:0;left:20%;right:20%;height:1px;background:linear-gradient(90deg,transparent,rgba(184,255,0,0.22),transparent);}
.stat-card:hover{transform:translateY(-4px);box-shadow:0 16px 44px rgba(0,0,0,0.45),0 0 20px rgba(184,255,0,0.055);border-color:var(--border-mid);}
.stat-number{font-family:"Syne",sans-serif;font-size:2.6rem;font-weight:800;line-height:1;background:linear-gradient(135deg,var(--lime),var(--teal));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.stat-label{font-size:0.62rem;color:var(--text-3);letter-spacing:0.18em;text-transform:uppercase;margin-top:0.5rem;font-weight:700;font-family:"Space Mono",monospace;}

/* ── HISTORY ROWS ── */
.history-row{background:rgba(4,9,4,0.85);border:1px solid rgba(184,255,0,0.055);border-radius:12px;padding:0.75rem 1rem;margin:0.25rem 0;display:flex;justify-content:space-between;align-items:center;gap:0.75rem;transition:border-color 0.2s,background 0.2s,transform 0.15s;}
.history-row:hover{border-color:rgba(184,255,0,0.14);background:rgba(8,16,8,0.95);transform:translateX(4px);}

/* ── UPLOAD ZONE ── */
.upload-hint{text-align:center;color:var(--text-3);font-size:0.72rem;margin:-0.2rem 0 0.9rem;letter-spacing:0.07em;font-family:"Space Mono",monospace;}
[data-testid="stFileUploadDropzone"]{background:rgba(4,9,4,0.85)!important;border:2px dashed rgba(184,255,0,0.13)!important;border-radius:18px!important;transition:border-color 0.25s,background 0.25s,box-shadow 0.25s!important;min-height:110px!important;}
[data-testid="stFileUploadDropzone"]:hover{background:rgba(184,255,0,0.022)!important;border-color:rgba(184,255,0,0.3)!important;box-shadow:0 0 30px rgba(184,255,0,0.065)!important;}
[data-testid="stFileUploadDropzone"] p{color:var(--text-3)!important;font-family:"Space Mono",monospace!important;font-size:0.76rem!important;}
[data-testid="stCameraInput"]{border-radius:18px!important;overflow:hidden;border:1px solid rgba(184,255,0,0.10)!important;}

/* ── BADGES ── */
.badge-card{background:rgba(4,9,4,0.92);border:1px solid rgba(184,255,0,0.065);border-radius:18px;padding:1.3rem 0.9rem;text-align:center;transition:transform 0.25s,box-shadow 0.3s,border-color 0.25s;position:relative;overflow:hidden;}
.badge-card::before{content:"";position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(184,255,0,0.11),transparent);}
.badge-card:hover{transform:translateY(-5px);}
.badge-card.unlocked{border-color:rgba(184,255,0,0.24);box-shadow:0 0 35px rgba(184,255,0,0.075),0 0 70px rgba(184,255,0,0.03);}
.badge-card.locked{opacity:0.24;filter:grayscale(1) blur(0.4px);}
.badge-icon{font-size:2.5rem;line-height:1;display:block;}
.badge-name{font-family:"Space Mono",monospace;font-size:0.62rem;font-weight:700;color:var(--lime);margin-top:0.6rem;letter-spacing:0.07em;text-transform:uppercase;}
.badge-desc{font-size:0.64rem;color:var(--text-2);margin-top:0.3rem;line-height:1.45;}

/* ── XP BAR ── */
.points-bar{background:rgba(184,255,0,0.045);border-radius:100px;height:5px;overflow:visible;margin:0.5rem 0;border:1px solid rgba(184,255,0,0.065);position:relative;}
.points-fill{background:linear-gradient(90deg,#6aad00,var(--lime),#d4ff66);height:100%;border-radius:100px;transition:width 1s cubic-bezier(0.4,0,0.2,1);box-shadow:0 0 12px rgba(184,255,0,0.5);position:relative;}
.points-fill::after{content:"";position:absolute;right:-1px;top:-2px;width:9px;height:9px;border-radius:50%;background:white;box-shadow:0 0 8px var(--lime),0 0 16px var(--lime);animation:xpPulse 1.8s ease-in-out infinite;}
@keyframes xpPulse{0%,100%{transform:scale(1);}50%{transform:scale(1.35);}}

/* ── LEVEL CARD ── */
.level-card{background:linear-gradient(160deg,rgba(3,9,3,0.98),rgba(8,16,8,0.99));border:1px solid rgba(184,255,0,0.17);border-radius:22px;padding:2.2rem 2rem;text-align:center;box-shadow:0 0 80px rgba(184,255,0,0.055),0 28px 80px rgba(0,0,0,0.75);position:relative;overflow:hidden;}
.level-card::before{content:"";position:absolute;top:-60%;left:-30%;width:160%;height:160%;background:radial-gradient(ellipse at 50% 0%,rgba(184,255,0,0.065) 0%,transparent 55%);pointer-events:none;}
.level-title{font-family:"Syne",sans-serif;font-size:2.5rem;font-weight:800;letter-spacing:-0.04em;background:linear-gradient(135deg,var(--lime),var(--teal));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.points-display{font-size:0.88rem;color:var(--text-2);font-weight:500;}

/* ── RECYCLE BTN ── */
.recycle-btn{display:inline-flex;align-items:center;gap:0.45rem;background:linear-gradient(135deg,rgba(184,255,0,0.09),rgba(184,255,0,0.04));color:var(--lime)!important;font-family:"Space Mono",monospace;font-size:0.72rem;font-weight:700;padding:0.55rem 1.2rem;border-radius:100px;border:1px solid rgba(184,255,0,0.24);text-decoration:none!important;letter-spacing:0.04em;transition:all 0.22s ease;white-space:nowrap;}
.recycle-btn:hover{background:linear-gradient(135deg,rgba(184,255,0,0.18),rgba(184,255,0,0.09));box-shadow:0 0 28px rgba(184,255,0,0.22);color:#fff!important;transform:translateY(-2px);}

/* ── IMPACT CARDS ── */
.impact-card{background:var(--glass);border:1px solid var(--border);border-radius:18px;padding:1.5rem 1.2rem;text-align:center;position:relative;overflow:hidden;transition:transform 0.25s,box-shadow 0.25s;backdrop-filter:blur(18px);}
.impact-card:hover{transform:translateY(-5px);box-shadow:0 18px 55px rgba(0,0,0,0.5);}
.impact-card::before{content:"";position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--lime),transparent);}
.impact-number{font-family:"Syne",sans-serif;font-size:2.5rem;font-weight:800;line-height:1;background:linear-gradient(135deg,var(--lime),var(--teal));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:0.25rem;}
.impact-unit{font-size:0.72rem;color:var(--text-2);letter-spacing:0.09em;font-weight:700;}
.impact-label{font-size:0.66rem;color:var(--text-3);margin-top:0.35rem;letter-spacing:0.05em;font-family:"Space Mono",monospace;}
.impact-icon{font-size:2rem;margin-bottom:0.5rem;display:block;}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"]{gap:2px!important;background:rgba(2,6,2,0.97)!important;padding:4px!important;border-radius:14px!important;border:1px solid rgba(184,255,0,0.06)!important;box-shadow:0 6px 24px rgba(0,0,0,0.5),inset 0 1px 0 rgba(184,255,0,0.035)!important;flex-wrap:nowrap!important;overflow-x:auto!important;}
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar{height:0!important;}
.stTabs [data-baseweb="tab"]{border-radius:10px!important;font-family:"Plus Jakarta Sans",sans-serif!important;font-size:0.73rem!important;font-weight:700!important;letter-spacing:0.01em!important;color:var(--text-3)!important;padding:0.38rem 0.8rem!important;transition:color 0.2s,background 0.2s!important;white-space:nowrap!important;}
.stTabs [aria-selected="true"]{background:rgba(184,255,0,0.085)!important;color:var(--lime)!important;box-shadow:0 0 18px rgba(184,255,0,0.1),inset 0 1px 0 rgba(184,255,0,0.13)!important;}

/* ── BUTTONS ── */
.stButton>button{background:linear-gradient(135deg,rgba(184,255,0,0.08),rgba(184,255,0,0.03))!important;border:1px solid rgba(184,255,0,0.18)!important;color:var(--lime)!important;border-radius:11px!important;font-family:"Plus Jakarta Sans",sans-serif!important;font-weight:700!important;font-size:0.83rem!important;letter-spacing:0.02em!important;transition:all 0.22s ease!important;position:relative;overflow:hidden;padding:0.5rem 1.1rem!important;}
.stButton>button::after{content:"";position:absolute;top:0;left:-110%;width:60%;height:100%;background:linear-gradient(90deg,transparent,rgba(184,255,0,0.07),transparent);transform:skewX(-20deg);transition:left 0.5s ease;}
.stButton>button:hover::after{left:160%;}
.stButton>button:hover{background:linear-gradient(135deg,rgba(184,255,0,0.14),rgba(184,255,0,0.065))!important;box-shadow:0 0 28px rgba(184,255,0,0.13),0 5px 18px rgba(0,0,0,0.35)!important;border-color:rgba(184,255,0,0.33)!important;transform:translateY(-1px);}
.stButton>button:active{transform:translateY(0)!important;}

/* ── INPUTS ── */
.stTextInput>div>div>input,.stTextArea textarea{background:rgba(2,7,2,0.95)!important;border:1px solid rgba(184,255,0,0.10)!important;border-radius:11px!important;color:var(--text)!important;font-family:"Plus Jakarta Sans",sans-serif!important;font-size:0.85rem!important;transition:all 0.22s!important;padding:0.6rem 0.9rem!important;}
.stTextInput>div>div>input:focus,.stTextArea textarea:focus{border-color:rgba(184,255,0,0.30)!important;box-shadow:0 0 0 3px rgba(184,255,0,0.05),0 0 22px rgba(184,255,0,0.065)!important;outline:none!important;}
.stTextInput>div>div>input::placeholder{color:var(--text-3)!important;opacity:1!important;}
.stTextInput label,.stTextArea label,.stSelectbox label{color:var(--text-2)!important;font-size:0.77rem!important;font-weight:600!important;}
.stSelectbox>div>div{background:rgba(2,7,2,0.95)!important;border:1px solid rgba(184,255,0,0.10)!important;border-radius:11px!important;color:var(--text)!important;font-size:0.85rem!important;}
.stRadio>div{gap:0.4rem!important;}
.stRadio [data-testid="stMarkdownContainer"] p{font-size:0.81rem!important;color:var(--text-2)!important;}
.stToggle>label{color:var(--text-2)!important;font-size:0.8rem!important;font-weight:600!important;}

/* ── ALERTS & SPINNER ── */
.stSpinner>div{border-top-color:var(--lime)!important;}
.stAlert{border-radius:13px!important;border:1px solid rgba(184,255,0,0.10)!important;background:rgba(2,8,2,0.96)!important;backdrop-filter:blur(14px)!important;}

/* ── SUGGEST CARDS ── */
.suggest-card{background:linear-gradient(145deg,rgba(8,16,8,0.99),rgba(4,10,4,0.99));border:1px solid rgba(184,255,0,0.095);border-radius:16px;padding:1.2rem 1.4rem;margin:0.5rem 0;position:relative;overflow:hidden;transition:border-color 0.25s,transform 0.22s,box-shadow 0.25s;}
.suggest-card::before{content:"";position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(184,255,0,0.19),transparent);}
.suggest-card:hover{border-color:rgba(184,255,0,0.20);transform:translateY(-2px);box-shadow:0 8px 30px rgba(0,0,0,0.4),0 0 20px rgba(184,255,0,0.025);}
.suggest-tag{display:inline-flex;align-items:center;gap:0.3rem;font-family:"Space Mono",monospace;font-size:0.58rem;font-weight:700;padding:0.2rem 0.75rem;border-radius:100px;border:1px solid currentColor;letter-spacing:0.09em;margin-bottom:0.65rem;text-transform:uppercase;}

/* ── CHAT ── */
.chat-bubble-user{background:rgba(184,255,0,0.062);border:1px solid rgba(184,255,0,0.14);border-radius:18px 18px 4px 18px;padding:0.85rem 1.15rem;margin:0.5rem 0 0.5rem auto;max-width:74%;font-size:0.85rem;color:var(--text);line-height:1.62;animation:bubbleIn 0.3s cubic-bezier(0.16,1,0.3,1);}
.chat-bubble-ai{background:rgba(8,16,8,0.97);border:1px solid rgba(184,255,0,0.078);border-radius:18px 18px 18px 4px;padding:0.85rem 1.15rem;margin:0.5rem auto 0.5rem 0;max-width:84%;font-size:0.85rem;color:var(--text);line-height:1.65;animation:bubbleIn 0.3s cubic-bezier(0.16,1,0.3,1);}
.chat-bubble-ai::before{content:"🤖 ";font-size:0.86rem;}
@keyframes bubbleIn{from{opacity:0;transform:translateY(10px) scale(0.97);}to{opacity:1;transform:translateY(0) scale(1);}}

/* ── SCROLLBAR ── */
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-track{background:var(--surface);}
::-webkit-scrollbar-thumb{background:rgba(184,255,0,0.13);border-radius:4px;}
::-webkit-scrollbar-thumb:hover{background:rgba(184,255,0,0.26);}

/* ── FOOTER ── */
.footer{text-align:center;color:var(--text-3);font-family:"Space Mono",monospace;font-size:0.6rem;letter-spacing:0.18em;margin-top:6rem;padding-top:1.5rem;border-top:1px solid rgba(184,255,0,0.04);}

/* ── UTILITIES ── */
.section-hdr{font-family:"Syne",sans-serif;font-size:0.78rem;font-weight:800;color:var(--lime);letter-spacing:0.05em;text-transform:uppercase;display:flex;align-items:center;gap:0.6rem;margin-bottom:0.85rem;}
.section-hdr::after{content:"";flex:1;height:1px;background:linear-gradient(90deg,rgba(184,255,0,0.13),transparent);}
.nearby-title{font-family:"Syne",sans-serif;font-size:0.9rem;font-weight:700;color:var(--text);}
.nearby-label{font-size:0.74rem;color:var(--text-2);}
.js-plotly-plot .plotly,.js-plotly-plot .bg{background:transparent!important;}
.env-badge{display:none;}

@keyframes glowPulse{0%,100%{box-shadow:0 0 20px rgba(184,255,0,0.1);}50%{box-shadow:0 0 45px rgba(184,255,0,0.22);}}
@keyframes fadeUp{from{opacity:0;transform:translateY(14px);}to{opacity:1;transform:translateY(0);}}

/* ── FONT SCALE STANDARDISATION ── */
:root { --fs-label:0.62rem; --fs-small:0.72rem; --fs-body:0.85rem; }

/* ── SPLASH SCREEN ── */
#sw-splash{position:fixed;inset:0;z-index:99999;background:radial-gradient(ellipse 80% 60% at 50% 40%,#0a1a0a 0%,#020602 100%);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:1.2rem;animation:splashOut 0.5s ease 1.8s forwards;pointer-events:none;}
#sw-splash .sp-icon{font-size:4rem;animation:splashBounce 0.7s cubic-bezier(0.16,1,0.3,1) both;}
#sw-splash .sp-title{font-family:"Syne",sans-serif;font-size:2rem;font-weight:800;letter-spacing:-0.04em;background:linear-gradient(130deg,#b8ff00,#00ffd5);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;animation:splashBounce 0.7s cubic-bezier(0.16,1,0.3,1) 0.1s both;}
#sw-splash .sp-sub{font-family:"Space Mono",monospace;font-size:0.62rem;color:#3a5628;letter-spacing:0.2em;text-transform:uppercase;animation:splashBounce 0.7s cubic-bezier(0.16,1,0.3,1) 0.2s both;}
#sw-splash .sp-bar-wrap{width:160px;height:2px;background:rgba(184,255,0,0.08);border-radius:2px;overflow:hidden;animation:splashBounce 0.7s cubic-bezier(0.16,1,0.3,1) 0.3s both;}
#sw-splash .sp-bar{height:100%;background:linear-gradient(90deg,#b8ff00,#00ffd5);animation:splashBarFill 1.4s ease 0.4s both;}
@keyframes splashBarFill{from{width:0%;}to{width:100%;}}
@keyframes splashBounce{from{opacity:0;transform:translateY(20px);}to{opacity:1;transform:translateY(0);}}
@keyframes splashOut{to{opacity:0;transform:scale(1.03);visibility:hidden;}}

/* ── MOBILE RESPONSIVE ── */
@media(max-width:640px){
  [data-testid="column"]{width:100%!important;flex:0 0 100%!important;}
  .stTabs [data-baseweb="tab"]{font-size:0.58rem!important;padding:0.3rem 0.45rem!important;}
  .hero-title{font-size:clamp(1.6rem,7vw,2.5rem)!important;}
  .stat-card{min-height:auto!important;}
  .impact-card{min-height:auto!important;}
}

/* ── STAT CARDS — uniform height ── */
.stat-card{background:var(--glass);border:1px solid var(--border);border-radius:16px;padding:1.2rem 1rem;text-align:center;backdrop-filter:blur(18px);min-height:110px;display:flex;flex-direction:column;align-items:center;justify-content:center;transition:border-color 0.3s,transform 0.25s;}
.stat-card:hover{border-color:var(--border-mid);transform:translateY(-2px);}
.stat-number{font-family:"Syne",sans-serif;font-size:2.2rem;font-weight:800;line-height:1;background:linear-gradient(135deg,var(--lime),var(--teal));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;animation:countUp 0.6s cubic-bezier(0.16,1,0.3,1) both;}
.stat-label{font-size:0.62rem;color:var(--text-3);letter-spacing:0.08em;margin-top:0.3rem;font-weight:700;}
@keyframes countUp{from{opacity:0;transform:translateY(8px);}to{opacity:1;transform:translateY(0);}}

/* ── SKELETON LOADER ── */
.skeleton{background:linear-gradient(90deg,rgba(184,255,0,0.04) 25%,rgba(184,255,0,0.09) 50%,rgba(184,255,0,0.04) 75%);background-size:200% 100%;animation:skelShimmer 1.4s ease infinite;border-radius:8px;}
@keyframes skelShimmer{0%{background-position:200% 0;}100%{background-position:-200% 0;}}
.skeleton-card{height:110px;border-radius:16px;margin:0.4rem 0;}
.skeleton-line{height:12px;margin:0.5rem 0;border-radius:6px;}
.skeleton-line.short{width:60%;}

/* ── EMPTY STATE ── */
.empty-state{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:4rem 2rem;text-align:center;border:1px dashed var(--border);border-radius:var(--r);background:rgba(8,14,8,0.5);margin:1rem 0;}
.empty-state-icon{font-size:3.5rem;margin-bottom:1rem;opacity:0.6;animation:emptyFloat 3s ease-in-out infinite alternate;}
.empty-state-title{font-family:"Syne",sans-serif;font-size:1.1rem;font-weight:800;color:var(--text-2);margin-bottom:0.5rem;}
.empty-state-sub{font-size:0.72rem;color:var(--text-3);line-height:1.6;max-width:320px;}
@keyframes emptyFloat{from{transform:translateY(0);}to{transform:translateY(-8px);}}
body.light-mode .empty-state{background:rgba(240,248,234,0.7)!important;border-color:rgba(46,125,0,0.10)!important;}

/* ── RESULT RING GLOW PULSE (post-scan) ── */
.conf-ring-fill-anim{animation:ringGlow 1.2s ease 0.55s 3;}
@keyframes ringGlow{0%,100%{filter:drop-shadow(0 0 4px currentColor);}50%{filter:drop-shadow(0 0 18px currentColor) drop-shadow(0 0 32px currentColor);}}

/* ── CUSTOM CALENDAR TOOLTIP ── */
.cal-cell{position:relative;cursor:default;}
.cal-cell:hover::after{content:attr(data-tip);position:absolute;bottom:calc(100% + 6px);left:50%;transform:translateX(-50%);background:rgba(4,12,4,0.97);border:1px solid rgba(184,255,0,0.22);color:#edfce4;font-size:0.58rem;font-family:"Space Mono",monospace;padding:0.3rem 0.6rem;border-radius:7px;white-space:nowrap;z-index:100;pointer-events:none;}

/* ── STREAK CALENDAR ── */
.cal-grid{display:grid;grid-template-columns:repeat(53,1fr);gap:3px;margin:0.5rem 0;}
.cal-cell{width:100%;aspect-ratio:1;border-radius:3px;transition:transform 0.15s;}
.cal-cell:hover{transform:scale(1.5);z-index:10;}
.cal-month-labels{display:grid;grid-template-columns:repeat(53,1fr);gap:3px;margin-bottom:2px;}
.cal-month-label{font-size:0.52rem;color:var(--text-3);font-family:"Space Mono",monospace;text-align:left;overflow:hidden;}
.cal-day-labels{display:flex;flex-direction:column;gap:3px;margin-right:4px;padding-top:2px;}
.cal-day-label{font-size:0.52rem;color:var(--text-3);font-family:"Space Mono",monospace;height:11px;line-height:11px;}
.cal-wrap{overflow-x:auto;padding-bottom:0.5rem;}

/* ── BATCH SCAN ── */
.batch-thumb{border-radius:10px;overflow:hidden;border:1px solid var(--border);transition:transform 0.2s;}
.batch-thumb:hover{transform:scale(1.03);}
.batch-result-badge{display:inline-flex;align-items:center;gap:0.3rem;font-size:0.72rem;font-weight:700;padding:0.2rem 0.7rem;border-radius:100px;border:1px solid currentColor;margin-top:0.3rem;}

/* ── UPLOAD HINT ── */
.upload-hint{font-size:0.72rem;color:var(--text-3);text-align:center;padding:0.5rem 0 1rem;letter-spacing:0.04em;}
</style>
""", unsafe_allow_html=True)

# ── Splash screen (first load only) ──────────────────────────────────────────
if "splash_shown" not in st.session_state:
    st.session_state.splash_shown = True
    st.markdown("""<div id="sw-splash">
        <div class="sp-icon">♻️</div>
        <div class="sp-title">Smart Waste AI</div>
        <div class="sp-sub">Powered by Gemini · Built with Streamlit</div>
        <div class="sp-bar-wrap"><div class="sp-bar"></div></div>
    </div>""", unsafe_allow_html=True)

# ── Theme injection ───────────────────────────────────────────────────────────
def inject_theme(light: bool):
    """Inject JS to toggle light-mode class on body."""
    cls = "light-mode" if light else ""
    st.markdown(f"""<script>
        (function(){{
            var b = window.parent.document.body;
            if({str(light).lower()}) {{ b.classList.add('light-mode'); }}
            else {{ b.classList.remove('light-mode'); }}
        }})();
    </script>""", unsafe_allow_html=True)

# ── Sound effects (Web Audio API via JS) ─────────────────────────────────────
def play_sound(sound_type: str):
    """
    Inject tiny JS Web Audio API snippet for scan success, badge unlock sounds.
    sound_type: 'scan' | 'badge' | 'error'
    """
    scripts = {
        "scan": """
            var ac=new(window.AudioContext||window.webkitAudioContext)();
            var o=ac.createOscillator(),g=ac.createGain();
            o.connect(g);g.connect(ac.destination);
            o.frequency.setValueAtTime(880,ac.currentTime);
            o.frequency.exponentialRampToValueAtTime(1320,ac.currentTime+0.12);
            g.gain.setValueAtTime(0.18,ac.currentTime);
            g.gain.exponentialRampToValueAtTime(0.001,ac.currentTime+0.28);
            o.start();o.stop(ac.currentTime+0.28);
        """,
        "badge": """
            var ac=new(window.AudioContext||window.webkitAudioContext)();
            [523,659,784,1047].forEach(function(f,i){
                var o=ac.createOscillator(),g=ac.createGain();
                o.connect(g);g.connect(ac.destination);
                o.frequency.value=f;
                g.gain.setValueAtTime(0,ac.currentTime+i*0.12);
                g.gain.linearRampToValueAtTime(0.15,ac.currentTime+i*0.12+0.04);
                g.gain.exponentialRampToValueAtTime(0.001,ac.currentTime+i*0.12+0.22);
                o.start(ac.currentTime+i*0.12);
                o.stop(ac.currentTime+i*0.12+0.22);
            });
        """,
        "error": """
            var ac=new(window.AudioContext||window.webkitAudioContext)();
            var o=ac.createOscillator(),g=ac.createGain();
            o.connect(g);g.connect(ac.destination);
            o.type='sawtooth';o.frequency.value=220;
            g.gain.setValueAtTime(0.12,ac.currentTime);
            g.gain.exponentialRampToValueAtTime(0.001,ac.currentTime+0.3);
            o.start();o.stop(ac.currentTime+0.3);
        """,
    }
    js = scripts.get(sound_type, "")
    if js:
        st.markdown(f"<script>try{{{js}}}catch(e){{}}</script>", unsafe_allow_html=True)

# ── Streak calendar builder ───────────────────────────────────────────────────
def build_streak_calendar(display_name: str) -> str:
    """Return HTML for a GitHub-style 52-week scan calendar."""
    uid = _get_user_id(display_name)
    scan_dates = set()
    if uid:
        with _get_conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT scanned_date FROM scan_history WHERE user_id=?", (uid,)
            ).fetchall()
        scan_dates = {r[0] for r in rows}

    today = datetime.date.today()
    # Start from 52 weeks ago, aligned to Monday
    start = today - datetime.timedelta(weeks=52)
    start -= datetime.timedelta(days=start.weekday())  # back to Monday

    weeks = []
    cur = start
    while cur <= today:
        week = []
        for _ in range(7):
            week.append(cur)
            cur += datetime.timedelta(days=1)
        weeks.append(week)

    # Month labels row
    month_labels = []
    prev_month = None
    for week in weeks:
        month = week[0].strftime("%b") if week[0].month != prev_month else ""
        prev_month = week[0].month
        month_labels.append(f'<div class="cal-month-label">{month}</div>')

    # Day labels
    day_labels = "".join(
        f'<div class="cal-day-label">{d}</div>'
        for d in ["M","","W","","F","",""]
    )

    # Grid cells
    cells = []
    for week in weeks:
        for day in week:
            ds = day.isoformat()
            if day > today:
                color = "rgba(184,255,0,0.04)"
                tip = ""
            elif ds in scan_dates:
                color = "#b8ff00"
                tip = f"♻ Scanned on {ds}"
            else:
                color = "rgba(184,255,0,0.07)"
                tip = f"No scan · {ds}"
            cells.append(
                f'<div class="cal-cell" style="background:{color};" data-tip="{tip}"></div>'
            )

    month_row = "".join(month_labels)
    cell_grid  = "".join(cells)

    return f"""
    <div class="cal-wrap">
      <div style="display:flex;align-items:flex-start;gap:0;">
        <div>
          <div style="height:14px;"></div>
          <div class="cal-day-labels">{day_labels}</div>
        </div>
        <div style="flex:1;min-width:0;">
          <div class="cal-month-labels">{month_row}</div>
          <div class="cal-grid">{cell_grid}</div>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:0.4rem;margin-top:0.4rem;font-size:0.65rem;color:var(--text-3);font-family:'Space Mono',monospace;">
        <span>Less</span>
        <div style="width:10px;height:10px;border-radius:2px;background:rgba(184,255,0,0.07);"></div>
        <div style="width:10px;height:10px;border-radius:2px;background:rgba(184,255,0,0.3);"></div>
        <div style="width:10px;height:10px;border-radius:2px;background:#b8ff00;"></div>
        <span>More</span>
      </div>
    </div>
    """

# ── SQLite Database ───────────────────────────────────────────────────────────
import sqlite3

DB_FILE = "smart_waste.db"

def _get_conn():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def _init_db():
    """Create tables and run migrations safely."""
    conn = _get_conn()

    # ── Step 1: core tables that don't depend on new columns ──────────────────
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    UNIQUE NOT NULL COLLATE NOCASE,
            display_name  TEXT    NOT NULL,
            email         TEXT    UNIQUE NOT NULL COLLATE NOCASE,
            password_hash TEXT    NOT NULL,
            created_at    TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS user_stats (
            user_id     INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
            points      INTEGER NOT NULL DEFAULT 0,
            streak      INTEGER NOT NULL DEFAULT 0,
            last_scan   TEXT,
            total_scans INTEGER NOT NULL DEFAULT 0,
            badges      TEXT    NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS scan_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            scanned_at  TEXT    NOT NULL,
            category    TEXT    NOT NULL,
            confidence  INTEGER NOT NULL,
            reason      TEXT,
            icon        TEXT,
            source      TEXT    NOT NULL DEFAULT 'gemini'
        );

        CREATE INDEX IF NOT EXISTS idx_scan_user ON scan_history(user_id);
    """)

    # ── Step 2: migrations — add columns / tables that may be missing ─────────
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(scan_history)")}
    if "scanned_date" not in existing_cols:
        # SQLite ALTER TABLE cannot use function calls as defaults —
        # add the column with a plain string default, then backfill.
        conn.execute(
            "ALTER TABLE scan_history ADD COLUMN scanned_date TEXT NOT NULL DEFAULT '1970-01-01'"
        )
        today = datetime.date.today().isoformat()
        conn.execute(
            "UPDATE scan_history SET scanned_date=? WHERE scanned_date='1970-01-01'", (today,)
        )
        conn.commit()

    # Now it's safe to create the index on scanned_date
    conn.execute("CREATE INDEX IF NOT EXISTS idx_scan_date ON scan_history(scanned_date)")
    conn.commit()

    existing_tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    if "email_settings" not in existing_tables:
        conn.execute("""
            CREATE TABLE email_settings (
                user_id       INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                smtp_host     TEXT    NOT NULL DEFAULT '',
                smtp_port     INTEGER NOT NULL DEFAULT 587,
                smtp_user     TEXT    NOT NULL DEFAULT '',
                smtp_pass     TEXT    NOT NULL DEFAULT '',
                notify_streak INTEGER NOT NULL DEFAULT 1,
                last_reminded TEXT
            )
        """)
        conn.commit()
    if "admin_config" not in existing_tables:
        conn.execute("""
            CREATE TABLE admin_config (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        conn.commit()
    conn.close()

_init_db()

def _hash_pw(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# ── Auth helpers ──────────────────────────────────────────────────────────────

def register_user(username: str, email: str, password: str) -> tuple[bool, str]:
    if not username.strip() or len(username.strip()) < 3:
        return False, "Username must be at least 3 characters."
    if not re.match(r"^[^@]+@[^@]+\.[^@]+$", email.strip()):
        return False, "Please enter a valid email address."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    try:
        with _get_conn() as conn:
            cur = conn.execute(
                "INSERT INTO users (username, display_name, email, password_hash, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (username.strip().lower(), username.strip(),
                 email.strip(), _hash_pw(password),
                 datetime.datetime.now().isoformat())
            )
            user_id = cur.lastrowid
            conn.execute("INSERT INTO user_stats (user_id) VALUES (?)", (user_id,))
        return True, "Account created! You can now log in."
    except sqlite3.IntegrityError as e:
        if "username" in str(e).lower():
            return False, "That username is already taken. Please choose another."
        if "email" in str(e).lower():
            return False, "An account with that email already exists."
        return False, "Registration failed. Please try again."

def authenticate_user(username_or_email: str, password: str) -> tuple[bool, str, str]:
    """Returns (success, display_name, error_msg)."""
    val = username_or_email.strip()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT id, display_name, password_hash FROM users "
            "WHERE username = ? OR email = ? COLLATE NOCASE",
            (val, val)
        ).fetchone()
    if not row:
        return False, "", "Invalid username / email or password."
    if row["password_hash"] != _hash_pw(password):
        return False, "", "Invalid username / email or password."
    return True, row["display_name"], ""

# ── Per-user stats persistence ────────────────────────────────────────────────

def _get_user_id(display_name: str) -> int | None:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM users WHERE display_name = ?", (display_name,)
        ).fetchone()
    return row["id"] if row else None

def load_user_stats(display_name: str) -> dict:
    """Load points, streak, badges, history from DB into session_state."""
    uid = _get_user_id(display_name)
    if not uid:
        return {}
    with _get_conn() as conn:
        stats = conn.execute(
            "SELECT points, streak, last_scan, total_scans, badges "
            "FROM user_stats WHERE user_id = ?", (uid,)
        ).fetchone()
        rows = conn.execute(
            "SELECT scanned_at, category, confidence, reason, icon, source "
            "FROM scan_history WHERE user_id = ? ORDER BY id ASC", (uid,)
        ).fetchall()

    badges_set = set(stats["badges"].split(",")) - {""} if stats["badges"] else set()
    last_scan = (datetime.date.fromisoformat(stats["last_scan"])
                 if stats["last_scan"] else None)
    history = [
        {
            "time": r["scanned_at"], "category": r["category"],
            "confidence": r["confidence"], "reason": r["reason"] or "",
            "icon": r["icon"] or "♻️", "source": r["source"],
        }
        for r in rows
    ]
    return {
        "points": stats["points"],
        "streak": stats["streak"],
        "last_scan": last_scan,
        "total_scans": stats["total_scans"],
        "badges": badges_set,
        "history": history,
    }

def save_user_stats():
    """Persist current session_state stats back to DB."""
    uid = _get_user_id(st.session_state.get("auth_username", ""))
    if not uid:
        return
    badges_str = ",".join(st.session_state.badges)
    last_scan_str = (st.session_state.last_scan.isoformat()
                     if st.session_state.last_scan else None)
    with _get_conn() as conn:
        conn.execute(
            "UPDATE user_stats SET points=?, streak=?, last_scan=?, "
            "total_scans=?, badges=? WHERE user_id=?",
            (st.session_state.points, st.session_state.streak,
             last_scan_str, st.session_state.total_scans,
             badges_str, uid)
        )

def save_scan_to_db(entry: dict):
    """Append one scan record to the DB."""
    uid = _get_user_id(st.session_state.get("auth_username", ""))
    if not uid:
        return
    today = datetime.date.today().isoformat()
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO scan_history (user_id, scanned_at, scanned_date, category, confidence, reason, icon, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (uid, entry["time"], today, entry["category"], entry["confidence"],
             entry.get("reason", ""), entry.get("icon", "♻️"),
             entry.get("source", "gemini"))
        )

# ── Login / Register wall ─────────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "auth_username" not in st.session_state:
    st.session_state.auth_username = ""
if "light_mode" not in st.session_state:
    st.session_state.light_mode = False
if "sounds_on" not in st.session_state:
    st.session_state.sounds_on = True

inject_theme(st.session_state.light_mode)

if not st.session_state.authenticated:
    # ── Login page CSS ─────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Space+Mono:wght@400;700&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
    :root{--lime:#b8ff00;--teal:#00ffd5;--surface:#040804;--glass:rgba(8,14,8,0.88);--border:rgba(184,255,0,0.12);--text:#edfce4;--text-2:#84a86c;--text-3:#3a5628;}
    html,body,[class*="css"]{font-family:"Plus Jakarta Sans",sans-serif!important;color:var(--text)!important;}
    .stApp{background:var(--surface)!important;min-height:100vh!important;}
    .stApp::before{content:"";position:fixed;inset:0;z-index:0;pointer-events:none;
      background:radial-gradient(ellipse 80% 60% at -5% -5%,rgba(184,255,0,0.09) 0%,transparent 55%),
        radial-gradient(ellipse 60% 45% at 105% 105%,rgba(0,255,213,0.07) 0%,transparent 50%);}
    .auth-wrap{max-width:420px;margin:6vh auto 0;padding:0 1rem;}
    .auth-logo{text-align:center;margin-bottom:1.8rem;}
    .auth-logo-icon{font-size:2.4rem;display:block;margin-bottom:0.4rem;}
    .auth-logo-title{font-family:"Syne",sans-serif;font-size:1.6rem;font-weight:800;
      background:linear-gradient(130deg,#b8ff00,#00ffd5);-webkit-background-clip:text;
      -webkit-text-fill-color:transparent;background-clip:text;letter-spacing:-0.03em;}
    .auth-logo-sub{font-family:"Space Mono",monospace;font-size:0.58rem;color:var(--text-3);
      letter-spacing:0.16em;text-transform:uppercase;margin-top:0.2rem;}
    .auth-card{background:var(--glass);border:1px solid var(--border);border-radius:20px;
      padding:2rem 2rem 1.5rem;backdrop-filter:blur(28px);
      box-shadow:0 8px 60px rgba(0,0,0,0.6),inset 0 1px 0 rgba(184,255,0,0.055);}
    .auth-tab-row{display:flex;gap:0.5rem;margin-bottom:1.5rem;}
    .auth-tab{flex:1;padding:0.5rem;border-radius:10px;text-align:center;font-weight:700;
      font-size:0.82rem;cursor:pointer;border:1px solid rgba(184,255,0,0.14);
      background:transparent;color:var(--text-2);transition:all 0.2s;}
    .auth-tab.active{background:rgba(184,255,0,0.09);border-color:rgba(184,255,0,0.35);color:var(--lime);}
    .auth-heading{font-family:"Syne",sans-serif;font-size:1.1rem;font-weight:800;
      color:var(--lime);margin-bottom:0.3rem;}
    .auth-sub{font-size:0.78rem;color:var(--text-2);margin-bottom:1.2rem;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="auth-wrap">', unsafe_allow_html=True)
    st.markdown("""
    <div class="auth-logo">
        <span class="auth-logo-icon">♻</span>
        <div class="auth-logo-title">Smart Waste AI</div>
        <div class="auth-logo-sub">Powered by Gemini · Free · Open</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="auth-card">', unsafe_allow_html=True)

    auth_mode = st.radio("", ["🔐 Login", "📝 Register"], horizontal=True, label_visibility="collapsed")

    if auth_mode == "🔐 Login":
        st.markdown('<div class="auth-heading">Welcome back!</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-sub">Sign in to access your recycling dashboard.</div>', unsafe_allow_html=True)
        login_id = st.text_input("Username or Email", placeholder="you@email.com or username", key="login_id")
        login_pw = st.text_input("Password", type="password", placeholder="••••••••", key="login_pw")
        if st.button("🔓 Sign In", use_container_width=True):
            if not login_id.strip() or not login_pw.strip():
                st.error("Please fill in all fields.")
            else:
                ok, display_name, err = authenticate_user(login_id, login_pw)
                if ok:
                    st.session_state.authenticated = True
                    st.session_state.auth_username = display_name
                    # Load persisted stats from DB
                    saved = load_user_stats(display_name)
                    for k, v in saved.items():
                        st.session_state[k] = v
                    st.success(f"Welcome back, {display_name}! 🌿")
                    st.rerun()
                else:
                    st.error(err)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;font-size:0.75rem;color:var(--text-3);">Don\'t have an account? Switch to <strong style="color:var(--lime);">Register</strong> above.</div>', unsafe_allow_html=True)

    else:  # Register
        st.markdown('<div class="auth-heading">Create your account</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-sub">Join thousands of eco-warriors tracking their impact.</div>', unsafe_allow_html=True)
        reg_name  = st.text_input("Username", placeholder="EcoWarrior_You", key="reg_name")
        reg_email = st.text_input("Email", placeholder="you@email.com", key="reg_email")
        reg_pw    = st.text_input("Password", type="password", placeholder="Min 6 characters", key="reg_pw")
        reg_pw2   = st.text_input("Confirm Password", type="password", placeholder="Repeat password", key="reg_pw2")
        if st.button("🌱 Create Account", use_container_width=True):
            if not reg_name.strip() or not reg_email.strip() or not reg_pw.strip() or not reg_pw2.strip():
                st.error("Please fill in all fields.")
            elif reg_pw != reg_pw2:
                st.error("Passwords do not match.")
            else:
                ok, msg = register_user(reg_name, reg_email, reg_pw)
                if ok:
                    st.success(msg)
                    st.info("Switch to Login to sign in.")
                else:
                    st.error(msg)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;font-size:0.75rem;color:var(--text-3);">Already have an account? Switch to <strong style="color:var(--lime);">Login</strong> above.</div>', unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)
    st.stop()

# ── Session state ─────────────────────────────────────────────────────────────
if "history"      not in st.session_state: st.session_state.history = []
if "points"       not in st.session_state: st.session_state.points = 0
if "streak"       not in st.session_state: st.session_state.streak = 0
if "last_scan"    not in st.session_state: st.session_state.last_scan = None
if "badges"       not in st.session_state: st.session_state.badges = set()
if "total_scans"  not in st.session_state: st.session_state.total_scans = 0
if "username"     not in st.session_state: st.session_state.username = st.session_state.auth_username or "You"
if "language"     not in st.session_state: st.session_state.language = "English"

# ── Data ──────────────────────────────────────────────────────────────────────
with open("classes.json") as f:
    CLASSES = json.load(f)

# ── Language strings ──────────────────────────────────────────────────────────
LANG = {
    "English": {
        "app_title": "♻ Smart Waste AI",
        "app_sub": "Powered by Gemini Vision · Free · Open",
        "upload_hint": "📸 JPG · PNG · WEBP — Try plastic, paper, metal, food waste, or electronics",
        "webcam_hint": "📷 Point your camera at any waste item and click capture",
        "result_label": "✦ Classification Result",
        "confidence": "confidence",
        "pts_earned": "pts earned",
        "disposal_tip": "💡 Disposal Tip:",
        "breakdown": "📊 Category Breakdown",
        "analysing": "🔍 Analysing waste...",
        "no_api": "⚠️ Enter your Gemini API key in the sidebar to classify.",
        "categories": "Categories:",
        "clear_history": "🗑️ Clear History",
        "config": "⚙️ Configuration",
        "voice": "🔊 Voice Announcements",
        "tab_upload": "📁 Upload",
        "tab_webcam": "📷 Webcam",
        "tab_achieve": "🏆 Achievements",
        "tab_dash": "📊 Dashboard",
        "tab_recycle": "📍 Recycling Centers",
        "tab_leader": "🏅 Leaderboard",
        "tab_report": "📄 Download Report",
        "tab_lang": "🌐 Language",
        "tab_impact": "🌍 Impact",
        "tab_suggest": "🤖 AI Suggestions",
    },
    "Tamil": {
        "app_title": "♻ ஸ்மார்ட் கழிவு AI",
        "app_sub": "Gemini Vision மூலம் இயக்கப்படுகிறது · இலவசம் · திறந்த",
        "upload_hint": "📸 JPG · PNG · WEBP — பிளாஸ்டிக், காகிதம், உலோகம், உணவு கழிவு அல்லது மின்னணுவை முயற்சிக்கவும்",
        "webcam_hint": "📷 கேமராவை கழிவு பொருளை நோக்கி வைத்து கிளிக் செய்யவும்",
        "result_label": "✦ வகைப்பாடு முடிவு",
        "confidence": "நம்பகத்தன்மை",
        "pts_earned": "புள்ளிகள் சம்பாதித்தது",
        "disposal_tip": "💡 அகற்றும் குறிப்பு:",
        "breakdown": "📊 வகை பிரிப்பு",
        "analysing": "🔍 கழிவை பகுப்பாய்கிறது...",
        "no_api": "⚠️ வகைப்படுத்த பக்கப்பட்டியில் Gemini API விசையை உள்ளிடவும்.",
        "categories": "வகைகள்:",
        "clear_history": "🗑️ வரலாற்றை அழிக்கவும்",
        "config": "⚙️ அமைப்பு",
        "voice": "🔊 குரல் அறிவிப்புகள்",
        "tab_upload": "📁 பதிவேற்றம்",
        "tab_webcam": "📷 வெப்கேம்",
        "tab_achieve": "🏆 சாதனைகள்",
        "tab_dash": "📊 டாஷ்போர்டு",
        "tab_recycle": "📍 மறுசுழற்சி மையங்கள்",
        "tab_leader": "🏅 தலைவர் பலகை",
        "tab_report": "📄 அறிக்கை பதிவிறக்கம்",
        "tab_lang": "🌐 மொழி",
        "tab_impact": "🌍 தாக்கம்",
        "tab_suggest": "🤖 AI பரிந்துரைகள்",
    },
    "Hindi": {
        "app_title": "♻ स्मार्ट वेस्ट AI",
        "app_sub": "Gemini Vision द्वारा संचालित · मुफ़्त · खुला",
        "upload_hint": "📸 JPG · PNG · WEBP — प्लास्टिक, कागज़, धातु, खाद्य अपशिष्ट या इलेक्ट्रॉनिक्स आज़माएं",
        "webcam_hint": "📷 कैमरा किसी भी कचरे की वस्तु पर इंगित करें और क्लिक करें",
        "result_label": "✦ वर्गीकरण परिणाम",
        "confidence": "विश्वास",
        "pts_earned": "अंक अर्जित",
        "disposal_tip": "💡 निपटान सुझाव:",
        "breakdown": "📊 श्रेणी विवरण",
        "analysing": "🔍 कचरे का विश्लेषण...",
        "no_api": "⚠️ वर्गीकृत करने के लिए साइडबार में Gemini API कुंजी दर्ज करें।",
        "categories": "श्रेणियाँ:",
        "clear_history": "🗑️ इतिहास साफ़ करें",
        "config": "⚙️ कॉन्फ़िगरेशन",
        "voice": "🔊 वॉयस घोषणाएं",
        "tab_upload": "📁 अपलोड",
        "tab_webcam": "📷 वेबकैम",
        "tab_achieve": "🏆 उपलब्धियां",
        "tab_dash": "📊 डैशबोर्ड",
        "tab_recycle": "📍 रीसाइक्लिंग केंद्र",
        "tab_leader": "🏅 लीडरबोर्ड",
        "tab_report": "📄 रिपोर्ट डाउनलोड",
        "tab_lang": "🌐 भाषा",
        "tab_impact": "🌍 प्रभाव",
        "tab_suggest": "🤖 AI सुझाव",
    },
}

def T(key):
    """Translate a key using the current language."""
    lang = st.session_state.get("language", "English")
    return LANG.get(lang, LANG["English"]).get(key, LANG["English"].get(key, key))

# ── Mock leaderboard data (realistic simulated users) ─────────────────────────
def _badge_for_points(points: int) -> str:
    if points >= 1000: return "🏆"
    if points >= 500:  return "⭐"
    if points >= 200:  return "🌿"
    return "🌱"

def load_leaderboard(limit: int = 20) -> list[dict]:
    """Fetch all real users from DB, sorted by points descending."""
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT u.display_name AS name,
                   s.points, s.total_scans AS scans, s.streak
            FROM user_stats s
            JOIN users u ON u.id = s.user_id
            ORDER BY s.points DESC
            LIMIT ?
            """,
            (limit,)
        ).fetchall()
    return [
        {
            "name":   r["name"],
            "points": r["points"],
            "scans":  r["scans"],
            "streak": r["streak"],
            "badge":  _badge_for_points(r["points"]),
        }
        for r in rows
    ]

# ── Email streak reminder ─────────────────────────────────────────────────────
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_streak_reminder(display_name: str) -> tuple[bool, str]:
    """Send a streak reminder email to the user if SMTP is configured."""
    uid = _get_user_id(display_name)
    if not uid:
        return False, "User not found."
    with _get_conn() as conn:
        es = conn.execute(
            "SELECT smtp_host, smtp_port, smtp_user, smtp_pass, notify_streak, last_reminded "
            "FROM email_settings WHERE user_id=?", (uid,)
        ).fetchone()
        user = conn.execute(
            "SELECT email, display_name FROM users WHERE id=?", (uid,)
        ).fetchone()
        stats = conn.execute(
            "SELECT streak FROM user_stats WHERE user_id=?", (uid,)
        ).fetchone()
    if not es or not es["smtp_host"] or not es["smtp_user"] or not es["smtp_pass"]:
        return False, "SMTP not configured."
    if not es["notify_streak"]:
        return False, "Streak notifications disabled."
    today = datetime.date.today().isoformat()
    if es["last_reminded"] == today:
        return False, "Already reminded today."
    streak = stats["streak"] if stats else 0
    to_email = user["email"]
    name     = user["display_name"]
    subject  = f"🔥 Don't break your {streak}-day streak, {name}!"
    body = f"""Hi {name},

You have a {streak}-day recycling streak on Smart Waste AI — don't let it break today!

Open the app and scan at least one waste item to keep it going:
http://localhost:8501

Keep up the great work! 🌍♻️

— Smart Waste AI Team
"""
    try:
        msg = MIMEMultipart()
        msg["From"]    = es["smtp_user"]
        msg["To"]      = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP(es["smtp_host"], es["smtp_port"], timeout=10) as server:
            server.starttls()
            server.login(es["smtp_user"], es["smtp_pass"])
            server.sendmail(es["smtp_user"], to_email, msg.as_string())
        # Update last_reminded
        with _get_conn() as conn:
            conn.execute(
                "UPDATE email_settings SET last_reminded=? WHERE user_id=?", (today, uid)
            )
        return True, f"Reminder sent to {to_email}!"
    except Exception as e:
        return False, f"Email failed: {str(e)[:80]}"

def save_email_settings(display_name: str, smtp_host: str, smtp_port: int,
                        smtp_user: str, smtp_pass: str, notify_streak: bool):
    uid = _get_user_id(display_name)
    if not uid:
        return
    with _get_conn() as conn:
        conn.execute(
            """INSERT INTO email_settings (user_id, smtp_host, smtp_port, smtp_user, smtp_pass, notify_streak)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET
               smtp_host=excluded.smtp_host, smtp_port=excluded.smtp_port,
               smtp_user=excluded.smtp_user, smtp_pass=excluded.smtp_pass,
               notify_streak=excluded.notify_streak""",
            (uid, smtp_host, smtp_port, smtp_user, smtp_pass, int(notify_streak))
        )

def load_email_settings(display_name: str) -> dict:
    uid = _get_user_id(display_name)
    if not uid:
        return {}
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT smtp_host, smtp_port, smtp_user, smtp_pass, notify_streak "
            "FROM email_settings WHERE user_id=?", (uid,)
        ).fetchone()
    if not row:
        return {"smtp_host": "", "smtp_port": 587, "smtp_user": "", "smtp_pass": "", "notify_streak": True}
    return dict(row)

# ── Admin DB helpers ──────────────────────────────────────────────────────────

def get_admin_password() -> str:
    with _get_conn() as conn:
        row = conn.execute("SELECT value FROM admin_config WHERE key='admin_password'").fetchone()
    return row["value"] if row else ""

def set_admin_password(pw: str):
    hashed = hashlib.sha256(pw.encode()).hexdigest()
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO admin_config (key, value) VALUES ('admin_password', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (hashed,)
        )

def get_admin_stats() -> dict:
    """Aggregate stats across all users for admin dashboard."""
    with _get_conn() as conn:
        total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        total_scans = conn.execute("SELECT COUNT(*) FROM scan_history").fetchone()[0]
        total_points = conn.execute("SELECT SUM(points) FROM user_stats").fetchone()[0] or 0
        active_today = conn.execute(
            "SELECT COUNT(DISTINCT user_id) FROM scan_history WHERE scanned_date=?",
            (datetime.date.today().isoformat(),)
        ).fetchone()[0]
        cat_rows = conn.execute(
            "SELECT category, COUNT(*) as cnt FROM scan_history GROUP BY category ORDER BY cnt DESC"
        ).fetchall()
        top_users = conn.execute(
            """SELECT u.display_name, s.points, s.total_scans, s.streak
               FROM user_stats s JOIN users u ON u.id=s.user_id
               ORDER BY s.points DESC LIMIT 10"""
        ).fetchall()
        daily_rows = conn.execute(
            """SELECT scanned_date, COUNT(*) as cnt
               FROM scan_history
               WHERE scanned_date >= date('now', '-29 days')
               GROUP BY scanned_date ORDER BY scanned_date ASC"""
        ).fetchall()
        new_users_7d = conn.execute(
            "SELECT COUNT(*) FROM users WHERE created_at >= date('now', '-7 days')"
        ).fetchone()[0]
    return {
        "total_users":  total_users,
        "total_scans":  total_scans,
        "total_points": total_points,
        "active_today": active_today,
        "new_users_7d": new_users_7d,
        "cat_breakdown": [(r["category"], r["cnt"]) for r in cat_rows],
        "top_users":    [dict(r) for r in top_users],
        "daily_scans":  [(r["scanned_date"], r["cnt"]) for r in daily_rows],
    }

# ── Trends query (per-user weekly/monthly breakdown) ─────────────────────────

def load_user_trends(display_name: str) -> pd.DataFrame:
    """Return a DataFrame of (scanned_date, category, count) for the user."""
    uid = _get_user_id(display_name)
    if not uid:
        return pd.DataFrame()
    with _get_conn() as conn:
        rows = conn.execute(
            """SELECT scanned_date, category, COUNT(*) as count
               FROM scan_history WHERE user_id=?
               GROUP BY scanned_date, category
               ORDER BY scanned_date ASC""",
            (uid,)
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["scanned_date"] = pd.to_datetime(df["scanned_date"])
    return df

# ── Google Translate helper (free endpoint) ───────────────────────────────────
def google_translate(text: str, target_lang: str) -> str:
    """Translate text using Google Translate free endpoint."""
    lang_code = {"Tamil": "ta", "Hindi": "hi"}.get(target_lang, "en")
    if lang_code == "en":
        return text
    try:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {"client": "gtx", "sl": "en", "tl": lang_code, "dt": "t", "q": text}
        r = requests.get(url, params=params, timeout=5)
        result = r.json()
        return "".join([s[0] for s in result[0] if s[0]])
    except Exception:
        return text  # fallback to English on error

# ── PDF Report Generator ──────────────────────────────────────────────────────
def generate_pdf_report(username, points, streak, total_scans, badges, history, level):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    # Custom styles
    title_style = ParagraphStyle("title", parent=styles["Title"],
        fontSize=22, textColor=colors.HexColor("#00c87a"),
        spaceAfter=4, alignment=TA_CENTER, fontName="Helvetica-Bold")
    sub_style = ParagraphStyle("sub", parent=styles["Normal"],
        fontSize=9, textColor=colors.HexColor("#6fa87a"),
        spaceAfter=2, alignment=TA_CENTER)
    heading_style = ParagraphStyle("heading", parent=styles["Heading2"],
        fontSize=13, textColor=colors.HexColor("#00c87a"),
        spaceBefore=14, spaceAfter=6, fontName="Helvetica-Bold")
    body_style = ParagraphStyle("body", parent=styles["Normal"],
        fontSize=10, textColor=colors.HexColor("#1a3320"), spaceAfter=4)

    # Header
    story.append(Paragraph("♻ Smart Waste AI", title_style))
    story.append(Paragraph("Personal Recycling Report", sub_style))
    story.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%d %B %Y, %H:%M')}", sub_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#00c87a"), spaceAfter=12))

    # User summary
    story.append(Paragraph("👤 User Summary", heading_style))
    summary_data = [
        ["Field", "Value"],
        ["Username",      username],
        ["Level",         level],
        ["Total Points",  str(points)],
        ["Total Scans",   str(total_scans)],
        ["Current Streak",f"{streak} days"],
        ["Badges Earned", str(len(badges))],
    ]
    t = Table(summary_data, colWidths=[6*cm, 10*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,0), colors.HexColor("#00c87a")),
        ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
        ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 10),
        ("BACKGROUND",   (0,1), (-1,-1), colors.HexColor("#f0fff6")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f0fff6"), colors.HexColor("#e0f5ea")]),
        ("GRID",         (0,0), (-1,-1), 0.5, colors.HexColor("#b2d8be")),
        ("LEFTPADDING",  (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
    ]))
    story.append(t)

    # Category breakdown
    if history:
        story.append(Paragraph("📊 Waste Category Breakdown", heading_style))
        from collections import Counter
        cat_counts = Counter(h["category"] for h in history)
        cat_data = [["Category", "Scans", "Percentage"]]
        for cat, count in cat_counts.most_common():
            pct = f"{count/len(history)*100:.1f}%"
            cat_data.append([cat, str(count), pct])
        ct = Table(cat_data, colWidths=[7*cm, 5*cm, 5*cm])
        ct.setStyle(TableStyle([
            ("BACKGROUND",   (0,0), (-1,0), colors.HexColor("#00c87a")),
            ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
            ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",     (0,0), (-1,-1), 10),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f0fff6"), colors.HexColor("#e0f5ea")]),
            ("GRID",         (0,0), (-1,-1), 0.5, colors.HexColor("#b2d8be")),
            ("ALIGN",        (1,0), (-1,-1), "CENTER"),
            ("LEFTPADDING",  (0,0), (-1,-1), 8),
            ("TOPPADDING",   (0,0), (-1,-1), 5),
            ("BOTTOMPADDING",(0,0), (-1,-1), 5),
        ]))
        story.append(ct)

        # Recent scan history
        story.append(Paragraph("🕘 Recent Scan History (Last 10)", heading_style))
        hist_data = [["#", "Category", "Confidence", "Time"]]
        for i, row in enumerate(reversed(history[-10:]), 1):
            hist_data.append([str(i), row["category"], f"{row['confidence']}%", row["time"]])
        ht = Table(hist_data, colWidths=[1.5*cm, 6*cm, 4*cm, 6*cm])
        ht.setStyle(TableStyle([
            ("BACKGROUND",   (0,0), (-1,0), colors.HexColor("#00c87a")),
            ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
            ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",     (0,0), (-1,-1), 9),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f0fff6"), colors.HexColor("#e0f5ea")]),
            ("GRID",         (0,0), (-1,-1), 0.5, colors.HexColor("#b2d8be")),
            ("LEFTPADDING",  (0,0), (-1,-1), 6),
            ("TOPPADDING",   (0,0), (-1,-1), 4),
            ("BOTTOMPADDING",(0,0), (-1,-1), 4),
        ]))
        story.append(ht)

    # Footer
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#b2d8be")))
    story.append(Paragraph("Smart Waste AI · Powered by Google Gemini · Keep Recycling! ♻", sub_style))

    doc.build(story)
    buf.seek(0)
    return buf

# ── Google Maps search queries per category (no API key needed) ───────────────
MAPS_QUERY = {
    "Plastic":  "plastic+recycling+center+near+me",
    "Organic":  "compost+drop+off+near+me",
    "Metal":    "scrap+metal+recycling+near+me",
    "Paper":    "paper+recycling+center+near+me",
    "E-waste":  "e-waste+drop+off+center+near+me",
}

DISPOSAL_TIPS = {
    "Plastic":  ("🧴", "Rinse the item, then place in the **plastic recycling bin**. Avoid black plastic — often non-recyclable."),
    "Organic":  ("🌿", "Add to your **compost bin** or organic/wet waste collection. Great for garden mulch!"),
    "Metal":    ("🔩", "Place in the **metal/dry recycling bin**. Aluminium cans can be recycled indefinitely."),
    "Paper":    ("📄", "Keep it dry and place in the **paper recycling bin**. Shredded paper can go in compost too."),
    "E-waste":  ("⚡", "Do **NOT** bin this. Take to an **e-waste drop-off center** to prevent toxic leaching."),
}
CATEGORY_COLORS = {
    "Plastic": "#38bdf8", "Organic": "#4ade80", "Metal": "#fbbf24",
    "Paper": "#f472b6",   "E-waste": "#a78bfa",
}

# ── Badges definition ─────────────────────────────────────────────────────────
BADGES = [
    {"id": "first_scan",   "icon": "🌱", "name": "First Scan",     "desc": "Complete your first scan",          "condition": lambda h, p, s: len(h) >= 1},
    {"id": "streak_3",     "icon": "🔥", "name": "On Fire",         "desc": "Scan 3 days in a row",              "condition": lambda h, p, s: s >= 3},
    {"id": "10_scans",     "icon": "🔬", "name": "Researcher",      "desc": "Complete 10 scans",                 "condition": lambda h, p, s: len(h) >= 10},
    {"id": "all_5",        "icon": "🌈", "name": "Explorer",        "desc": "Find all 5 waste categories",       "condition": lambda h, p, s: len(set(x["category"] for x in h)) == 5},
    {"id": "100_points",   "icon": "⭐", "name": "Star Recycler",   "desc": "Earn 100 points",                   "condition": lambda h, p, s: p >= 100},
    {"id": "ewaste_scan",  "icon": "⚡", "name": "E-Hero",          "desc": "Scan an e-waste item",              "condition": lambda h, p, s: any(x["category"] == "E-waste" for x in h)},
    {"id": "500_points",   "icon": "🏆", "name": "Champion",        "desc": "Earn 500 points",                   "condition": lambda h, p, s: p >= 500},
    {"id": "25_scans",     "icon": "🎖️", "name": "Waste Warrior",  "desc": "Complete 25 scans",                 "condition": lambda h, p, s: len(h) >= 25},
]

LEVELS = [
    (0,    "🌱 Beginner"),
    (50,   "♻️ Recycler"),
    (150,  "🌿 Eco Aware"),
    (300,  "🌍 Green Hero"),
    (600,  "⭐ Waste Master"),
    (1000, "🏆 Eco Champion"),
]

def get_level(points):
    level_name = LEVELS[0][1]
    for threshold, name in LEVELS:
        if points >= threshold:
            level_name = name
    return level_name

def get_next_level_points(points):
    for i, (threshold, name) in enumerate(LEVELS):
        if points < threshold:
            prev = LEVELS[i-1][0] if i > 0 else 0
            return threshold, prev
    return LEVELS[-1][0], LEVELS[-2][0]

def check_badges():
    h = st.session_state.history
    p = st.session_state.points
    s = st.session_state.streak
    new_badges = []
    for badge in BADGES:
        if badge["id"] not in st.session_state.badges and badge["condition"](h, p, s):
            st.session_state.badges.add(badge["id"])
            new_badges.append(badge)
    return new_badges

def add_points(confidence):
    pts = 10 + int(confidence * 0.1)  # base 10 + bonus for high confidence
    st.session_state.points += pts
    st.session_state.total_scans += 1
    # Update streak
    today = datetime.date.today()
    if st.session_state.last_scan:
        diff = (today - st.session_state.last_scan).days
        if diff == 1:
            st.session_state.streak += 1
        elif diff > 1:
            st.session_state.streak = 1
    else:
        st.session_state.streak = 1
    st.session_state.last_scan = today
    return pts

# ── Voice announcement (browser TTS) ─────────────────────────────────────────
def speak(text: str):
    import streamlit.components.v1 as components
    safe = text.replace("'", "\\'").replace('"', '\\"').replace("\n", " ")
    components.html(f"""
    <script>
    (function() {{
        function trySpeak() {{
            if (!window.speechSynthesis) return;
            window.speechSynthesis.cancel();
            const msg = new SpeechSynthesisUtterance('{safe}');
            msg.rate = 0.95;
            msg.pitch = 1.1;
            msg.volume = 1;
            // Some browsers need a tiny delay after cancel()
            setTimeout(function() {{
                window.speechSynthesis.speak(msg);
            }}, 150);
        }}
        // Wait for the page to be ready
        if (document.readyState === 'complete') {{
            trySpeak();
        }} else {{
            window.addEventListener('load', trySpeak);
        }}
    }})();
    </script>
    """, height=0)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""<div class="hero-wrap">
  <div class="hero-eyebrow">⬡ AI-Powered Waste Intelligence</div>
  <div class="hero-title">{T('app_title')}</div>
  <div class="hero-sub">{T('app_sub')}</div>
  <div class="hero-divider"></div>
</div>""", unsafe_allow_html=True)
with st.sidebar:
    # Premium logo
    st.markdown("""<div class="sidebar-logo">
        <div class="sidebar-logo-icon">♻</div>
        <div>
            <div class="sidebar-logo-text">Smart Waste AI</div>
            <div class="sidebar-logo-sub">GEMINI · TENSORFLOW · FREE</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # API Key section
    if _ENV_API_KEY:
        api_key = _ENV_API_KEY
    else:
        st.markdown(f'<div style="font-size:0.72rem;color:var(--text-3);letter-spacing:0.08em;text-transform:uppercase;font-family:Space Mono,monospace;margin-bottom:0.4rem;">{T("config")}</div>', unsafe_allow_html=True)
        api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            placeholder="AIza... (or set in .env file)",
        )
        st.markdown("[🔑 Get FREE API key →](https://aistudio.google.com/app/apikey)", unsafe_allow_html=True)
    st.markdown("---")

    # Logged-in user info + logout
    st.markdown(f"""<div style="background:rgba(184,255,0,0.04);border:1px solid rgba(184,255,0,0.10);
        border-radius:12px;padding:0.65rem 0.9rem;margin-bottom:0.6rem;display:flex;
        align-items:center;gap:0.5rem;">
        <span style="font-size:1.1rem;">👤</span>
        <div>
            <div style="font-family:'Syne',sans-serif;font-size:0.82rem;font-weight:700;color:var(--lime);">
                {st.session_state.auth_username}</div>
            <div style="font-size:0.65rem;color:var(--text-3);font-family:'Space Mono',monospace;">LOGGED IN</div>
        </div>
    </div>""", unsafe_allow_html=True)
    if st.button("🚪 Logout", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # Username
    uname = st.text_input("👤 Your Name / Username", value=st.session_state.username, placeholder="Enter your name...")
    if uname: st.session_state.username = uname

    # Language selector
    lang_choice = st.selectbox("🌐 Language / மொழி / भाषा", ["English", "Tamil", "Hindi"],
                                index=["English","Tamil","Hindi"].index(st.session_state.language))
    st.session_state.language = lang_choice

    st.markdown("---")

    # Mini stats in sidebar — premium styled
    level = get_level(st.session_state.points)
    next_pts, prev_pts = get_next_level_points(st.session_state.points)
    progress = min(100, int((st.session_state.points - prev_pts) / max(1, next_pts - prev_pts) * 100))
    st.markdown(f"""<div style="background:rgba(184,255,0,0.04);border:1px solid rgba(184,255,0,0.10);
        border-radius:14px;padding:1rem 1rem 0.8rem;">
        <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:800;
             color:var(--lime);margin-bottom:0.15rem;">{level}</div>
        <div style="font-size:0.72rem;color:var(--text-3);margin-bottom:0.6rem;">
            ⭐ <strong style="color:var(--text-2);">{st.session_state.points}</strong> pts &nbsp;·&nbsp;
            🔥 <strong style="color:#ffaa00;">{st.session_state.streak}</strong> day streak
        </div>
        <div class="points-bar"><div class="points-fill" style="width:{progress}%"></div></div>
        <div style="font-size:0.65rem;color:var(--text-3);margin-top:0.35rem;font-family:'Space Mono',monospace;">
            {st.session_state.points} / {next_pts} to next level
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    with st.expander("⚙️ Settings", expanded=False):
        # ── Local AI (TensorFlow) toggle ──────────────────────────────────────
        st.markdown("**🧠 Classification Engine**")
        tf_mode = st.radio(
            "Engine",
            ["🌐 Gemini Vision (API)", "⚡ Local TensorFlow (No API)"],
            label_visibility="collapsed",
            index=0,
        )
        use_tf = tf_mode.startswith("⚡")
        if use_tf:
            with st.spinner("Loading EfficientNetB0…"):
                try:
                    load_tf_model()
                    st.success("✅ Model ready · EfficientNetB0 (ImageNet)")
                except Exception as e:
                    st.error(f"❌ TF load failed: {e}")
        else:
            st.caption("Uses Gemini API key above.")
        st.markdown("---")
        voice_on = st.toggle(T("voice"), value=True)
        light_mode = st.toggle("☀️ Light Mode", value=st.session_state.light_mode)
        if light_mode != st.session_state.light_mode:
            st.session_state.light_mode = light_mode
            inject_theme(light_mode)
            st.rerun()
        sounds_on = st.toggle("🔊 Sound Effects", value=st.session_state.sounds_on)
        st.session_state.sounds_on = sounds_on
    st.markdown(f"**{T('categories')}**")
    for cls in CLASSES:
        st.markdown(f"{DISPOSAL_TIPS[cls][0]} {cls}")
    st.markdown("---")
    if st.button(T("clear_history"), use_container_width=True):
        st.session_state.history = []
        st.session_state.points = 0
        st.session_state.streak = 0
        st.session_state.badges = set()
        st.session_state.total_scans = 0
        st.session_state.last_scan = None
        st.rerun()

# ── Classify function ─────────────────────────────────────────────────────────
def classify_waste(image: Image.Image, api_key: str) -> dict:
    client = genai.Client(api_key=api_key)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    image_bytes = buf.getvalue()
    prompt = f"""You are a waste classification AI. Analyze this image and classify the waste.
The possible categories are ONLY: {', '.join(CLASSES)}
Respond ONLY with a valid JSON object, no markdown, no extra text:
{{"category": "<one of the categories above>","confidence": <integer 0-100>,"reason": "<one short sentence explaining why>","scores": {{"Plastic": <0-100>,"Organic": <0-100>,"Metal": <0-100>,"Paper": <0-100>,"E-waste": <0-100>}}}}"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"), prompt]
    )
    raw = response.text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

def process_result(result):
    category   = result.get("category", "Unknown")
    confidence = result.get("confidence", 0)
    reason     = result.get("reason", "")
    scores     = result.get("scores", {})
    color      = CATEGORY_COLORS.get(category, "#69f0ae")
    icon, tip  = DISPOSAL_TIPS.get(category, ("♻️", "Dispose responsibly."))

    # Points & streak
    pts_earned = add_points(confidence)

    # Save history
    scan_entry = {
        "time": datetime.datetime.now().strftime("%d %b %Y, %H:%M"),
        "category": category, "confidence": confidence,
        "reason": reason, "icon": icon,
        "source": result.get("source", "gemini"),
    }
    st.session_state.history.append(scan_entry)
    save_scan_to_db(scan_entry)

    # Check badges
    new_badges = check_badges()

    # Persist updated stats to DB
    save_user_stats()

    # Sound effects
    if st.session_state.get("sounds_on", True):
        if new_badges:
            play_sound("badge")
        else:
            play_sound("scan")

    # Voice
    if voice_on:
        speak(f"This is {category} waste. {reason}. Confidence: {confidence} percent. You earned {pts_earned} points!")

    # Translate reason if needed
    lang = st.session_state.get("language", "English")
    display_reason = google_translate(reason, lang) if lang != "English" else reason

    # Result card — premium with confidence ring
    source = result.get("source", "gemini")
    source_badge = (
        '<span style="font-family:\'Space Mono\',monospace;font-size:0.6rem;'
        'background:rgba(184,255,0,0.08);color:var(--lime);border:1px solid rgba(184,255,0,0.22);'
        'border-radius:100px;padding:2px 9px;letter-spacing:0.07em;">⚡ EfficientNetB0</span>'
        if source == "tensorflow" else
        '<span style="font-family:\'Space Mono\',monospace;font-size:0.6rem;'
        'background:rgba(0,255,213,0.07);color:#00ffd5;border:1px solid rgba(0,255,213,0.22);'
        'border-radius:100px;padding:2px 9px;letter-spacing:0.07em;">🌐 Gemini 2.5 Flash</span>'
    )

    # SVG confidence ring
    radius     = 46
    circ       = 2 * 3.14159 * radius
    dash_fill  = circ * confidence / 100
    dash_gap   = circ - dash_fill

    rc1, rc2 = st.columns([1, 1.6])
    with rc1:
        st.markdown(f"""<div class="result-card" style="border-color:{color}28;height:100%;">
            <div style="position:absolute;top:0;left:0;right:0;height:2px;
                 background:linear-gradient(90deg,transparent,{color},transparent);border-radius:var(--r) var(--r) 0 0;"></div>
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
                <div style="font-size:0.6rem;color:var(--text-3);letter-spacing:0.2em;
                     text-transform:uppercase;font-family:'Space Mono',monospace;">{T('result_label')}</div>
                {source_badge}
            </div>
            <div style="display:flex;align-items:center;gap:1.2rem;margin-bottom:1rem;">
                <div class="conf-ring-wrap">
                    <div class="conf-ring">
                        <svg viewBox="0 0 110 110" width="110" height="110">
                            <circle class="conf-ring-track" cx="55" cy="55" r="{radius}"/>
                            <circle class="conf-ring-fill conf-ring-fill-anim"
                                cx="55" cy="55" r="{radius}"
                                stroke="{color}"
                                stroke-dasharray="{dash_fill:.1f} {dash_gap:.1f}"
                                style="filter:drop-shadow(0 0 6px {color})"/>
                        </svg>
                        <div class="conf-ring-label">
                            <span class="conf-ring-number" style="color:{color};">{confidence}</span>
                            <span class="conf-ring-text">% conf</span>
                        </div>
                    </div>
                </div>
                <div>
                    <div class="category-badge" style="border-color:{color}55;color:{color};">{icon} {category}</div>
                    <div style="font-size:0.82rem;color:var(--text-2);line-height:1.6;">{display_reason}</div>
                </div>
            </div>
            <div style="display:flex;align-items:center;justify-content:space-between;
                 padding:0.6rem 0.8rem;background:rgba(184,255,0,0.035);border-radius:10px;
                 border:1px solid rgba(184,255,0,0.08);margin-top:0.5rem;">
                <div style="font-size:0.72rem;color:var(--text-2);">Points earned</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
                     color:var(--lime);letter-spacing:-0.02em;">+{pts_earned} ⭐</div>
            </div>
        </div>""", unsafe_allow_html=True)

    with rc2:
        # Animated score bars
        bars_html = ""
        for cls in CLASSES:
            sc   = scores.get(cls, 0)
            clr2 = CATEGORY_COLORS.get(cls, "#b8ff00")
            ico2 = DISPOSAL_TIPS[cls][0]
            bars_html += f"""<div class="score-bar-wrap">
                <div class="score-bar-label">
                    <span>{ico2} {cls}</span><span style="color:{clr2};font-weight:700;">{sc}%</span>
                </div>
                <div class="score-bar-track">
                    <div class="score-bar-fill" style="width:{sc}%;background:linear-gradient(90deg,{clr2}88,{clr2});"></div>
                </div>
            </div>"""
        st.markdown(f"""<div class="glass-card" style="border-color:{color}18;height:100%;">
            <div style="font-size:0.65rem;color:var(--text-3);letter-spacing:0.18em;
                 text-transform:uppercase;font-family:'Space Mono',monospace;margin-bottom:0.9rem;">
                📊 {T('breakdown')}
            </div>
            {bars_html}
        </div>""", unsafe_allow_html=True)

    tip_translated = google_translate(tip, lang) if lang != "English" else tip
    st.markdown(f'<div class="tip-box">{T("disposal_tip")} {tip_translated}</div>', unsafe_allow_html=True)

    # ── AI Suggestions (inline after result) ──────────────────────────────────
    with st.spinner("🤖 Generating AI suggestions…"):
        suggestions = get_ai_suggestions(category, reason, api_key if not use_tf else api_key)
        st.session_state.last_suggestions = suggestions
        st.session_state.last_suggest_cat = category

    # Reuse ideas
    st.markdown(f"""<div class="suggest-card">
        <div class="suggest-tag" style="color:#00e8c8;border-color:#00e8c822;">♻️ REUSE IDEAS</div>
        <div style="display:flex;flex-direction:column;gap:0.5rem;">
            {"".join(f'<div style="display:flex;gap:0.6rem;align-items:flex-start;font-size:0.85rem;color:var(--text);line-height:1.5;"><span style="color:#00e8c8;margin-top:2px;">→</span><span>{tip}</span></div>' for tip in suggestions["reuse"])}
        </div>
    </div>""", unsafe_allow_html=True)

    # Upcycle ideas
    st.markdown(f"""<div class="suggest-card">
        <div class="suggest-tag" style="color:#c8f135;border-color:#c8f13522;">✨ UPCYCLE IDEAS</div>
        <div style="display:flex;flex-direction:column;gap:0.5rem;">
            {"".join(f'<div style="display:flex;gap:0.6rem;align-items:flex-start;font-size:0.85rem;color:var(--text);line-height:1.5;"><span style="color:#c8f135;margin-top:2px;">→</span><span>{tip}</span></div>' for tip in suggestions["upcycle"])}
        </div>
    </div>""", unsafe_allow_html=True)

    # Impact + Fun Fact in two columns
    s1, s2 = st.columns(2)
    with s1:
        st.markdown(f"""<div class="suggest-card" style="height:100%;">
            <div class="suggest-tag" style="color:#ffb800;border-color:#ffb80022;">🌍 YOUR IMPACT</div>
            <div style="font-size:0.84rem;color:var(--text);line-height:1.6;">{suggestions["impact"]}</div>
        </div>""", unsafe_allow_html=True)
    with s2:
        st.markdown(f"""<div class="suggest-card" style="height:100%;">
            <div class="suggest-tag" style="color:#bb88ff;border-color:#bb88ff22;">💡 DID YOU KNOW?</div>
            <div style="font-size:0.84rem;color:var(--text);line-height:1.6;">{suggestions["did_you_know"]}</div>
        </div>""", unsafe_allow_html=True)

    # Extra Gemini tips if available
    if suggestions.get("extra_tips"):
        st.markdown(f"""<div class="suggest-card" style="border-color:rgba(200,241,53,0.22);background:linear-gradient(135deg,rgba(200,241,53,0.05),rgba(0,232,200,0.03));">
            <div class="suggest-tag" style="color:#c8f135;border-color:#c8f13533;">🌐 GEMINI AI · ITEM-SPECIFIC TIPS</div>
            <div style="display:flex;flex-direction:column;gap:0.5rem;">
                {"".join(f'<div style="display:flex;gap:0.6rem;align-items:flex-start;font-size:0.85rem;color:var(--text);line-height:1.5;"><span style="color:#c8f135;margin-top:2px;">✦</span><span>{t}</span></div>' for t in suggestions["extra_tips"])}
            </div>
            {"" if not suggestions.get("local_action") else f'<div style="margin-top:0.8rem;padding:0.6rem 0.8rem;background:rgba(200,241,53,0.06);border-radius:8px;font-size:0.82rem;color:#c8f135;"><strong>📍 Take Action Today:</strong> {suggestions["local_action"]}</div>'}
            {"" if not suggestions.get("motivational") else f'<div style="margin-top:0.6rem;font-size:0.8rem;color:#9ab87a;font-style:italic;">{suggestions["motivational"]}</div>'}
        </div>""", unsafe_allow_html=True)

    # Fun fact at the bottom
    st.markdown(f"""<div style="background:rgba(255,184,0,0.05);border:1px solid rgba(255,184,0,0.15);
        border-radius:12px;padding:0.8rem 1.1rem;margin-top:0.4rem;
        font-size:0.82rem;color:#ffb800;line-height:1.55;">
        🎯 <strong>Fun Fact:</strong> {suggestions["fun_fact"]}
    </div>""", unsafe_allow_html=True)

    # New badge notifications
    for badge in new_badges:
        st.balloons()
        st.success(f"🎉 Badge Unlocked: **{badge['icon']} {badge['name']}** — {badge['desc']}")

# ── Tabs ──────────────────────────────────────────────────────────────────────
# ── AI Suggestions helpers ────────────────────────────────────────────────────
def get_ai_suggestions(category: str, reason: str, api_key: str = "") -> dict:
    """
    Generate rich AI suggestions for a classified waste item.
    Falls back to curated static suggestions if no API key.
    Returns dict with keys: reuse, upcycle, impact, fun_fact, did_you_know
    """
    STATIC = {
        "Plastic": {
            "reuse":      ["Refill water bottles instead of buying new ones", "Use plastic containers as storage boxes or plant pots", "Reuse plastic bags as bin liners"],
            "upcycle":    ["Turn plastic bottles into a drip irrigation system for plants", "Cut plastic bottles into strips to make cable organisers", "Use bottle caps to make colourful mosaics or art"],
            "impact":     "Recycling one plastic bottle saves enough energy to power a 60W light bulb for 6 hours.",
            "fun_fact":   "Plastic takes up to 1000 years to decompose in a landfill.",
            "did_you_know": "Only 9% of all plastic ever produced has been recycled. You're part of the solution!",
        },
        "Organic": {
            "reuse":      ["Use vegetable peels to make a nutritious stock", "Banana peels can be used as a natural shoe polish", "Coffee grounds repel garden pests naturally"],
            "upcycle":    ["Start a home compost bin — organic waste becomes rich garden fertiliser in 8 weeks", "Make natural dye from fruit skins (avocado pits = pink, onion skins = gold)", "Dry citrus peels to make natural fire starters"],
            "impact":     "Composting one kg of organic waste prevents 0.5 kg of methane — a greenhouse gas 25x more potent than CO₂.",
            "fun_fact":   "Food waste in landfills is the 3rd largest source of human-caused methane emissions globally.",
            "did_you_know": "A home compost bin can divert up to 150 kg of waste from landfill every year.",
        },
        "Metal": {
            "reuse":      ["Tin cans make excellent pencil holders or small plant pots", "Use metal lids as coasters or palette trays", "Old keys can become wind chimes or decorative art"],
            "upcycle":    ["Turn aluminium cans into a tiny stove for camping", "Use tin cans to make lanterns by punching patterns with a nail", "Flatten aluminium foil and roll it into a ball — recycle when you have a large ball"],
            "impact":     "Recycling one aluminium can saves enough energy to run a TV for 3 hours.",
            "fun_fact":   "Aluminium can be recycled indefinitely without losing quality — it's the most valuable material in your recycling bin.",
            "did_you_know": "Recycling steel uses 75% less energy than making new steel from raw ore.",
        },
        "Paper": {
            "reuse":      ["Use the blank side of printed paper for notes or sketches", "Newspaper makes excellent wrapping paper or packing material", "Use paper bags as book covers"],
            "upcycle":    ["Shred newspaper into strips and use as packing material instead of bubble wrap", "Make seed-starter pots from rolled newspaper — they're biodegradable!", "Create papier-mâché art or bowls from old newspapers"],
            "impact":     "Recycling one tonne of paper saves 17 trees, 26,000 litres of water and 4,000 kWh of electricity.",
            "fun_fact":   "Paper can typically be recycled 5–7 times before the fibres become too short.",
            "did_you_know": "The average person uses about 100 kg of paper per year. Choosing recycled paper cuts that carbon footprint in half.",
        },
        "E-waste": {
            "reuse":      ["Donate working electronics to schools or charities", "Old phones make great dedicated music players or smart home controllers", "Use old tablets as digital photo frames or recipe screens in the kitchen"],
            "upcycle":    ["Strip old circuit boards for decorative jewellery or art pieces", "Turn old keyboards into cable organisers by removing the keys", "Use old hard drives as external storage with a cheap USB enclosure"],
            "impact":     "E-waste contains gold, silver, copper and palladium — recycling one million phones recovers about 24 kg of gold.",
            "fun_fact":   "E-waste is the world's fastest-growing waste stream — 53 million tonnes generated in 2019 alone.",
            "did_you_know": "A single smartphone contains over 60 different elements from the periodic table.",
        },
    }

    # Always get the static base
    base = STATIC.get(category, STATIC["Plastic"])

    # If API key available, enrich with Gemini
    if api_key:
        try:
            client = genai.Client(api_key=api_key)
            prompt = f"""You are an expert sustainability coach. A waste item was classified as: {category}.
Classification reason: {reason}

Give 3 additional creative, practical suggestions specific to THIS item (not generic {category} advice).
Respond ONLY with valid JSON, no markdown:
{{
  "extra_tips": ["tip 1", "tip 2", "tip 3"],
  "local_action": "one specific local action the user can take today",
  "motivational": "one short motivational sentence about their recycling impact"
}}"""
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt]
            )
            raw = response.text.strip().replace("```json","").replace("```","").strip()
            extra = json.loads(raw)
            base["extra_tips"]   = extra.get("extra_tips", [])
            base["local_action"] = extra.get("local_action", "")
            base["motivational"] = extra.get("motivational", "")
        except Exception:
            pass  # fall back silently to static

    return base


def get_chat_response(question: str, history: list, api_key: str) -> str:
    """Get a response from Gemini for the AI Coach chat."""
    if not api_key:
        return "Please enter your Gemini API key in the sidebar to use the AI Coach chat."
    try:
        client  = genai.Client(api_key=api_key)
        history_str = "\n".join([f"User: {h['q']}\nAI: {h['a']}" for h in history[-4:]])
        cat_summary = ", ".join(set(h["category"] for h in st.session_state.history)) if st.session_state.history else "none yet"
        prompt = f"""You are a friendly, expert waste management and sustainability coach embedded in Smart Waste AI.
The user has scanned these waste categories so far: {cat_summary}.
Total scans: {len(st.session_state.history)}, Points: {st.session_state.points}.

Previous conversation:
{history_str}

User question: {question}

Give a helpful, concise, practical answer in 2-4 sentences. Be encouraging and specific.
If asked about recycling, disposal, upcycling, sustainability or waste — answer confidently.
If asked something unrelated, gently redirect to waste/sustainability topics."""
        response = client.models.generate_content(model="gemini-2.5-flash", contents=[prompt])
        return response.text.strip()
    except Exception as e:
        return f"Sorry, I couldn't connect to the AI right now. ({str(e)[:60]})"


# ── Session state for suggestions & chat ──────────────────────────────────────
if "last_suggestions" not in st.session_state: st.session_state.last_suggestions = None
if "last_suggest_cat" not in st.session_state: st.session_state.last_suggest_cat = ""
if "chat_history"     not in st.session_state: st.session_state.chat_history = []

tab_scan, tab_stats, tab_explore, tab_reports, tab_admin = st.tabs([
    "🔬 Scan", "📊 My Stats", "🌍 Explore", "📋 Reports", "🔐 Admin"
])

# ── TAB: SCAN ─────────────────────────────────────────────────────────────────
with tab_scan:
    sub_upload, sub_webcam = st.tabs([T("tab_upload"), T("tab_webcam")])

# ── TAB 1: UPLOAD ─────────────────────────────────────────────────────────────
with sub_upload:
    # ── Single or Batch toggle ────────────────────────────────────────────────
    scan_mode = st.radio(
        "Scan mode", ["📁 Single Image", "📦 Batch Scan (Multiple)"],
        horizontal=True, label_visibility="collapsed", key="scan_mode_radio"
    )
    st.markdown(f'<div class="upload-hint">{T("upload_hint")}</div>', unsafe_allow_html=True)

    if scan_mode == "📁 Single Image":
        uploaded_file = st.file_uploader("Drop a waste image here", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(image, caption="Uploaded Image", width=300)
                if use_tf:
                    st.markdown("""<div style="display:inline-flex;align-items:center;gap:0.4rem;
                        background:rgba(200,241,53,0.08);border:1px solid rgba(200,241,53,0.25);
                        border-radius:100px;padding:0.3rem 0.8rem;font-size:0.72rem;
                        font-family:'DM Mono',monospace;color:#c8f135;letter-spacing:0.08em;margin-top:0.5rem;">
                        ⚡ LOCAL · EfficientNetB0 · ImageNet
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""<div style="display:inline-flex;align-items:center;gap:0.4rem;
                        background:rgba(0,232,200,0.08);border:1px solid rgba(0,232,200,0.25);
                        border-radius:100px;padding:0.3rem 0.8rem;font-size:0.72rem;
                        font-family:'DM Mono',monospace;color:#00e8c8;letter-spacing:0.08em;margin-top:0.5rem;">
                        🌐 CLOUD · Gemini 2.5 Flash
                    </div>""", unsafe_allow_html=True)

            with col2:
                if use_tf:
                    with st.spinner("⚡ Running EfficientNetB0 locally…"):
                        try:
                            result = classify_with_tf(image)
                            st.markdown("""<div style="background:linear-gradient(135deg,rgba(200,241,53,0.07),rgba(0,0,0,0));
                                border:1px solid rgba(200,241,53,0.18);border-radius:12px;
                                padding:0.6rem 1rem;margin-bottom:0.8rem;font-size:0.78rem;
                                font-family:'DM Mono',monospace;color:#c8f135;">
                                ⚡ Classified by <strong>EfficientNetB0</strong> — pretrained on ImageNet (1000 classes)<br>
                                <span style="color:#9ab87a;font-size:0.7rem;">No API key required · Runs entirely on your server</span>
                            </div>""", unsafe_allow_html=True)
                            process_result(result)
                        except Exception as e:
                            st.error(f"❌ TensorFlow error: {str(e)}")
                elif not api_key:
                    st.warning("⚠️ Enter your **Gemini API key** in the sidebar, or switch to **Local TensorFlow** mode (no key needed).")
                else:
                    with st.spinner("🔍 Analysing waste..."):
                        try:
                            result = classify_waste(image, api_key)
                            process_result(result)
                        except json.JSONDecodeError:
                            st.error("❌ Could not parse AI response. Please try again.")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")

    else:
        # ── BATCH SCAN ────────────────────────────────────────────────────────
        batch_files = st.file_uploader(
            "Upload multiple waste images",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if batch_files:
            st.markdown(f"""<div style="font-size:0.82rem;color:var(--text-2);margin-bottom:0.8rem;">
                📦 <strong style="color:var(--lime);">{len(batch_files)}</strong> images selected — click below to classify all
            </div>""", unsafe_allow_html=True)

            if not use_tf and not api_key:
                st.warning("⚠️ Enter your Gemini API key or switch to Local TF mode to run batch scan.")
            else:
                if st.button(f"🚀 Classify All {len(batch_files)} Images", use_container_width=True):
                    batch_results = []
                    progress_bar = st.progress(0, text="Starting batch scan…")
                    for i, f in enumerate(batch_files):
                        progress_bar.progress((i) / len(batch_files), text=f"Scanning {i+1}/{len(batch_files)}: {f.name}")
                        try:
                            img = Image.open(f).convert("RGB")
                            if use_tf:
                                res = classify_with_tf(img)
                            else:
                                res = classify_waste(img, api_key)
                            # Save to DB silently
                            scan_entry = {
                                "time": datetime.datetime.now().strftime("%d %b %Y, %H:%M"),
                                "category": res["category"], "confidence": res["confidence"],
                                "reason": res.get("reason",""), "icon": DISPOSAL_TIPS.get(res["category"],("♻️",""))[0],
                                "source": res.get("source","gemini"),
                            }
                            st.session_state.history.append(scan_entry)
                            save_scan_to_db(scan_entry)
                            add_points(res["confidence"])
                            batch_results.append({"file": f.name, "img": img, "result": res, "error": None})
                        except Exception as e:
                            batch_results.append({"file": f.name, "img": None, "result": None, "error": str(e)})
                    save_user_stats()
                    check_badges()
                    progress_bar.progress(1.0, text="✅ Batch scan complete!")
                    if st.session_state.get("sounds_on", True):
                        play_sound("scan")

                    # Summary
                    ok_results = [r for r in batch_results if r["result"]]
                    from collections import Counter
                    cat_counts = Counter(r["result"]["category"] for r in ok_results)
                    avg_conf   = int(sum(r["result"]["confidence"] for r in ok_results) / max(1, len(ok_results)))
                    # Build category breakdown HTML separately to avoid f-string backslash issues
                    cat_html = ""
                    for cat, cnt in cat_counts.most_common():
                        c_color = CATEGORY_COLORS.get(cat, "#69f0ae")
                        c_icon  = DISPOSAL_TIPS.get(cat, ("♻️",""))[0]
                        cat_html += f"""<div><div style="font-size:0.65rem;color:var(--text-3);letter-spacing:0.1em;">{cat.upper()}</div>
                            <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:{c_color};">{c_icon} {cnt}</div></div>"""

                    st.markdown(f"""<div class="card" style="margin:0.8rem 0;">
                        <div style="display:flex;gap:2rem;flex-wrap:wrap;align-items:center;">
                            <div><div style="font-size:0.65rem;color:var(--text-3);letter-spacing:0.1em;">SCANNED</div>
                                <div style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;color:var(--lime);">{len(ok_results)}</div></div>
                            <div><div style="font-size:0.65rem;color:var(--text-3);letter-spacing:0.1em;">AVG CONFIDENCE</div>
                                <div style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;color:var(--lime);">{avg_conf}%</div></div>
                            {cat_html}
                        </div>
                    </div>""", unsafe_allow_html=True)

                    # Thumbnail grid
                    st.markdown("#### 🖼️ Results")
                    cols = st.columns(min(4, len(batch_results)))
                    for i, br in enumerate(batch_results):
                        with cols[i % len(cols)]:
                            if br["img"]:
                                st.image(br["img"], width=150)
                                cat   = br["result"]["category"]
                                conf  = br["result"]["confidence"]
                                color = CATEGORY_COLORS.get(cat, "#69f0ae")
                                icon  = DISPOSAL_TIPS.get(cat, ("♻️",""))[0]
                                st.markdown(f"""<div class="batch-result-badge" style="color:{color};border-color:{color}55;">
                                    {icon} {cat} · {conf}%
                                </div>
                                <div style="font-size:0.65rem;color:var(--text-3);margin-top:0.2rem;word-break:break-all;">{br['file'][:22]}</div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""<div style="padding:1rem;border:1px solid rgba(255,100,100,0.2);border-radius:10px;font-size:0.75rem;color:#ff6b6b;">
                                    ❌ {br['file'][:18]}<br>{br['error'][:40]}
                                </div>""", unsafe_allow_html=True)

# ── TAB 2: WEBCAM ─────────────────────────────────────────────────────────────
with sub_webcam:
    st.markdown(f'<div class="upload-hint">{T("webcam_hint")}</div>', unsafe_allow_html=True)

    # ── Auto-classify toggle ──────────────────────────────────────────────────
    auto_col1, auto_col2 = st.columns([2, 1])
    with auto_col1:
        auto_classify = st.toggle(
            "⚡ Auto-Classify Mode",
            value=False,
            help="When ON, every photo you take is automatically classified instantly — no button press needed."
        )
    with auto_col2:
        if auto_classify:
            st.markdown("""<div style="background:rgba(184,255,0,0.08);border:1px solid rgba(184,255,0,0.25);
                border-radius:10px;padding:0.4rem 0.8rem;font-size:0.75rem;color:#b8ff00;
                font-family:'Space Mono',monospace;text-align:center;">
                🟢 LIVE MODE ON
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div style="background:rgba(100,100,100,0.08);border:1px solid rgba(150,150,150,0.2);
                border-radius:10px;padding:0.4rem 0.8rem;font-size:0.75rem;color:#666;
                font-family:'Space Mono',monospace;text-align:center;">
                ⚪ MANUAL MODE
            </div>""", unsafe_allow_html=True)

    cam_col1, cam_col2 = st.columns([1, 1])
    with cam_col1:
        camera_image = st.camera_input("", label_visibility="collapsed")
    if camera_image:
        image = Image.open(camera_image).convert("RGB")
        with cam_col2:
            # Auto-classify fires immediately; manual mode needs a button press
            should_classify = auto_classify
            if not auto_classify:
                should_classify = st.button("🔍 Classify This Photo", use_container_width=True, key="cam_classify_btn")

            if should_classify:
                if use_tf:
                    with st.spinner("⚡ Running EfficientNetB0 locally…"):
                        try:
                            result = classify_with_tf(image)
                            st.markdown("""<div style="background:linear-gradient(135deg,rgba(200,241,53,0.07),rgba(0,0,0,0));
                                border:1px solid rgba(200,241,53,0.18);border-radius:12px;
                                padding:0.6rem 1rem;margin-bottom:0.8rem;font-size:0.78rem;
                                font-family:'DM Mono',monospace;color:#c8f135;">
                                ⚡ Classified by <strong>EfficientNetB0</strong> — pretrained on ImageNet (1000 classes)<br>
                                <span style="color:#9ab87a;font-size:0.7rem;">No API key required · Runs entirely on your server</span>
                            </div>""", unsafe_allow_html=True)
                            process_result(result)
                        except Exception as e:
                            st.error(f"❌ TensorFlow error: {str(e)}")
                elif not api_key:
                    st.warning("⚠️ Enter your **Gemini API key** in the sidebar, or switch to **Local TensorFlow** mode.")
                else:
                    with st.spinner("🔍 Analysing waste..."):
                        try:
                            result = classify_waste(image, api_key)
                            process_result(result)
                        except json.JSONDecodeError:
                            st.error("❌ Could not parse AI response. Please try again.")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
            else:
                st.markdown("""<div style="text-align:center;padding:2rem 1rem;color:#3a5628;
                    font-size:0.85rem;font-family:'Space Mono',monospace;">
                    📸 Photo captured — press <strong style="color:#b8ff00;">Classify This Photo</strong> to analyse it
                </div>""", unsafe_allow_html=True)

# ── TAB: MY STATS ─────────────────────────────────────────────────────────────
with tab_stats:
    sub_dash, sub_achieve, sub_leader = st.tabs([T("tab_dash"), T("tab_achieve"), T("tab_leader")])

# ── TAB 3: ACHIEVEMENTS ───────────────────────────────────────────────────────
with sub_achieve:
    # Level card
    level      = get_level(st.session_state.points)
    next_pts, prev_pts = get_next_level_points(st.session_state.points)
    progress   = min(100, int((st.session_state.points - prev_pts) / max(1, next_pts - prev_pts) * 100))

    lc1, lc2, lc3 = st.columns([1,2,1])
    with lc2:
        st.markdown(f"""<div class="level-card">
            <div style="font-size:0.75rem;color:#81c784;letter-spacing:0.1em;margin-bottom:0.3rem">CURRENT LEVEL</div>
            <div class="level-title">{level}</div>
            <div class="points-display" style="margin:0.5rem 0">⭐ {st.session_state.points} points</div>
            <div class="points-bar"><div class="points-fill" style="width:{progress}%"></div></div>
            <div style="font-size:0.75rem;color:#4caf50">{st.session_state.points} / {next_pts} pts to next level</div>
            <div style="margin-top:0.8rem">
                <span class="streak-fire">🔥</span>
                <span style="font-family:'Space Mono',monospace;color:#ffb74d;font-size:1rem"> {st.session_state.streak} day streak</span>
            </div>
            <div style="font-size:0.8rem;color:#81c784;margin-top:0.3rem">🔬 {st.session_state.total_scans} total scans</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🏅 Badges")

    cols = st.columns(4)
    for i, badge in enumerate(BADGES):
        unlocked = badge["id"] in st.session_state.badges
        status   = "unlocked" if unlocked else "locked"
        lock_txt = "" if unlocked else "🔒"
        with cols[i % 4]:
            st.markdown(f"""<div class="badge-card {status}">
                <div class="badge-icon">{badge['icon']}{lock_txt}</div>
                <div class="badge-name">{badge['name']}</div>
                <div class="badge-desc">{badge['desc']}</div>
            </div><br>""", unsafe_allow_html=True)

    # ── Streak Calendar ───────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📅 Scan Streak Calendar")
    st.markdown("<div style='font-size:0.78rem;color:var(--text-2);margin-bottom:0.8rem;'>Each green square = a day you scanned waste · Last 52 weeks</div>", unsafe_allow_html=True)
    st.markdown(f"""<div class="glass-card">{build_streak_calendar(st.session_state.auth_username)}</div>""", unsafe_allow_html=True)

# ── TAB 4: DASHBOARD ──────────────────────────────────────────────────────────
with sub_dash:
    history = st.session_state.history
    if not history:
        st.markdown("""<div class="empty-state">
            <div class="empty-state-icon">📭</div>
            <div class="empty-state-title">No scans yet</div>
            <div class="empty-state-sub">Head over to the <strong>Upload</strong> or <strong>Webcam</strong> tab and scan your first waste item to see your analytics here.</div>
        </div>""", unsafe_allow_html=True)
    else:
        df = pd.DataFrame(history)
        total    = len(df)
        avg_conf = int(df["confidence"].mean())
        top_cat  = df["category"].value_counts().idxmax()
        top_icon = DISPOSAL_TIPS[top_cat][0]
        unique   = df["category"].nunique()

        c1, c2, c3, c4 = st.columns(4)
        for col, num, label in zip(
            [c1, c2, c3, c4],
            [total, f"{avg_conf}%", top_icon, unique],
            ["Total Scans", "Avg Confidence", "Most Common", "Categories Found"]
        ):
            with col:
                st.markdown(f'<div class="stat-card"><div class="stat-number">{num}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        ch1, ch2 = st.columns(2)

        with ch1:
            st.markdown("#### 🥧 Waste Breakdown")
            cat_counts = df["category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            fig_pie = px.pie(cat_counts, values="Count", names="Category", hole=0.45,
                             color_discrete_sequence=[CATEGORY_COLORS.get(c, "#69f0ae") for c in cat_counts["Category"]])
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font_color="#c8e6c9", margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_pie, use_container_width=True)

        with ch2:
            st.markdown("#### 📈 Confidence per Scan")
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=list(range(1, len(df)+1)), y=df["confidence"],
                mode="lines+markers",
                line=dict(color="#69f0ae", width=2),
                marker=dict(color=[CATEGORY_COLORS.get(c, "#69f0ae") for c in df["category"]], size=10),
                hovertext=df["category"],
                hovertemplate="Scan %{x}<br>%{hovertext}<br>%{y}%<extra></extra>",
            ))
            fig_line.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   font_color="#c8e6c9",
                                   xaxis=dict(title="Scan #", gridcolor="#1b2e1c", color="#81c784"),
                                   yaxis=dict(title="Confidence %", gridcolor="#1b2e1c", color="#81c784", range=[0,105]),
                                   margin=dict(t=20, b=40, l=40, r=20))
            st.plotly_chart(fig_line, use_container_width=True)

        st.markdown("#### 📊 Scans per Category")
        cat_bar = df["category"].value_counts().reset_index()
        cat_bar.columns = ["Category", "Count"]
        fig_bar = px.bar(cat_bar, x="Category", y="Count", color="Category",
                         color_discrete_sequence=[CATEGORY_COLORS.get(c, "#69f0ae") for c in cat_bar["Category"]])
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#c8e6c9",
                               xaxis=dict(gridcolor="#1b2e1c", color="#81c784"),
                               yaxis=dict(gridcolor="#1b2e1c", color="#81c784"),
                               margin=dict(t=10, b=40, l=40, r=20), showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("#### 🕘 Scan History")
        for i, row in enumerate(reversed(history)):
            color = CATEGORY_COLORS.get(row["category"], "#69f0ae")
            src_tag = (
                '<span style="font-size:0.65rem;background:rgba(200,241,53,0.1);color:#c8f135;'
                'border:1px solid rgba(200,241,53,0.2);border-radius:6px;padding:1px 6px;margin-left:4px;">⚡ TF</span>'
                if row.get("source") == "tensorflow" else
                '<span style="font-size:0.65rem;background:rgba(0,232,200,0.1);color:#00e8c8;'
                'border:1px solid rgba(0,232,200,0.2);border-radius:6px;padding:1px 6px;margin-left:4px;">🌐 Gemini</span>'
            )
            st.markdown(f"""<div class="history-row">
                <span style="color:#81c784;font-size:0.78rem">#{len(history)-i}</span>
                <span style="font-size:1.2rem">{row['icon']}</span>
                <span style="font-family:'Space Mono',monospace;color:{color};font-weight:700">{row['category']}</span>
                <span style="color:#a5d6a7;font-size:0.85rem">{row['confidence']}% confidence</span>
                {src_tag}
                <span style="color:#4caf50;font-size:0.75rem">{row['time']}</span>
            </div>""", unsafe_allow_html=True)

        # ── Model Engine Stats ──
        tf_count  = sum(1 for h in history if h.get("source") == "tensorflow")
        gem_count = len(history) - tf_count
        if tf_count > 0 or gem_count > 0:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🧠 Classification Engine Usage")
            me1, me2 = st.columns(2)
            with me1:
                st.markdown(f"""<div class="stat-card" style="border-color:rgba(200,241,53,0.2);">
                    <div style="font-size:1.6rem;margin-bottom:0.3rem;">⚡</div>
                    <div class="stat-number" style="font-size:2rem;">{tf_count}</div>
                    <div class="stat-label">EfficientNetB0 (Local TF)</div>
                    <div style="font-size:0.72rem;color:var(--text-3);margin-top:0.3rem;">No API · ImageNet pretrained</div>
                </div>""", unsafe_allow_html=True)
            with me2:
                st.markdown(f"""<div class="stat-card" style="border-color:rgba(0,232,200,0.2);">
                    <div style="font-size:1.6rem;margin-bottom:0.3rem;">🌐</div>
                    <div class="stat-number" style="font-size:2rem;background:linear-gradient(135deg,#00e8c8,#c8f135);
                         -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">{gem_count}</div>
                    <div class="stat-label">Gemini 2.5 Flash (Cloud)</div>
                    <div style="font-size:0.72rem;color:var(--text-3);margin-top:0.3rem;">Multimodal · Highest accuracy</div>
                </div>""", unsafe_allow_html=True)

        # ── Weekly / Monthly Trends ───────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📅 Trends Over Time")
        trends_df = load_user_trends(st.session_state.auth_username)
        if trends_df.empty:
            st.markdown("""<div class="empty-state">
                <div class="empty-state-icon">📅</div>
                <div class="empty-state-title">Not enough data yet</div>
                <div class="empty-state-sub">Scan items over multiple days to see your weekly and monthly trend charts appear here.</div>
            </div>""", unsafe_allow_html=True)
        else:
            trend_period = st.radio(
                "View by", ["Daily", "Weekly", "Monthly"],
                horizontal=True, key="trend_period"
            )
            CAT_COLORS_TREND = {
                "Plastic": "#38bdf8", "Organic": "#4ade80",
                "Metal": "#fbbf24",   "Paper": "#f472b6", "E-waste": "#a78bfa"
            }
            if trend_period == "Daily":
                grp_df = trends_df.copy()
                grp_df["period"] = grp_df["scanned_date"].dt.strftime("%d %b")
            elif trend_period == "Weekly":
                grp_df = trends_df.copy()
                grp_df["period"] = grp_df["scanned_date"].dt.to_period("W").dt.start_time.dt.strftime("Wk %d %b")
            else:
                grp_df = trends_df.copy()
                grp_df["period"] = grp_df["scanned_date"].dt.strftime("%b %Y")

            pivot = grp_df.groupby(["period", "category"])["count"].sum().reset_index()
            # Stacked bar chart
            fig_trend = px.bar(
                pivot, x="period", y="count", color="category",
                color_discrete_map=CAT_COLORS_TREND,
                labels={"period": "Period", "count": "Scans", "category": "Category"},
                barmode="stack",
            )
            fig_trend.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#c8e6c9",
                xaxis=dict(title="", gridcolor="#1b2e1c", color="#81c784"),
                yaxis=dict(title="Scans", gridcolor="#1b2e1c", color="#81c784"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=30, b=40, l=40, r=20),
            )
            st.plotly_chart(fig_trend, use_container_width=True)

            # Line chart: total scans per period
            total_by_period = pivot.groupby("period")["count"].sum().reset_index()
            fig_tline = go.Figure()
            fig_tline.add_trace(go.Scatter(
                x=total_by_period["period"], y=total_by_period["count"],
                mode="lines+markers",
                line=dict(color="#b8ff00", width=2),
                marker=dict(size=8, color="#b8ff00"),
                fill="tozeroy", fillcolor="rgba(184,255,0,0.06)",
                name="Total Scans",
            ))
            fig_tline.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#c8e6c9",
                xaxis=dict(title="", gridcolor="#1b2e1c", color="#81c784"),
                yaxis=dict(title="Total Scans", gridcolor="#1b2e1c", color="#81c784", rangemode="tozero"),
                margin=dict(t=10, b=40, l=40, r=20), showlegend=False,
            )
            st.plotly_chart(fig_tline, use_container_width=True)

# ── TAB: EXPLORE ──────────────────────────────────────────────────────────────
with tab_explore:
    sub_recycle, sub_impact, sub_suggest = st.tabs([T("tab_recycle"), T("tab_impact"), T("tab_suggest")])

# ── TAB 5: RECYCLING CENTERS ─────────────────────────────────────────────────
with sub_recycle:
    st.markdown("""<div style="text-align:center;padding:1.2rem 0 0.5rem;">
        <div style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;color:var(--emerald);letter-spacing:-0.02em;">📍 Find Recycling Centers Near You</div>
        <div style="color:var(--text-muted);font-size:0.85rem;margin-top:0.4rem;letter-spacing:0.05em;">Opens Google Maps instantly · No sign-in required · Always free</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    CENTER_DATA = [
        {
            "category": "Plastic",
            "icon": "🧴",
            "color": "#60b4ff",
            "label": "Plastic Recycling Centers",
            "desc": "Drop off bottles, containers, bags & plastic packaging",
            "query": "plastic+recycling+center+near+me",
            "tips": "Rinse containers before dropping off. Check local rules on black plastic.",
        },
        {
            "category": "Organic",
            "icon": "🌿",
            "color": "#6dffb0",
            "label": "Compost & Organic Drop-offs",
            "desc": "Food scraps, garden waste, biodegradable materials",
            "query": "compost+drop+off+near+me",
            "tips": "Separate wet and dry organic waste for faster composting.",
        },
        {
            "category": "Metal",
            "icon": "🔩",
            "color": "#ffc96d",
            "label": "Scrap Metal Recyclers",
            "desc": "Cans, appliances, wires, aluminium & ferrous metals",
            "query": "scrap+metal+recycling+near+me",
            "tips": "Aluminium cans can be recycled indefinitely — always worth dropping off.",
        },
        {
            "category": "Paper",
            "icon": "📄",
            "color": "#ff93c4",
            "label": "Paper Recycling Centers",
            "desc": "Newspapers, cardboard, office paper & magazines",
            "query": "paper+recycling+center+near+me",
            "tips": "Keep paper dry. Shredded paper can go to compost instead.",
        },
        {
            "category": "E-waste",
            "icon": "⚡",
            "color": "#c893ff",
            "label": "E-Waste Drop-off Centers",
            "desc": "Phones, laptops, batteries, cables & electronics",
            "query": "e-waste+drop+off+center+near+me",
            "tips": "Never bin e-waste — it contains toxic materials that leach into soil.",
        },
    ]

    col_a, col_b = st.columns(2)
    for i, item in enumerate(CENTER_DATA):
        col = col_a if i % 2 == 0 else col_b
        maps_url = f"https://www.google.com/maps/search/{item['query']}"
        with col:
            st.markdown(f"""<div class="glass-card" style="border-color:{item['color']}22;margin-bottom:0.8rem;">
                <div style="position:absolute;top:0;left:0;right:0;height:2px;
                     background:linear-gradient(90deg,transparent,{item['color']},transparent);
                     border-radius:20px 20px 0 0;"></div>
                <div style="display:flex;align-items:center;gap:0.7rem;margin-bottom:0.8rem;">
                    <span style="font-size:2rem;line-height:1;">{item['icon']}</span>
                    <div>
                        <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:0.95rem;
                             color:{item['color']};letter-spacing:0.01em;">{item['label']}</div>
                        <div style="font-size:0.78rem;color:var(--text-muted);margin-top:0.1rem;">{item['desc']}</div>
                    </div>
                </div>
                <div style="background:rgba(0,0,0,0.2);border-radius:10px;padding:0.6rem 0.8rem;
                     margin-bottom:0.9rem;font-size:0.78rem;color:#8fc99b;line-height:1.5;">
                    💡 {item['tips']}
                </div>
                <a href="{maps_url}" target="_blank" class="recycle-btn"
                   style="border-color:{item['color']}55;color:{item['color']} !important;width:100%;justify-content:center;">
                    🗺️ Find {item['category']} Centers →
                </a>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # General search button
    gc1, gc2, gc3 = st.columns([1, 2, 1])
    with gc2:
        st.markdown("""<div style="text-align:center;background:var(--glass-bg);border:1px solid var(--glass-border);
             border-radius:16px;padding:1.4rem 1.2rem;">
            <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;
                 color:var(--emerald);margin-bottom:0.4rem;">♻️ Search All Recycling Centers</div>
            <div style="font-size:0.8rem;color:var(--text-muted);margin-bottom:1rem;">
                Not sure what category? Search for any recycling facility nearby.
            </div>
            <a href="https://www.google.com/maps/search/recycling+center+near+me"
               target="_blank" class="recycle-btn" style="justify-content:center;font-size:0.9rem;padding:0.75rem 1.8rem;">
                🗺️ Find Any Recycling Center →
            </a>
        </div>""", unsafe_allow_html=True)

# ── TAB 6: LEADERBOARD ────────────────────────────────────────────────────────
with sub_leader:
    st.markdown("""<div style="text-align:center;padding:1.2rem 0 0.5rem;">
        <div style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;
             color:var(--emerald);letter-spacing:-0.02em;">🏅 Weekly Leaderboard</div>
        <div style="color:var(--text-muted);font-size:0.85rem;margin-top:0.4rem;">
            Top recyclers this week · Updated in real time
        </div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Build leaderboard from real DB — all registered users
    db_board = load_leaderboard(limit=50)
    current_name = st.session_state.username
    board = []
    current_user_in_board = False
    for u in db_board:
        is_you = (u["name"] == current_name)
        if is_you:
            current_user_in_board = True
        board.append(dict(u, is_you=is_you))
    # Ensure the current user always appears even if not yet in DB result
    if not current_user_in_board:
        board.append({
            "name": current_name,
            "points": st.session_state.points,
            "scans": st.session_state.total_scans,
            "streak": st.session_state.streak,
            "badge": _badge_for_points(st.session_state.points),
            "is_you": True,
        })
    board = sorted(board, key=lambda x: x["points"], reverse=True)

    # Top 3 podium
    podium_cols = st.columns(3)
    podium_order = [1, 0, 2]  # silver, gold, bronze positions
    podium_colors = ["#c0c0c0", "#ffd700", "#cd7f32"]
    podium_labels = ["🥈 2nd", "🥇 1st", "🥉 3rd"]
    for pi, idx in enumerate(podium_order):
        if idx < len(board):
            u = board[idx]
            is_you = u.get("is_you", False)
            you_tag = ' <span style="font-size:0.65rem;background:#00ffaa22;color:#00ffaa;padding:2px 6px;border-radius:6px;">YOU</span>' if is_you else ''
            with podium_cols[pi]:
                st.markdown(f"""<div class="glass-card" style="text-align:center;border-color:{podium_colors[pi]}44;
                     {'box-shadow:0 0 30px '+podium_colors[pi]+'33;' if pi==1 else ''}padding:1.4rem 1rem;">
                    <div style="font-size:2.2rem;margin-bottom:0.3rem;">{u['badge']}</div>
                    <div style="font-family:'Syne',sans-serif;font-size:0.9rem;font-weight:700;
                         color:{podium_colors[pi]};margin-bottom:0.2rem;">{podium_labels[pi]}</div>
                    <div style="font-size:0.85rem;color:var(--text-primary);font-weight:600;margin-bottom:0.5rem;">
                        {u['name'][:16]}{you_tag}
                    </div>
                    <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;
                         color:{podium_colors[pi]};text-shadow:0 0 20px {podium_colors[pi]}55;">
                        {u['points']}
                    </div>
                    <div style="font-size:0.72rem;color:var(--text-muted);letter-spacing:0.08em;">POINTS</div>
                    <div style="font-size:0.75rem;color:var(--text-muted);margin-top:0.4rem;">
                        🔬 {u['scans']} scans &nbsp;·&nbsp; 🔥 {u['streak']}d
                    </div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📋 Full Rankings")

    # Full rankings table
    for rank, u in enumerate(board, 1):
        is_you = u.get("is_you", False)
        bg     = "rgba(0,255,170,0.08)" if is_you else "rgba(10,22,12,0.6)"
        border = "rgba(0,255,170,0.35)" if is_you else "rgba(0,255,170,0.08)"
        medal  = {1:"🥇",2:"🥈",3:"🥉"}.get(rank, f"#{rank}")
        you_tag = ' <span style="font-size:0.65rem;background:#00ffaa22;color:#00ffaa;padding:2px 5px;border-radius:5px;margin-left:4px;">YOU</span>' if is_you else ''
        bar_w  = min(100, int(u["points"] / max(board[0]["points"], 1) * 100))
        st.markdown(f"""<div style="background:{bg};border:1px solid {border};border-radius:12px;
             padding:0.75rem 1.1rem;margin:0.3rem 0;">
            <div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.4rem;">
                <span style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;
                      color:var(--emerald);min-width:2rem;">{medal}</span>
                <span style="font-size:1.2rem;">{u['badge']}</span>
                <span style="flex:1;font-weight:600;color:var(--text-primary);font-size:0.88rem;">
                    {u['name'][:22]}{you_tag}
                </span>
                <span style="font-family:'Syne',sans-serif;font-weight:800;color:var(--emerald);font-size:1rem;">
                    {u['points']} pts
                </span>
                <span style="font-size:0.75rem;color:var(--text-muted);min-width:4rem;text-align:right;">
                    🔬{u['scans']} 🔥{u['streak']}d
                </span>
            </div>
            <div style="background:rgba(0,255,170,0.06);border-radius:4px;height:4px;overflow:hidden;">
                <div style="width:{bar_w}%;background:linear-gradient(90deg,#00c87a,#00ffaa);
                     height:100%;border-radius:4px;"></div>
            </div>
        </div>""", unsafe_allow_html=True)

# ── TAB: REPORTS ──────────────────────────────────────────────────────────────
with tab_reports:
    sub_report, sub_email = st.tabs([T("tab_report"), "📧 Email Alerts"])

# ── TAB 7: DOWNLOAD REPORT ────────────────────────────────────────────────────
with sub_report:
    st.markdown("""<div style="text-align:center;padding:1.2rem 0 0.5rem;">
        <div style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;
             color:var(--emerald);letter-spacing:-0.02em;">📄 Download Your Report</div>
        <div style="color:var(--text-muted);font-size:0.85rem;margin-top:0.4rem;">
            Export your full recycling activity as a PDF
        </div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    rc1, rc2, rc3 = st.columns([1, 2, 1])
    with rc2:
        level_now = get_level(st.session_state.points)

        # Preview card
        st.markdown(f"""<div class="glass-card" style="text-align:center;">
            <div style="font-size:2.5rem;margin-bottom:0.6rem;">📊</div>
            <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.1rem;
                 color:var(--emerald);margin-bottom:1rem;">Report Preview</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;text-align:left;margin-bottom:1.2rem;">
                <div style="background:rgba(0,255,170,0.06);border-radius:10px;padding:0.7rem;">
                    <div style="font-size:0.7rem;color:var(--text-muted);letter-spacing:0.08em;">USER</div>
                    <div style="font-weight:600;color:var(--text-primary);font-size:0.9rem;">
                        {st.session_state.username}
                    </div>
                </div>
                <div style="background:rgba(0,255,170,0.06);border-radius:10px;padding:0.7rem;">
                    <div style="font-size:0.7rem;color:var(--text-muted);letter-spacing:0.08em;">LEVEL</div>
                    <div style="font-weight:600;color:var(--text-primary);font-size:0.9rem;">{level_now}</div>
                </div>
                <div style="background:rgba(0,255,170,0.06);border-radius:10px;padding:0.7rem;">
                    <div style="font-size:0.7rem;color:var(--text-muted);letter-spacing:0.08em;">TOTAL POINTS</div>
                    <div style="font-weight:700;color:var(--emerald);font-size:1.1rem;">
                        {st.session_state.points} ⭐
                    </div>
                </div>
                <div style="background:rgba(0,255,170,0.06);border-radius:10px;padding:0.7rem;">
                    <div style="font-size:0.7rem;color:var(--text-muted);letter-spacing:0.08em;">TOTAL SCANS</div>
                    <div style="font-weight:700;color:var(--emerald);font-size:1.1rem;">
                        {st.session_state.total_scans} 🔬
                    </div>
                </div>
                <div style="background:rgba(0,255,170,0.06);border-radius:10px;padding:0.7rem;">
                    <div style="font-size:0.7rem;color:var(--text-muted);letter-spacing:0.08em;">STREAK</div>
                    <div style="font-weight:700;color:#ffc96d;font-size:1.1rem;">
                        {st.session_state.streak} days 🔥
                    </div>
                </div>
                <div style="background:rgba(0,255,170,0.06);border-radius:10px;padding:0.7rem;">
                    <div style="font-size:0.7rem;color:var(--text-muted);letter-spacing:0.08em;">BADGES</div>
                    <div style="font-weight:700;color:#c893ff;font-size:1.1rem;">
                        {len(st.session_state.badges)} 🏅
                    </div>
                </div>
            </div>
            <div style="font-size:0.78rem;color:var(--text-muted);margin-bottom:1rem;">
                Includes: summary · category breakdown · scan history (last 10)
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("⬇️ Generate & Download PDF Report", use_container_width=True):
            if not st.session_state.history:
                st.warning("⚠️ No scan history yet! Do some scans first, then download your report.")
            else:
                with st.spinner("📄 Generating your PDF report..."):
                    pdf_buf = generate_pdf_report(
                        username    = st.session_state.username,
                        points      = st.session_state.points,
                        streak      = st.session_state.streak,
                        total_scans = st.session_state.total_scans,
                        badges      = st.session_state.badges,
                        history     = st.session_state.history,
                        level       = level_now,
                    )
                fname = f"SmartWasteAI_Report_{st.session_state.username}_{datetime.date.today()}.pdf"
                st.download_button(
                    label    = "📥 Click to Save PDF",
                    data     = pdf_buf,
                    file_name= fname,
                    mime     = "application/pdf",
                    use_container_width=True,
                )
                st.success("✅ Report ready! Click the button above to save it.")

# ── TAB 8: IMPACT CALCULATOR ─────────────────────────────────────────────────
with sub_impact:
    st.markdown("""<div style="text-align:center;padding:1.2rem 0 0.5rem;">
        <div style="font-family:'Cabinet Grotesk',sans-serif;font-size:1.8rem;font-weight:900;
             background:linear-gradient(135deg,#c8f135,#00e8c8);-webkit-background-clip:text;
             -webkit-text-fill-color:transparent;background-clip:text;letter-spacing:-0.02em;">
            🌍 Your Environmental Impact
        </div>
        <div style="color:var(--text-2);font-size:0.85rem;margin-top:0.5rem;">
            Every scan counts — see the real-world difference you're making
        </div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Impact data per category (per item recycled) ──────────────────────────
    IMPACT_DATA = {
        "Plastic": {
            "co2_kg":    0.5,   # kg CO2 saved per item
            "water_l":   1.5,   # litres of water saved
            "energy_kj": 5.4,   # kJ energy saved
            "land_m2":   0.002, # m2 of landfill space saved
        },
        "Organic": {
            "co2_kg":    0.3,
            "water_l":   0.8,
            "energy_kj": 2.1,
            "land_m2":   0.003,
        },
        "Metal": {
            "co2_kg":    1.8,
            "water_l":   4.0,
            "energy_kj": 14.0,
            "land_m2":   0.001,
        },
        "Paper": {
            "co2_kg":    0.9,
            "water_l":   10.0,
            "energy_kj": 4.2,
            "land_m2":   0.004,
        },
        "E-waste": {
            "co2_kg":    2.5,
            "water_l":   0.5,
            "energy_kj": 20.0,
            "land_m2":   0.0005,
        },
    }

    history = st.session_state.history

    if not history:
        st.markdown("""<div class="card" style="text-align:center;padding:2.5rem;">
            <div style="font-size:3rem;margin-bottom:0.8rem;">🌱</div>
            <div style="font-family:'Cabinet Grotesk',sans-serif;font-size:1.1rem;font-weight:700;color:var(--lime);margin-bottom:0.4rem;">
                No impact yet
            </div>
            <div style="color:var(--text-2);font-size:0.88rem;">
                Start scanning waste items to see your environmental impact grow!
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        # Calculate cumulative impact
        total_co2   = sum(IMPACT_DATA.get(h["category"], {}).get("co2_kg",   0) for h in history)
        total_water = sum(IMPACT_DATA.get(h["category"], {}).get("water_l",  0) for h in history)
        total_energy= sum(IMPACT_DATA.get(h["category"], {}).get("energy_kj",0) for h in history)
        total_land  = sum(IMPACT_DATA.get(h["category"], {}).get("land_m2",  0) for h in history)
        total_items = len(history)

        # ── Hero impact number ──
        st.markdown(f"""<div class="card" style="text-align:center;background:linear-gradient(160deg,rgba(8,18,8,0.95),rgba(13,22,13,0.98));border-color:rgba(200,241,53,0.2);">
            <div style="font-size:0.7rem;color:var(--text-2);letter-spacing:0.18em;text-transform:uppercase;font-family:'DM Mono',monospace;margin-bottom:0.6rem;">
                Total Items Recycled
            </div>
            <div style="font-family:'Cabinet Grotesk',sans-serif;font-size:5rem;font-weight:900;line-height:1;
                 background:linear-gradient(135deg,#c8f135,#00e8c8);-webkit-background-clip:text;
                 -webkit-text-fill-color:transparent;background-clip:text;filter:drop-shadow(0 0 30px rgba(200,241,53,0.3));">
                {total_items}
            </div>
            <div style="font-size:0.85rem;color:var(--text-2);margin-top:0.5rem;">items correctly identified & sorted</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div style="font-family:'Cabinet Grotesk',sans-serif;font-size:1rem;font-weight:700;
             color:var(--lime);letter-spacing:0.02em;margin-bottom:0.5rem;">
             ♻ Environmental Savings
        </div>""", unsafe_allow_html=True)

        # ── 4 impact metric cards ──
        ic1, ic2, ic3, ic4 = st.columns(4)
        metrics = [
            (ic1, "🌫️", f"{total_co2:.2f}", "kg CO₂", "Carbon Emissions Avoided", "#c8f135"),
            (ic2, "💧", f"{total_water:.1f}", "Litres", "Water Conserved", "#00e8c8"),
            (ic3, "⚡", f"{total_energy:.1f}", "kJ", "Energy Saved", "#ffb800"),
            (ic4, "🏔️", f"{total_land*1000:.2f}", "dm²", "Landfill Space Saved", "#bb88ff"),
        ]
        for col, icon, num, unit, label, clr in metrics:
            with col:
                st.markdown(f"""<div class="impact-card" style="border-color:{clr}22;">
                    <div style="position:absolute;top:0;left:0;right:0;height:2px;
                         background:linear-gradient(90deg,transparent,{clr},transparent);"></div>
                    <div class="impact-icon">{icon}</div>
                    <div class="impact-number" style="background:linear-gradient(135deg,{clr},#00e8c8);
                         -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
                        {num}
                    </div>
                    <div class="impact-unit" style="color:{clr}bb;">{unit}</div>
                    <div class="impact-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        # ── Real-world equivalents ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div style="font-family:'Cabinet Grotesk',sans-serif;font-size:1rem;font-weight:700;
             color:var(--lime);letter-spacing:0.02em;margin-bottom:0.5rem;">
             🌿 What This Means in Real Life
        </div>""", unsafe_allow_html=True)

        trees_equiv   = total_co2 / 21        # avg tree absorbs 21kg CO2/year
        showers_equiv = total_water / 60      # avg shower = 60L
        phone_charges = total_energy / 18.5   # avg phone charge = 18.5kJ
        plastic_bags  = total_items * 0.6

        eq1, eq2, eq3, eq4 = st.columns(4)
        equivalents = [
            (eq1, "🌳", f"{trees_equiv:.2f}", "Trees", "working for a year to absorb the same CO₂"),
            (eq2, "🚿", f"{showers_equiv:.1f}", "Showers", "worth of water saved from going to waste"),
            (eq3, "📱", f"{phone_charges:.0f}", "Phone charges", "worth of energy conserved"),
            (eq4, "🛍️", f"{plastic_bags:.0f}", "Plastic bags", "kept out of landfill"),
        ]
        for col, icon, num, label, desc in equivalents:
            with col:
                st.markdown(f"""<div class="card" style="text-align:center;padding:1.1rem 0.8rem;">
                    <div style="font-size:1.8rem;margin-bottom:0.3rem;">{icon}</div>
                    <div style="font-family:'Cabinet Grotesk',sans-serif;font-size:1.6rem;font-weight:900;
                         color:var(--lime);line-height:1.1;">{num}</div>
                    <div style="font-size:0.78rem;font-weight:700;color:var(--text);margin:0.2rem 0;">{label}</div>
                    <div style="font-size:0.68rem;color:var(--text-2);line-height:1.4;">{desc}</div>
                </div>""", unsafe_allow_html=True)

        # ── Per-category breakdown ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div style="font-family:'Cabinet Grotesk',sans-serif;font-size:1rem;font-weight:700;
             color:var(--lime);letter-spacing:0.02em;margin-bottom:0.5rem;">
             📊 Impact by Category
        </div>""", unsafe_allow_html=True)

        from collections import Counter
        cat_counts = Counter(h["category"] for h in history)
        CAT_COLORS = {"Plastic":"#5db8ff","Organic":"#6fffa0","Metal":"#ffcc55","Paper":"#ff88bb","E-waste":"#bb88ff"}

        for cat, count in cat_counts.most_common():
            d = IMPACT_DATA.get(cat, {})
            clr = CAT_COLORS.get(cat, "#c8f135")
            icon_c = {"Plastic":"🧴","Organic":"🌿","Metal":"🔩","Paper":"📄","E-waste":"⚡"}.get(cat,"♻️")
            co2_cat   = d.get("co2_kg",0)   * count
            water_cat = d.get("water_l",0)  * count
            pct = int(count / total_items * 100)
            st.markdown(f"""<div class="history-row" style="gap:1rem;">
                <span style="font-size:1.3rem;min-width:1.5rem;">{icon_c}</span>
                <span style="font-family:'Cabinet Grotesk',sans-serif;font-weight:700;
                      color:{clr};min-width:80px;">{cat}</span>
                <span style="flex:1;background:rgba(200,241,53,0.06);border-radius:6px;height:6px;overflow:hidden;">
                    <span style="display:block;width:{pct}%;background:{clr};height:100%;border-radius:6px;"></span>
                </span>
                <span style="font-size:0.82rem;color:var(--text-2);min-width:60px;text-align:right;">
                    {count} items
                </span>
                <span style="font-size:0.78rem;color:var(--lime);min-width:100px;text-align:right;font-family:'DM Mono',monospace;">
                    -{co2_cat:.2f}kg CO₂
                </span>
            </div>""", unsafe_allow_html=True)

        # ── Share card ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""<div class="card" style="text-align:center;
             background:linear-gradient(135deg,rgba(200,241,53,0.06),rgba(0,232,200,0.04));
             border-color:rgba(200,241,53,0.2);">
            <div style="font-size:1.5rem;margin-bottom:0.5rem;">🌍</div>
            <div style="font-family:'Cabinet Grotesk',sans-serif;font-size:1rem;font-weight:800;
                 color:var(--lime);margin-bottom:0.3rem;">Keep Going, {st.session_state.username}!</div>
            <div style="font-size:0.85rem;color:var(--text-2);line-height:1.6;">
                You've already saved <strong style="color:var(--lime)">{total_co2:.2f} kg of CO₂</strong> and
                <strong style="color:#00e8c8">{total_water:.1f} litres of water</strong>.<br>
                Every item you correctly sort makes a real difference to our planet. 🌱
            </div>
        </div>""", unsafe_allow_html=True)

# ── TAB 9: AI SUGGESTIONS & COACH ────────────────────────────────────────────
with sub_suggest:
    st.markdown("""<div style="text-align:center;padding:1.2rem 0 0.5rem;">
        <div style="font-family:'Cabinet Grotesk',sans-serif;font-size:1.8rem;font-weight:900;
             background:linear-gradient(135deg,#c8f135,#00e8c8);-webkit-background-clip:text;
             -webkit-text-fill-color:transparent;background-clip:text;letter-spacing:-0.02em;">
            🤖 AI Sustainability Coach
        </div>
        <div style="color:var(--text-2);font-size:0.85rem;margin-top:0.5rem;">
            Personalised suggestions · Upcycle ideas · Ask anything about waste & recycling
        </div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 1: Last scan suggestions ──────────────────────────────────────
    if st.session_state.last_suggestions and st.session_state.last_suggest_cat:
        cat  = st.session_state.last_suggest_cat
        sugg = st.session_state.last_suggestions
        clr  = CATEGORY_COLORS.get(cat, "#c8f135")
        icon_c = {"Plastic":"🧴","Organic":"🌿","Metal":"🔩","Paper":"📄","E-waste":"⚡"}.get(cat,"♻️")

        st.markdown(f"""<div style="font-family:'Cabinet Grotesk',sans-serif;font-size:1rem;
             font-weight:800;color:var(--lime);margin-bottom:0.8rem;">
             {icon_c} Suggestions for your last scan: <span style="color:{clr};">{cat}</span>
        </div>""", unsafe_allow_html=True)

        sc1, sc2, sc3 = st.columns(3)

        with sc1:
            reuse_items = "".join(
                f'<li style="margin-bottom:0.45rem;line-height:1.5;">{t}</li>'
                for t in sugg["reuse"]
            )
            st.markdown(f"""<div class="suggest-card" style="height:100%;">
                <div class="suggest-tag" style="color:#00e8c8;border-color:#00e8c830;">♻️ REUSE IDEAS</div>
                <ul style="margin:0;padding-left:1.1rem;font-size:0.84rem;color:var(--text);">
                    {reuse_items}
                </ul>
            </div>""", unsafe_allow_html=True)

        with sc2:
            upcycle_items = "".join(
                f'<li style="margin-bottom:0.45rem;line-height:1.5;">{t}</li>'
                for t in sugg["upcycle"]
            )
            st.markdown(f"""<div class="suggest-card" style="height:100%;">
                <div class="suggest-tag" style="color:#c8f135;border-color:#c8f13530;">✨ UPCYCLE IDEAS</div>
                <ul style="margin:0;padding-left:1.1rem;font-size:0.84rem;color:var(--text);">
                    {upcycle_items}
                </ul>
            </div>""", unsafe_allow_html=True)

        with sc3:
            st.markdown(f"""<div class="suggest-card" style="height:100%;">
                <div class="suggest-tag" style="color:#ffb800;border-color:#ffb80030;">🌍 YOUR IMPACT</div>
                <div style="font-size:0.84rem;color:var(--text);line-height:1.6;margin-bottom:0.8rem;">{sugg["impact"]}</div>
                <div class="suggest-tag" style="color:#bb88ff;border-color:#bb88ff30;">💡 DID YOU KNOW?</div>
                <div style="font-size:0.84rem;color:var(--text);line-height:1.6;">{sugg["did_you_know"]}</div>
            </div>""", unsafe_allow_html=True)

        # Fun fact
        st.markdown(f"""<div style="background:rgba(255,184,0,0.05);border:1px solid rgba(255,184,0,0.18);
            border-radius:12px;padding:0.8rem 1.2rem;margin:0.8rem 0;
            font-size:0.84rem;color:#ffb800;line-height:1.6;">
            🎯 <strong>Fun Fact:</strong> {sugg["fun_fact"]}
        </div>""", unsafe_allow_html=True)

        # Gemini extra tips
        if sugg.get("extra_tips"):
            st.markdown(f"""<div class="suggest-card" style="border-color:rgba(200,241,53,0.25);
                background:linear-gradient(135deg,rgba(200,241,53,0.05),rgba(0,232,200,0.03));">
                <div class="suggest-tag" style="color:#c8f135;border-color:#c8f13540;">
                    🌐 GEMINI AI · ITEM-SPECIFIC TIPS
                </div>
                <div style="display:flex;flex-direction:column;gap:0.5rem;">
                    {"".join(f'<div style="display:flex;gap:0.6rem;align-items:flex-start;font-size:0.85rem;color:var(--text);line-height:1.5;"><span style="color:#c8f135;margin-top:2px;">✦</span><span>{t}</span></div>' for t in sugg["extra_tips"])}
                </div>
                {f'<div style="margin-top:0.9rem;padding:0.6rem 0.9rem;background:rgba(200,241,53,0.07);border-radius:9px;font-size:0.82rem;color:#c8f135;"><strong>📍 Take Action Today:</strong> {sugg["local_action"]}</div>' if sugg.get("local_action") else ""}
                {f'<div style="margin-top:0.6rem;font-size:0.8rem;color:#9ab87a;font-style:italic;">"{sugg["motivational"]}"</div>' if sugg.get("motivational") else ""}
            </div>""", unsafe_allow_html=True)

        # Regenerate button
        rg1, rg2, rg3 = st.columns([1,1,1])
        with rg2:
            if st.button("🔄 Regenerate Suggestions", use_container_width=True):
                with st.spinner("🤖 Generating fresh suggestions…"):
                    new_sugg = get_ai_suggestions(cat, "", api_key)
                    st.session_state.last_suggestions = new_sugg
                st.rerun()

    else:
        st.markdown("""<div class="card" style="text-align:center;padding:2.5rem;">
            <div style="font-size:3rem;margin-bottom:0.8rem;">🤖</div>
            <div style="font-family:'Cabinet Grotesk',sans-serif;font-size:1.1rem;font-weight:700;
                 color:var(--lime);margin-bottom:0.4rem;">No suggestions yet</div>
            <div style="color:var(--text-2);font-size:0.88rem;">
                Scan a waste item in the <strong>Upload</strong> or <strong>Webcam</strong> tab first —
                personalised suggestions will appear here automatically!
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Section 2: Category Quick Suggestions ─────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div style="font-family:'Cabinet Grotesk',sans-serif;font-size:1rem;font-weight:800;
         color:var(--lime);margin-bottom:0.8rem;">📚 Browse Tips by Category</div>""",
        unsafe_allow_html=True)

    browse_cat = st.selectbox(
        "Choose a category",
        ["Plastic", "Organic", "Metal", "Paper", "E-waste"],
        label_visibility="collapsed"
    )
    if st.button(f"💡 Get {browse_cat} Suggestions", use_container_width=False):
        with st.spinner("🤖 Loading suggestions…"):
            browse_sugg = get_ai_suggestions(browse_cat, f"General {browse_cat} waste", api_key)
        bc1, bc2 = st.columns(2)
        with bc1:
            reuse_html = "".join(f'<li style="margin-bottom:0.4rem;line-height:1.5;">{t}</li>' for t in browse_sugg["reuse"])
            st.markdown(f"""<div class="suggest-card">
                <div class="suggest-tag" style="color:#00e8c8;border-color:#00e8c830;">♻️ REUSE</div>
                <ul style="margin:0;padding-left:1.1rem;font-size:0.84rem;color:var(--text);">{reuse_html}</ul>
            </div>""", unsafe_allow_html=True)
            st.markdown(f"""<div style="background:rgba(255,184,0,0.05);border:1px solid rgba(255,184,0,0.18);
                border-radius:12px;padding:0.8rem 1.1rem;font-size:0.83rem;color:#ffb800;line-height:1.6;">
                🎯 <strong>Fun Fact:</strong> {browse_sugg["fun_fact"]}
            </div>""", unsafe_allow_html=True)
        with bc2:
            upcycle_html = "".join(f'<li style="margin-bottom:0.4rem;line-height:1.5;">{t}</li>' for t in browse_sugg["upcycle"])
            st.markdown(f"""<div class="suggest-card">
                <div class="suggest-tag" style="color:#c8f135;border-color:#c8f13530;">✨ UPCYCLE</div>
                <ul style="margin:0;padding-left:1.1rem;font-size:0.84rem;color:var(--text);">{upcycle_html}</ul>
            </div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="suggest-card">
                <div class="suggest-tag" style="color:#bb88ff;border-color:#bb88ff30;">💡 DID YOU KNOW?</div>
                <div style="font-size:0.84rem;color:var(--text);line-height:1.6;">{browse_sugg["did_you_know"]}</div>
                <div style="margin-top:0.7rem;font-size:0.84rem;color:#ffb800;line-height:1.6;">
                    🌍 <strong>Impact:</strong> {browse_sugg["impact"]}
                </div>
            </div>""", unsafe_allow_html=True)

    # ── Section 3: AI Coach Chat ───────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div style="font-family:'Cabinet Grotesk',sans-serif;font-size:1rem;font-weight:800;
         color:var(--lime);margin-bottom:0.5rem;">💬 Ask the AI Coach</div>
         <div style="color:var(--text-2);font-size:0.8rem;margin-bottom:0.8rem;">
             Ask anything about recycling, upcycling, sustainability, or how to dispose of specific items.
         </div>""", unsafe_allow_html=True)

    # Chat history display
    if st.session_state.chat_history:
        for chat in st.session_state.chat_history:
            st.markdown(f'<div class="chat-bubble-user">{chat["q"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-bubble-ai">{chat["a"]}</div>', unsafe_allow_html=True)

    # Quick question chips
    st.markdown("<div style='font-size:0.75rem;color:var(--text-3);margin-bottom:0.4rem;'>Quick questions:</div>", unsafe_allow_html=True)
    qc1, qc2, qc3, qc4 = st.columns(4)
    quick_questions = [
        "Can I recycle a greasy pizza box?",
        "How do I dispose of old batteries?",
        "What can I do with broken electronics?",
        "How to start composting at home?",
    ]
    for col, qq in zip([qc1, qc2, qc3, qc4], quick_questions):
        with col:
            if st.button(qq, use_container_width=True, key=f"qq_{qq[:10]}"):
                with st.spinner("🤖 Thinking…"):
                    ans = get_chat_response(qq, st.session_state.chat_history, api_key)
                st.session_state.chat_history.append({"q": qq, "a": ans})
                st.rerun()

    # Free-form input
    user_q = st.text_input(
        "Ask the AI Coach",
        placeholder="e.g. How do I recycle bubble wrap? Can I compost paper towels?",
        label_visibility="collapsed",
        key="chat_input"
    )
    ask_col, clear_col = st.columns([3, 1])
    with ask_col:
        if st.button("📨 Ask", use_container_width=True) and user_q.strip():
            with st.spinner("🤖 Thinking…"):
                ans = get_chat_response(user_q, st.session_state.chat_history, api_key)
            st.session_state.chat_history.append({"q": user_q, "a": ans})
            st.rerun()
    with clear_col:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    if not api_key:
        st.info("💡 Enter your Gemini API key in the sidebar to unlock the AI Coach chat and item-specific tips. Browse suggestions above work without a key!")

# ── TAB 10: EMAIL ALERTS ─────────────────────────────────────────────────────
with sub_email:
    st.markdown("""<div style="text-align:center;padding:1.2rem 0 0.5rem;">
        <div style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;
             color:var(--emerald);letter-spacing:-0.02em;">📧 Email Streak Reminders</div>
        <div style="color:var(--text-muted);font-size:0.85rem;margin-top:0.4rem;">
            Get an email reminder when your streak is about to break · Uses your own SMTP
        </div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    em_col1, em_col2, em_col3 = st.columns([1, 2, 1])
    with em_col2:
        es = load_email_settings(st.session_state.auth_username)

        st.markdown("""<div class="glass-card">
            <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;
                 color:var(--emerald);margin-bottom:1.2rem;">⚙️ SMTP Configuration</div>
        """, unsafe_allow_html=True)

        smtp_host = st.text_input("SMTP Host", value=es.get("smtp_host",""), placeholder="smtp.gmail.com")
        smtp_port = st.number_input("SMTP Port", value=es.get("smtp_port", 587), min_value=1, max_value=65535)
        smtp_user = st.text_input("SMTP Email (sender)", value=es.get("smtp_user",""), placeholder="you@gmail.com")
        smtp_pass = st.text_input("SMTP Password / App Password", value=es.get("smtp_pass",""), type="password",
                                   placeholder="App password (not your Google password)")
        notify_streak = st.toggle("🔥 Send reminder when streak may break", value=bool(es.get("notify_streak", True)))

        st.markdown("""<div style="font-size:0.75rem;color:var(--text-muted);margin:0.6rem 0 1rem;line-height:1.6;">
            💡 For Gmail: use an <strong>App Password</strong> (Google Account → Security → App passwords).
            Your password is stored locally in the SQLite DB — it never leaves your server.
        </div>""", unsafe_allow_html=True)

        if st.button("💾 Save Email Settings", use_container_width=True):
            save_email_settings(
                st.session_state.auth_username,
                smtp_host, int(smtp_port), smtp_user, smtp_pass, notify_streak
            )
            st.success("✅ Email settings saved!")

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Send reminder manually
        st.markdown("""<div class="glass-card">
            <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;
                 color:var(--emerald);margin-bottom:0.8rem;">📨 Send Test / Manual Reminder</div>
            <div style="font-size:0.82rem;color:var(--text-muted);margin-bottom:1rem;">
                Send a streak reminder to your registered email right now.
            </div>
        """, unsafe_allow_html=True)
        if st.button("📧 Send Streak Reminder Now", use_container_width=True):
            ok, msg = send_streak_reminder(st.session_state.auth_username)
            if ok:
                st.success(f"✅ {msg}")
            else:
                st.error(f"❌ {msg}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""<div style="background:rgba(0,255,170,0.04);border:1px solid rgba(0,255,170,0.12);
            border-radius:12px;padding:1rem 1.2rem;font-size:0.8rem;color:#8fc99b;line-height:1.7;margin-top:0.5rem;">
            <strong style="color:var(--emerald);">How it works:</strong><br>
            • A reminder is sent once per day if you haven't scanned yet<br>
            • The reminder shows your current streak and links back to the app<br>
            • "Last reminded" is tracked — you won't be spammed<br>
            • To automate daily reminders, run a cron job calling your Streamlit backend
        </div>""", unsafe_allow_html=True)

# ── TAB 11: ADMIN DASHBOARD ───────────────────────────────────────────────────
with tab_admin:
    st.markdown("""<div style="text-align:center;padding:1.2rem 0 0.5rem;">
        <div style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;
             color:var(--emerald);letter-spacing:-0.02em;">🔐 Admin Dashboard</div>
        <div style="color:var(--text-muted);font-size:0.85rem;margin-top:0.4rem;">
            Platform-wide analytics · Password protected
        </div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Admin auth ────────────────────────────────────────────────────────────
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False

    stored_admin_pw = get_admin_password()

    if not st.session_state.admin_authenticated:
        adm_col1, adm_col2, adm_col3 = st.columns([1, 2, 1])
        with adm_col2:
            st.markdown("""<div class="glass-card" style="text-align:center;">
                <div style="font-size:2.5rem;margin-bottom:0.8rem;">🔒</div>
                <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;
                     color:var(--emerald);margin-bottom:1.2rem;">Enter Admin Password</div>
            """, unsafe_allow_html=True)

            if not stored_admin_pw:
                st.info("ℹ️ No admin password set yet. Set one below to protect this dashboard.")
                new_pw  = st.text_input("Set Admin Password", type="password", key="new_admin_pw")
                new_pw2 = st.text_input("Confirm Password",   type="password", key="new_admin_pw2")
                if st.button("🔑 Set Password & Enter", use_container_width=True):
                    if not new_pw or len(new_pw) < 4:
                        st.error("Password must be at least 4 characters.")
                    elif new_pw != new_pw2:
                        st.error("Passwords do not match.")
                    else:
                        set_admin_password(new_pw)
                        st.session_state.admin_authenticated = True
                        st.rerun()
            else:
                adm_pw_input = st.text_input("Admin Password", type="password", key="adm_pw_input")
                if st.button("🔓 Enter Admin", use_container_width=True):
                    if hashlib.sha256(adm_pw_input.encode()).hexdigest() == stored_admin_pw:
                        st.session_state.admin_authenticated = True
                        st.rerun()
                    else:
                        st.error("❌ Incorrect password.")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        # ── Admin is logged in — show full dashboard ──────────────────────────
        if st.button("🔒 Lock Admin", key="admin_lock"):
            st.session_state.admin_authenticated = False
            st.rerun()

        astats = get_admin_stats()

        # KPI cards
        ak1, ak2, ak3, ak4, ak5 = st.columns(5)
        kpis = [
            (ak1, "👥", astats["total_users"],  "Total Users"),
            (ak2, "🔬", astats["total_scans"],  "Total Scans"),
            (ak3, "⭐", astats["total_points"], "Total Points"),
            (ak4, "🟢", astats["active_today"], "Active Today"),
            (ak5, "🆕", astats["new_users_7d"], "New (7d)"),
        ]
        for col, icon, val, label in kpis:
            with col:
                st.markdown(f"""<div class="stat-card">
                    <div style="font-size:1.5rem;margin-bottom:0.3rem;">{icon}</div>
                    <div class="stat-number" style="font-size:2rem;">{val:,}</div>
                    <div class="stat-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        adm_ch1, adm_ch2 = st.columns(2)

        with adm_ch1:
            st.markdown("#### 📅 Daily Scans (Last 30 Days)")
            if astats["daily_scans"]:
                days_df = pd.DataFrame(astats["daily_scans"], columns=["date", "scans"])
                fig_adm_line = go.Figure()
                fig_adm_line.add_trace(go.Bar(
                    x=days_df["date"], y=days_df["scans"],
                    marker_color="#00c87a", opacity=0.8, name="Scans"
                ))
                fig_adm_line.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#c8e6c9",
                    xaxis=dict(gridcolor="#1b2e1c", color="#81c784"),
                    yaxis=dict(gridcolor="#1b2e1c", color="#81c784"),
                    margin=dict(t=10, b=40, l=40, r=20), showlegend=False,
                )
                st.plotly_chart(fig_adm_line, use_container_width=True)
            else:
                st.markdown("""<div class="empty-state" style="padding:2rem;">
                    <div class="empty-state-icon">📊</div>
                    <div class="empty-state-title">No scan data yet</div>
                    <div class="empty-state-sub">Platform scans will appear here once users start classifying waste.</div>
                </div>""", unsafe_allow_html=True)

        with adm_ch2:
            st.markdown("#### 🥧 Platform Category Breakdown")
            if astats["cat_breakdown"]:
                cat_df = pd.DataFrame(astats["cat_breakdown"], columns=["Category", "Count"])
                ADMIN_COLORS = {"Plastic":"#38bdf8","Organic":"#4ade80","Metal":"#fbbf24","Paper":"#f472b6","E-waste":"#a78bfa"}
                fig_adm_pie = px.pie(
                    cat_df, values="Count", names="Category", hole=0.4,
                    color_discrete_sequence=[ADMIN_COLORS.get(c,"#69f0ae") for c in cat_df["Category"]]
                )
                fig_adm_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#c8e6c9", margin=dict(t=20,b=20,l=20,r=20)
                )
                st.plotly_chart(fig_adm_pie, use_container_width=True)
            else:
                st.markdown("""<div class="empty-state" style="padding:2rem;">
                    <div class="empty-state-icon">🥧</div>
                    <div class="empty-state-title">No categories yet</div>
                    <div class="empty-state-sub">Category breakdown will appear once users scan waste items.</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("#### 🏆 Top 10 Users")
        if astats["top_users"]:
            for i, u in enumerate(astats["top_users"], 1):
                medal = {1:"🥇",2:"🥈",3:"🥉"}.get(i, f"#{i}")
                bar_w = min(100, int(u["points"] / max(astats["top_users"][0]["points"], 1) * 100))
                st.markdown(f"""<div style="background:rgba(10,22,12,0.6);border:1px solid rgba(0,255,170,0.08);
                    border-radius:12px;padding:0.7rem 1.1rem;margin:0.25rem 0;">
                    <div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.3rem;">
                        <span style="font-size:1rem;font-weight:700;color:#00c87a;min-width:2rem;">{medal}</span>
                        <span style="flex:1;font-weight:600;font-size:0.88rem;">{u['display_name'][:24]}</span>
                        <span style="font-weight:800;color:#00c87a;">{u['points']:,} pts</span>
                        <span style="font-size:0.75rem;color:#4caf50;min-width:5rem;text-align:right;">
                            🔬{u['total_scans']} 🔥{u['streak']}d
                        </span>
                    </div>
                    <div style="background:rgba(0,255,170,0.06);border-radius:4px;height:4px;overflow:hidden;">
                        <div style="width:{bar_w}%;background:linear-gradient(90deg,#00c87a,#00ffaa);height:100%;border-radius:4px;"></div>
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="empty-state">
                <div class="empty-state-icon">👥</div>
                <div class="empty-state-title">No users yet</div>
                <div class="empty-state-sub">Users will appear here once they register and earn points.</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        # Change admin password
        with st.expander("🔑 Change Admin Password"):
            cp1 = st.text_input("New Password", type="password", key="chg_adm_pw1")
            cp2 = st.text_input("Confirm New Password", type="password", key="chg_adm_pw2")
            if st.button("Update Password", key="update_adm_pw"):
                if len(cp1) < 4:
                    st.error("Password must be at least 4 characters.")
                elif cp1 != cp2:
                    st.error("Passwords don't match.")
                else:
                    set_admin_password(cp1)
                    st.success("✅ Admin password updated.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<div class="footer">SMART WASTE AI &nbsp;·&nbsp; Powered by Google Gemini &nbsp;·&nbsp; ♻ Built with Streamlit</div>', unsafe_allow_html=True)
