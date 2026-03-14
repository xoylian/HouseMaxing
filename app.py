"""
app.py — House Check: AI-Powered Home Maintenance Spotter & Airbnb Room Stager
Run: streamlit run app.py
"""

import base64
import os
import io
import json
import time
from pathlib import Path
from typing import Optional

import streamlit as st
from PIL import Image
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# ─── Config ────────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME    = "house_check_rules"
EMBEDDING_MODEL    = "text-embedding-3-small"
VISION_MODEL       = "gpt-4.1"
REPORT_MODEL       = "gpt-4.1"
TOP_K_RULES        = 8
# ───────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="HouseMaxing — AI Επιθεώρηση Σπιτιού",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
  /* ── Google Fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

  /* ── Root Variables ── */
  :root {
    --bg:           #0f0f11;
    --surface:      #17171a;
    --surface2:     #1e1e22;
    --border:       #2a2a30;
    --accent:       #e8c547;
    --accent-dim:   #a8902f;
    --text:         #f0ede8;
    --muted:        #888897;
    --critical:     #ff5c5c;
    --high:         #ff9a3c;
    --medium:       #5bc4fa;
    --low:          #6bcb77;
    --radius:       12px;
    --radius-lg:    20px;
  }

  /* ── Global Reset ── */
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
  }

  .stApp { background-color: var(--bg) !important; }

  /* Hide Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 0 !important; max-width: 100% !important; }

  /* ── Hero Header ── */
  .hc-hero {
    background: linear-gradient(135deg, #0f0f11 0%, #1a1a1f 50%, #0f0f11 100%);
    border-bottom: 1px solid var(--border);
    padding: 36px 60px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
  }
  .hc-logo-row { display: flex; align-items: center; gap: 16px; }
  .hc-logo-icon {
    width: 52px; height: 52px;
    background: var(--accent);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 26px;
  }
  .hc-logo-text h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 28px;
    color: var(--text);
    margin: 0; line-height: 1;
  }
  .hc-logo-text p {
    font-size: 13px;
    color: var(--muted);
    margin: 4px 0 0;
    letter-spacing: 0.04em;
  }
  .hc-badge {
    background: rgba(232,197,71,0.12);
    border: 1px solid rgba(232,197,71,0.35);
    color: var(--accent);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 5px 12px;
    border-radius: 20px;
  }

  /* ── Main Layout ── */
  .hc-layout {
    display: grid;
    grid-template-columns: 420px 1fr;
    min-height: calc(100vh - 108px);
  }

  /* ── Left Panel ── */
  .hc-left {
    background: var(--surface);
    border-right: 1px solid var(--border);
    padding: 32px 28px;
    display: flex;
    flex-direction: column;
    gap: 24px;
  }

  /* ── Section Label ── */
  .hc-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 0 0 10px;
  }

  /* ── Upload Zone ── */
  .hc-upload-zone {
    background: var(--surface2);
    border: 2px dashed var(--border);
    border-radius: var(--radius-lg);
    padding: 36px 20px;
    text-align: center;
    transition: border-color 0.2s;
  }
  .hc-upload-zone:hover { border-color: var(--accent-dim); }
  .hc-upload-icon { font-size: 40px; margin-bottom: 12px; }
  .hc-upload-title { font-size: 15px; font-weight: 600; color: var(--text); margin: 0 0 6px; }
  .hc-upload-sub { font-size: 13px; color: var(--muted); margin: 0; }

  /* ── Mode Selector Cards ── */
  .hc-mode-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .hc-mode-card {
    background: var(--surface2);
    border: 1.5px solid var(--border);
    border-radius: var(--radius);
    padding: 14px 12px;
    cursor: pointer;
    text-align: center;
    transition: all 0.15s;
  }
  .hc-mode-card.active {
    border-color: var(--accent);
    background: rgba(232,197,71,0.07);
  }
  .hc-mode-card .icon { font-size: 22px; margin-bottom: 6px; }
  .hc-mode-card .name { font-size: 12px; font-weight: 600; color: var(--text); }
  .hc-mode-card .desc { font-size: 11px; color: var(--muted); margin-top: 2px; }

  /* ── Image Preview ── */
  .hc-img-preview {
    border-radius: var(--radius-lg);
    overflow: hidden;
    border: 1px solid var(--border);
    position: relative;
  }
  .hc-img-preview img { width: 100%; display: block; }
  .hc-img-overlay {
    position: absolute; bottom: 0; left: 0; right: 0;
    background: linear-gradient(transparent, rgba(0,0,0,0.7));
    padding: 20px 16px 12px;
    font-size: 12px; color: rgba(255,255,255,0.7);
  }

  /* ── Analyze Button ── */
  .hc-btn {
    background: var(--accent);
    color: #0f0f11;
    border: none;
    border-radius: var(--radius);
    padding: 14px 24px;
    font-size: 15px;
    font-weight: 700;
    font-family: 'DM Sans', sans-serif;
    width: 100%;
    cursor: pointer;
    letter-spacing: 0.01em;
    transition: all 0.15s;
  }
  .hc-btn:hover { background: #f0d060; transform: translateY(-1px); }
  .hc-btn:disabled { background: var(--border); color: var(--muted); cursor: not-allowed; transform: none; }

  /* ── Right Panel ── */
  .hc-right {
    padding: 32px 40px;
    overflow-y: auto;
  }

  /* ── Empty State ── */
  .hc-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 60vh;
    text-align: center;
    gap: 16px;
  }
  .hc-empty-icon {
    font-size: 64px;
    opacity: 0.3;
  }
  .hc-empty h2 {
    font-family: 'DM Serif Display', serif;
    font-size: 24px;
    color: var(--text);
    opacity: 0.5;
    margin: 0;
  }
  .hc-empty p { font-size: 14px; color: var(--muted); max-width: 320px; margin: 0; }

  /* ── Score Card ── */
  .hc-score-card {
    background: linear-gradient(135deg, var(--surface2), var(--surface));
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 28px 32px;
    display: flex;
    align-items: center;
    gap: 28px;
    margin-bottom: 28px;
  }
  .hc-score-ring {
    width: 96px; height: 96px;
    flex-shrink: 0;
    position: relative;
  }
  .hc-score-ring svg { width: 100%; height: 100%; transform: rotate(-90deg); }
  .hc-score-number {
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    font-family: 'DM Serif Display', serif;
    font-size: 28px;
    color: var(--text);
    transform: translate(-50%, -50%);
  }
  .hc-score-info h2 {
    font-family: 'DM Serif Display', serif;
    font-size: 22px;
    color: var(--text);
    margin: 0 0 6px;
  }
  .hc-score-info p { font-size: 14px; color: var(--muted); margin: 0; line-height: 1.5; }
  .hc-score-sub { font-size: 12px; color: var(--muted); margin-top: 10px !important; }

  /* ── Sections ── */
  .hc-section { margin-bottom: 28px; }
  .hc-section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 16px;
  }
  .hc-section-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .hc-section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 17px;
    color: var(--text);
    margin: 0;
  }

  /* ── Finding Cards ── */
  .hc-finding {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 18px;
    margin-bottom: 10px;
    display: grid;
    grid-template-columns: auto 1fr auto;
    gap: 12px;
    align-items: start;
  }
  .hc-finding-icon { font-size: 20px; padding-top: 1px; }
  .hc-finding-body {}
  .hc-finding-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text);
    margin: 0 0 4px;
  }
  .hc-finding-desc {
    font-size: 13px;
    color: var(--muted);
    margin: 0;
    line-height: 1.55;
  }
  .hc-severity-pill {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 3px 9px;
    border-radius: 20px;
    white-space: nowrap;
    flex-shrink: 0;
  }
  .sev-critical { background: rgba(255,92,92,0.15); color: var(--critical); border: 1px solid rgba(255,92,92,0.3); }
  .sev-high     { background: rgba(255,154,60,0.15); color: var(--high);     border: 1px solid rgba(255,154,60,0.3); }
  .sev-medium   { background: rgba(91,196,250,0.15); color: var(--medium);   border: 1px solid rgba(91,196,250,0.3); }
  .sev-low      { background: rgba(107,203,119,0.15);color: var(--low);      border: 1px solid rgba(107,203,119,0.3);}

  /* ── What Works Cards ── */
  .hc-positive {
    background: rgba(107,203,119,0.07);
    border: 1px solid rgba(107,203,119,0.2);
    border-radius: var(--radius);
    padding: 13px 16px;
    margin-bottom: 8px;
    font-size: 13.5px;
    color: #a8ddb0;
    display: flex;
    gap: 10px;
    align-items: flex-start;
  }
  .hc-positive .bullet { color: var(--low); font-size: 16px; line-height: 1.3; }

  /* ── Pro Tips ── */
  .hc-tip {
    background: rgba(232,197,71,0.07);
    border: 1px solid rgba(232,197,71,0.2);
    border-left: 3px solid var(--accent);
    border-radius: 0 var(--radius) var(--radius) 0;
    padding: 13px 16px;
    margin-bottom: 8px;
    font-size: 13.5px;
    color: #d4bb6a;
    display: flex;
    gap: 10px;
    align-items: flex-start;
  }
  .hc-tip .bullet { color: var(--accent); font-size: 15px; line-height: 1.4; }

  /* ── Rules Used ── */
  .hc-rules-grid { display: flex; flex-wrap: wrap; gap: 8px; }
  .hc-rule-tag {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 11px;
    color: var(--muted);
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .hc-rule-tag-sev {
    width: 6px; height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  /* ── Divider ── */
  .hc-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 28px 0;
  }

  /* ── Streamlit widget overrides ── */
  .stFileUploader > div { background: transparent !important; border: none !important; padding: 0 !important; }
  .stFileUploader label { display: none !important; }
  .stButton > button {
    background: var(--accent) !important;
    color: #0f0f11 !important;
    border: none !important;
    font-weight: 700 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    padding: 14px 24px !important;
    border-radius: var(--radius) !important;
    width: 100% !important;
    transition: all 0.15s !important;
  }
  .stButton > button:hover {
    background: #f0d060 !important;
    transform: translateY(-1px) !important;
  }
  .stButton > button:disabled {
    background: var(--border) !important;
    color: var(--muted) !important;
  }
  .stRadio > div { gap: 0 !important; }
  .stRadio label { color: var(--text) !important; font-size: 13px !important; }
  .stSpinner > div { color: var(--accent) !important; }
  div[data-testid="stSidebarNav"] { display: none; }

  /* ── Animations ── */
  @keyframes fadeInUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .animate-in { animation: fadeInUp 0.4s ease forwards; }
  .delay-1 { animation-delay: 0.05s; opacity: 0; }
  .delay-2 { animation-delay: 0.12s; opacity: 0; }
  .delay-3 { animation-delay: 0.19s; opacity: 0; }
  .delay-4 { animation-delay: 0.26s; opacity: 0; }
  .delay-5 { animation-delay: 0.33s; opacity: 0; }

  /* ── Progress Steps ── */
  .hc-progress {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin: 20px 0;
  }
  .hc-step {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 13px;
    color: var(--muted);
  }
  .hc-step.active { color: var(--text); }
  .hc-step.done { color: var(--low); }
  .hc-step-icon {
    width: 22px; height: 22px;
    border-radius: 50%;
    background: var(--surface2);
    border: 1px solid var(--border);
    display: flex; align-items: center; justify-content: center;
    font-size: 11px;
    flex-shrink: 0;
  }
  .hc-step.active .hc-step-icon { border-color: var(--accent); background: rgba(232,197,71,0.1); }
  .hc-step.done .hc-step-icon { border-color: var(--low); background: rgba(107,203,119,0.1); }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _get_secret_key() -> str:
    try:
        return st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        return ""


def get_openai_client() -> OpenAI:
    api_key = (
        st.session_state.get("openai_api_key", "")
        or os.getenv("OPENAI_API_KEY", "")
        or _get_secret_key()
    )
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


@st.cache_resource(show_spinner=False)
def load_vectorstore() -> Optional[Chroma]:
    """Load the persisted ChromaDB — returns None if not yet ingested."""
    api_key = (
        st.session_state.get("openai_api_key", "")
        or os.getenv("OPENAI_API_KEY", "")
        or _get_secret_key()
    )
    if not api_key:
        return None

    # Check the chroma_db folder actually exists before trying to load
    if not Path(CHROMA_PERSIST_DIR).exists():
        return None

    try:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=api_key)
        vs = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        # Try multiple ways to count — different chromadb versions expose different APIs
        try:
            count = vs._collection.count()
        except Exception:
            try:
                count = len(vs.get()["ids"])
            except Exception:
                count = 1  # assume non-empty if we can't count
        if count == 0:
            return None
        return vs
    except Exception:
        return None


def image_to_base64(pil_image: Image.Image) -> str:
    """Convert a PIL image to base64 string for the OpenAI Vision API."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def vision_describe(client: OpenAI, b64_image: str, mode: str) -> str:
    """
    Step 1: Ask GPT-4o to describe the room/problem in rich detail
    so we can embed it for RAG retrieval.
    """
    mode_context = {
        "Staging Airbnb":         "Airbnb short-term rental staging and interior design quality",
        "Έλεγχος Συντήρησης":     "home maintenance issues, structural problems, and repair needs",
        "Πλήρης Επιθεώρηση":      "both interior design/staging quality and home maintenance/structural issues",
        "Εξωτερικός Έλεγχος":     "exterior curb appeal, structural integrity, and landscaping quality",
    }

    prompt = f"""You are an expert home inspector and Airbnb staging specialist.
Analyze this image in the context of: {mode_context.get(mode, 'general home assessment')}.

Provide a detailed, technical description in ENGLISH covering ALL of the following:
1. Lighting: quality, color temperature, sources, shadows, glare, missing lamps
2. Furniture: layout, scale, symmetry, traffic flow, rug sizing, nightstands
3. Aesthetics: color palette, textures, art, decor, focal points, clutter, cables
4. Surfaces: walls (paint, damage, stains), floors, ceilings, windows, curtains
5. Kitchen/Bathroom: cleanliness, appliances, accessories, mold, caulk, towels
6. Safety: visible detectors, handrails, extinguishers, GFCI outlets
7. Exterior (if visible): paint, landscaping, door, lighting, pathways
8. Structural concerns: cracks, water stains, rot, pest signs, sagging
9. Guest experience signals: welcome book, WiFi sign, storage, workspace

Be maximally specific — mention colors, materials, dimensions, exact locations.
This description is used for semantic search against an expert rules database.
Output ONLY the description, no headers, no bullets, plain prose."""

    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}", "detail": "high"}},
                    {"type": "text", "text": prompt},
                ]
            }],
            max_tokens=800,
        )
        usage = response.usage
        tokens = {"vision_prompt": usage.prompt_tokens, "vision_completion": usage.completion_tokens}
        return response.choices[0].message.content.strip(), tokens
    except Exception as e:
        raise RuntimeError(f"Vision API call failed: {e}")


def retrieve_rules(vectorstore: Chroma, description: str) -> list[dict]:
    """Step 2: Embed the description and retrieve top-K matching rules."""
    try:
        results = vectorstore.similarity_search_with_score(description, k=TOP_K_RULES)
        rules = []
        for doc, score in results:
            rules.append({
                "id":          doc.metadata["id"],
                "category":    doc.metadata["category"],
                "title":       doc.metadata["title"],
                "severity":    doc.metadata["severity"],
                "action":      doc.metadata["action"],
                "description": doc.metadata["description"],
                "score":       round(float(score), 4),
            })
        return rules
    except Exception as e:
        raise RuntimeError(f"ChromaDB retrieval failed: {e}")


def generate_report(client: OpenAI, b64_image: str, description: str, rules: list[dict], mode: str) -> dict:
    """
    Step 3: Generate the final structured JSON report.
    The LLM receives the image + vision description + RAG-retrieved rules.
    """
    rules_json = json.dumps(rules, indent=2)

    system_prompt = """Είσαι το "HouseMaxing", ένα κορυφαίο AI σύστημα επιθεώρησης κατοικιών και staging Airbnb.
Παράγεις ακριβείς, εφαρμόσιμες αναφορές αξιολόγησης στα ΕΛΛΗΝΙΚΑ.

ΑΥΣΤΗΡΟΙ ΚΑΝΟΝΕΣ:
- ΚΑΘΕ εύρημα ΠΡΕΠΕΙ να βασίζεται στα δεδομένα RETRIEVED_RULES που σου παρέχονται. Μη εφεύρεις προβλήματα.
- Χρησιμοποίησε ΟΛΑ τα πεδία κάθε κανόνα: category, description, action, severity, photo_tip.
- Για κάθε Critical ή High εύρημα, ΑΝΤΙΓΡΑΨΕ ακριβώς το πεδίο "action" από τον αντίστοιχο κανόνα.
- Η βαθμολογία υπολογίζεται αφαιρώντας: Critical: -2, High: -1.5, Medium: -0.8, Low: -0.3 από το 10.
- Τα photo_tip πεδία να συμπεριλαμβάνονται στα pro_tips όταν είναι σχετικά.
- Output ONLY valid JSON. Χωρίς markdown, χωρίς code fences, χωρίς εισαγωγικό κείμενο."""

    user_prompt = f"""ΤΡΟΠΟΣ ΑΝΑΛΥΣΗΣ: {mode}

ΠΕΡΙΓΡΑΦΗ ΕΙΚΟΝΑΣ ΑΠΟ AI VISION:
{description}

ΑΝΑΚΤΗΜΕΝΟΙ ΚΑΝΟΝΕΣ ΕΙΔΙΚΩΝ (μοναδική πηγή αλήθειας — χρησιμοποίησε ΟΛΑ τα πεδία):
{rules_json}

Δημιούργησε αναφορά JSON με αυτή ακριβώς τη δομή (ΟΛΑ τα κείμενα στα ΕΛΛΗΝΙΚΑ):
{{
  "overall_score": <float 0.0-10.0>,
  "score_label": "<ένα από: Άριστα | Καλά | Μέτρια | Χρειάζεται Βελτίωση | Κρίσιμα Προβλήματα>",
  "summary": "<2-3 προτάσεις σύνοψης του χώρου στα ελληνικά>",
  "what_works": [
    "<συγκεκριμένο θετικό στοιχείο στα ελληνικά>",
    "<συγκεκριμένο θετικό στοιχείο στα ελληνικά>"
  ],
  "critical_fixes": [
    {{
      "title": "<σύντομο όνομα προβλήματος στα ελληνικά>",
      "description": "<τι βλέπεις στην εικόνα στα ελληνικά>",
      "action": "<ακριβώς το πεδίο action από τον κανόνα, μεταφρασμένο στα ελληνικά>",
      "severity": "<Critical|High|Medium|Low>",
      "rule_id": "<το ID του κανόνα>",
      "icon": "<ένα σχετικό emoji>"
    }}
  ],
  "pro_tips": [
    "<εφαρμόσιμη συμβουλή βασισμένη στο photo_tip ή action του κανόνα, στα ελληνικά>",
    "<εφαρμόσιμη συμβουλή, στα ελληνικά>",
    "<εφαρμόσιμη συμβουλή, στα ελληνικά>"
  ]
}}

Συμπερίληψε μόνο ευρήματα που είναι πραγματικά ορατά στην εικόνα.
Συμπερίληψε 2-4 στοιχεία σε critical_fixes. 2-4 σε what_works. 2-3 pro_tips.
Αξιοποίησε το πεδίο photo_tip κάθε κανόνα για πρακτικές συμβουλές φωτογράφισης στα pro_tips."""

    try:
        response = client.chat.completions.create(
            model=REPORT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}", "detail": "high"}},
                        {"type": "text", "text": user_prompt},
                    ]
                }
            ],
            max_tokens=2000,
            response_format={"type": "json_object"},
        )
        usage = response.usage
        tokens = {"report_prompt": usage.prompt_tokens, "report_completion": usage.completion_tokens}
        raw = response.choices[0].message.content.strip()
        return json.loads(raw), tokens
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Report JSON parse failed: {e}")
    except Exception as e:
        raise RuntimeError(f"Report generation API call failed: {e}")


def score_color(score: float) -> str:
    if score >= 8:   return "#6bcb77"
    if score >= 6:   return "#5bc4fa"
    if score >= 4:   return "#ff9a3c"
    return "#ff5c5c"


def severity_css(severity: str) -> str:
    return {"Critical": "sev-critical", "High": "sev-high", "Medium": "sev-medium", "Low": "sev-low"}.get(severity, "sev-low")


def render_score_ring(score: float) -> str:
    """Render an SVG score ring."""
    pct = score / 10
    circumference = 2 * 3.14159 * 38
    dash = pct * circumference
    color = score_color(score)
    return f"""
    <div class="hc-score-ring" style="position:relative;width:96px;height:96px;">
      <svg viewBox="0 0 96 96" style="width:100%;height:100%;transform:rotate(-90deg);">
        <circle cx="48" cy="48" r="38" fill="none" stroke="#2a2a30" stroke-width="7"/>
        <circle cx="48" cy="48" r="38" fill="none" stroke="{color}" stroke-width="7"
                stroke-dasharray="{dash:.1f} {circumference:.1f}"
                stroke-linecap="round"/>
      </svg>
      <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
                  font-family:'DM Serif Display',serif;font-size:26px;color:#f0ede8;
                  line-height:1;">{score:.1f}</div>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
#  RENDER REPORT
# ══════════════════════════════════════════════════════════════════════════════

def render_report(report: dict, rules_used: list[dict]):
    score = report.get("overall_score", 0)
    label = report.get("score_label", "")
    summary = report.get("summary", "")
    what_works = report.get("what_works", [])
    critical_fixes = report.get("critical_fixes", [])
    pro_tips = report.get("pro_tips", [])

    ring_html = render_score_ring(score)
    color = score_color(score)

    # Score Card
    st.markdown(f"""
    <div class="hc-score-card animate-in delay-1">
      {ring_html}
      <div class="hc-score-info">
        <h2>{label}</h2>
        <p>{summary}</p>
        <p class="hc-score-sub">Βαθμολογία: <strong style="color:{color}">{score:.1f}/10</strong> &nbsp;·&nbsp; {len(critical_fixes)} ευρήματα &nbsp;·&nbsp; {len(rules_used)} κανόνες</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # What Works
    if what_works:
        st.markdown("""
        <div class="hc-section animate-in delay-2">
          <div class="hc-section-header">
            <div class="hc-section-dot" style="background:#6bcb77;"></div>
            <h3 class="hc-section-title">Τι Λειτουργεί Καλά</h3>
          </div>
        """, unsafe_allow_html=True)
        for item in what_works:
            st.markdown(f"""
            <div class="hc-positive">
              <span class="bullet">✓</span>
              <span>{item}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Critical Fixes
    if critical_fixes:
        st.markdown("""
        <div class="hc-section animate-in delay-3">
          <div class="hc-section-header">
            <div class="hc-section-dot" style="background:#ff5c5c;"></div>
            <h3 class="hc-section-title">Ευρήματα & Σχέδιο Δράσης</h3>
          </div>
        """, unsafe_allow_html=True)
        for fix in critical_fixes:
            sev_css = severity_css(fix.get("severity", "Low"))
            rule_ref = fix.get("rule_id", "")
            st.markdown(f"""
            <div class="hc-finding">
              <div class="hc-finding-icon">{fix.get('icon','🔧')}</div>
              <div class="hc-finding-body">
                <p class="hc-finding-title">{fix.get('title','')}</p>
                <p class="hc-finding-desc">{fix.get('description','')}</p>
                <p class="hc-finding-desc" style="margin-top:8px;color:#c8c8d8;">
                  <strong style="color:#f0ede8;">Ενέργεια:</strong> {fix.get('action','')}
                </p>
                {f'<p style="font-size:11px;color:#555568;margin-top:6px;">Κανόνας: {rule_ref}</p>' if rule_ref else ''}
              </div>
              <span class="hc-severity-pill {sev_css}">{fix.get('severity','')}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Pro Tips
    if pro_tips:
        st.markdown("""
        <div class="hc-section animate-in delay-4">
          <div class="hc-section-header">
            <div class="hc-section-dot" style="background:#e8c547;"></div>
            <h3 class="hc-section-title">Επαγγελματικές Συμβουλές</h3>
          </div>
        """, unsafe_allow_html=True)
        for tip in pro_tips:
            st.markdown(f"""
            <div class="hc-tip">
              <span class="bullet">★</span>
              <span>{tip}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Rules Used
    if rules_used:
        st.markdown('<hr class="hc-divider">', unsafe_allow_html=True)
        st.markdown("""
        <div class="hc-section animate-in delay-5">
          <p class="hc-label">Κανόνες Βάσης Γνώσης που Αντιστοιχήθηκαν</p>
          <div class="hc-rules-grid">
        """, unsafe_allow_html=True)
        for r in rules_used:
            dot_color = {"Critical": "#ff5c5c", "High": "#ff9a3c", "Medium": "#5bc4fa", "Low": "#6bcb77"}.get(r["severity"], "#888")
            st.markdown(f"""
            <div class="hc-rule-tag">
              <div class="hc-rule-tag-sev" style="background:{dot_color};"></div>
              <span>{r['id']} — {r['title']}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Hero Header ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hc-hero">
      <div class="hc-logo-row">
        <div class="hc-logo-icon">🏠</div>
        <div class="hc-logo-text">
          <h1>HouseMaxing</h1>
          <p>AI Επιθεώρηση Σπιτιού & Staging Airbnb</p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── API Key Gate ─────────────────────────────────────────────────────────
    env_key = os.getenv("OPENAI_API_KEY", "") or _get_secret_key()
    if not env_key and not st.session_state.get("openai_api_key"):
        st.markdown("""
        <div style="max-width:480px;margin:80px auto;background:#17171a;border:1px solid #2a2a30;
                    border-radius:20px;padding:40px 36px;text-align:center;">
          <div style="font-size:48px;margin-bottom:16px;">🔑</div>
          <h2 style="font-family:'DM Serif Display',serif;font-size:22px;color:#f0ede8;margin:0 0 8px;">
            Εισάγετε το OpenAI API Key σας
          </h2>
          <p style="font-size:13px;color:#888897;margin:0 0 24px;line-height:1.6;">
            Το κλειδί αποθηκεύεται μόνο στη συνεδρία του browser και δεν αποθηκεύεται πουθενά.
          </p>
        </div>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            key_input = st.text_input(
                label="API Key",
                placeholder="sk-...",
                type="password",
                label_visibility="collapsed",
            )
            if st.button("✦ Συνέχεια", use_container_width=True):
                if key_input.startswith("sk-"):
                    st.session_state["openai_api_key"] = key_input
                    st.rerun()
                else:
                    st.error("Το κλειδί δεν φαίνεται έγκυρο. Πρέπει να ξεκινά με sk-")
            st.markdown(
                '<p style="font-size:11px;color:#555568;text-align:center;margin-top:12px;">'                'Αποκτήστε το κλειδί σας στο <a href="https://platform.openai.com/api-keys" target="_blank" '                'style="color:#e8c547;">platform.openai.com/api-keys</a></p>',
                unsafe_allow_html=True
            )
        return

    # ── Load Resources ────────────────────────────────────────────────────────
    client = get_openai_client()
    vectorstore = load_vectorstore()
    
    if client is None:
        st.error("⚠️ Αδυναμία σύνδεσης. Ελέγξτε το API key σας.")
        if st.button("🔑 Αλλαγή API Key"):
            st.session_state.pop("openai_api_key", None)
            st.rerun()
        return

    # ── Key Reset in Sidebar ─────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<p style="font-size:11px;color:#888897;margin-bottom:8px;">OPENAI API KEY</p>', unsafe_allow_html=True)
        masked = "sk-..." + st.session_state.get("openai_api_key", "")[-4:] if st.session_state.get("openai_api_key") else "From environment"
        st.markdown(f'<p style="font-size:12px;color:#f0ede8;margin-bottom:12px;">{masked}</p>', unsafe_allow_html=True)
        if st.button("🔑 Αλλαγή Κλειδιού", use_container_width=True):
            st.session_state.pop("openai_api_key", None)
            st.rerun()

    # ── Two-Column Layout ─────────────────────────────────────────────────────
    left_col, right_col = st.columns([4, 6], gap="large")

    with left_col:
        st.markdown('<div style="padding: 28px 8px 0;">', unsafe_allow_html=True)

        # DB Status
        if vectorstore is None:
            st.markdown("""
            <div style="background:rgba(255,92,92,0.07);border:1px solid rgba(255,92,92,0.25);
                        border-radius:10px;padding:14px 16px;margin-bottom:16px;">
              <p style="font-size:13px;font-weight:600;color:#ff5c5c;margin:0 0 6px;">
                ⚠️ Η βάση γνώσης δεν φορτώθηκε
              </p>
              <p style="font-size:12px;color:#888897;margin:0 0 10px;line-height:1.6;">
                Τρέξτε αυτή την εντολή στο terminal, μετά πατήστε Επαναφόρτωση:
              </p>
              <code style="background:#0f0f11;border:1px solid #2a2a30;border-radius:6px;
                           padding:6px 10px;font-size:12px;color:#e8c547;display:block;">
                python ingest.py
              </code>
            </div>""", unsafe_allow_html=True)
            if st.button("↻ Επαναφόρτωση Βάσης Γνώσης", use_container_width=True):
                st.cache_resource.clear()
                st.rerun()
        else:
            rule_count = vectorstore._collection.count()
            st.markdown(f"""
            <div style="background:rgba(107,203,119,0.07);border:1px solid rgba(107,203,119,0.2);
                        border-radius:10px;padding:10px 14px;display:flex;align-items:center;
                        gap:10px;font-size:12px;color:#a8ddb0;margin-bottom:16px;">
              <span style="font-size:16px;">✓</span>
              <span>Βάση γνώσης ενεργή — <strong>{rule_count} κανόνες ειδικών</strong> φορτωμένοι</span>
            </div>""", unsafe_allow_html=True)

        # Mode Selection
        st.markdown('<p class="hc-label">Τρόπος Ανάλυσης</p>', unsafe_allow_html=True)
        mode = st.radio(
            label="mode",
            options=["Staging Airbnb", "Έλεγχος Συντήρησης", "Πλήρης Επιθεώρηση", "Εξωτερικός Έλεγχος"],
            label_visibility="collapsed",
            horizontal=False,
        )

        mode_info = {
            "Staging Airbnb":         ("🛋️", "Ποιότητα design & διακόσμησης για βραχυχρόνιες μισθώσεις"),
            "Έλεγχος Συντήρησης":     ("🔧", "Εντοπισμός επισκευών, ζημιών & θεμάτων ασφάλειας"),
            "Πλήρης Επιθεώρηση":      ("🔍", "Πλήρης αισθητική & δομική αξιολόγηση"),
            "Εξωτερικός Έλεγχος":     ("🏡", "Εμφάνιση, στέγη, θεμέλια & διαμόρφωση"),
        }
        icon, desc = mode_info[mode]
        st.markdown(f"""
        <div style="background:rgba(232,197,71,0.07);border:1px solid rgba(232,197,71,0.2);
                    border-radius:10px;padding:10px 14px;display:flex;align-items:center;
                    gap:10px;font-size:12px;color:#d4bb6a;margin:8px 0 20px;">
          <span style="font-size:18px;">{icon}</span>
          <span>{desc}</span>
        </div>""", unsafe_allow_html=True)

        # File Uploader — multiple files
        st.markdown('<p class="hc-label">Ανέβασμα Φωτογραφιών</p>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            label="Upload",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        # Image Previews or Upload Prompt
        if uploaded_files:
            st.markdown(f"""
            <div style="font-size:11px;color:#888897;margin-bottom:8px;">
              {len(uploaded_files)} φωτογραφί{"ες" if len(uploaded_files) > 1 else "α"} επιλεγμέν{"ες" if len(uploaded_files) > 1 else "η"}
            </div>""", unsafe_allow_html=True)
            # Show thumbnails in a 2-col grid
            thumb_cols = st.columns(2)
            for i, uf in enumerate(uploaded_files):
                with thumb_cols[i % 2]:
                    img = Image.open(uf).convert("RGB")
                    st.image(img, use_column_width=True)
                    st.markdown(f"""<div style="font-size:10px;color:#555568;text-align:center;
                        margin:-4px 0 8px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
                        {uf.name}</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="hc-upload-zone">
              <div class="hc-upload-icon">📸</div>
              <p class="hc-upload-title">Σύρετε φωτογραφίες εδώ</p>
              <p class="hc-upload-sub">JPG, PNG ή WEBP &nbsp;·&nbsp; Υποστηρίζονται πολλαπλά αρχεία</p>
            </div>""", unsafe_allow_html=True)

        # Analyze Button
        st.markdown("<div style='margin-top:16px;'>", unsafe_allow_html=True)
        n_photos = len(uploaded_files) if uploaded_files else 0
        analyze_disabled = (n_photos == 0 or vectorstore is None)
        btn_label = f"✦ Ανάλυση {n_photos} Φωτογραφι{'ών' if n_photos != 1 else 'ας'}" if n_photos > 0 else "✦ Ανάλυση Χώρου"
        analyze_clicked  = st.button(
            btn_label,
            disabled=analyze_disabled,
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Right Panel ───────────────────────────────────────────────────────────
    with right_col:
        st.markdown("<div style='padding: 28px 16px 0;'>", unsafe_allow_html=True)

        if "report" not in st.session_state:
            # Empty State
            st.markdown("""
            <div class="hc-empty">
              <div class="hc-empty-icon">🏠</div>
              <h2>Η αναφορά σας θα εμφανιστεί εδώ</h2>
              <p>Ανεβάστε μια φωτογραφία, επιλέξτε τρόπο ανάλυσης και πατήστε Ανάλυση Χώρου.</p>
            </div>""", unsafe_allow_html=True)

        # ── Run Pipeline ──────────────────────────────────────────────────────
        if analyze_clicked and uploaded_files and vectorstore:
            st.session_state.pop("reports", None)
            st.session_state.pop("total_tokens", None)

            all_reports   = []
            total_tokens  = {"vision_prompt": 0, "vision_completion": 0,
                              "report_prompt": 0, "report_completion": 0}

            progress_placeholder = st.empty()

            def show_step(step: int, message: str, photo_idx: int, total: int):
                steps = [
                    ("Κωδικοποίηση εικόνας",   step > 0),
                    ("Ανάλυση εικόνας με AI",   step > 1),
                    ("Αναζήτηση κανόνων",       step > 2),
                    ("Δημιουργία αναφοράς",     step > 3),
                ]
                html = '<div class="hc-progress">'
                for i, (label, done) in enumerate(steps):
                    current = (i == step - 1)
                    css  = "done" if done else ("active" if current else "")
                    icon = "✓"    if done else ("●"      if current else str(i + 1))
                    html += f"""<div class="hc-step {css}">
                      <div class="hc-step-icon">{icon}</div>
                      <span>{label}</span></div>"""
                html += "</div>"
                progress_placeholder.markdown(
                    f'<div style="padding:20px 0;">'
                    f'<p style="font-size:11px;color:#888897;margin-bottom:2px;">'
                    f'Φωτογραφία {photo_idx} από {total}</p>'
                    f'<div style="font-size:14px;font-weight:600;color:#f0ede8;margin-bottom:12px;">{message}</div>'
                    f'{html}</div>',
                    unsafe_allow_html=True
                )

            try:
                for idx, uf in enumerate(uploaded_files, 1):
                    pil_img = Image.open(uf).convert("RGB")

                    show_step(1, f"Προετοιμασία {uf.name}…", idx, len(uploaded_files))
                    b64 = image_to_base64(pil_img)

                    show_step(2, f"Το AI εξετάζει {uf.name}…", idx, len(uploaded_files))
                    description, vtok = vision_describe(client, b64, mode)

                    show_step(3, "Αντιστοίχιση κανόνων ειδικών…", idx, len(uploaded_files))
                    rules = retrieve_rules(vectorstore, description)

                    show_step(4, "Σύνταξη αναφοράς…", idx, len(uploaded_files))
                    report, rtok = generate_report(client, b64, description, rules, mode)

                    all_reports.append({
                        "filename":    uf.name,
                        "report":      report,
                        "rules_used":  rules,
                        "description": description,
                    })
                    for k in vtok: total_tokens[k] += vtok[k]
                    for k in rtok: total_tokens[k] += rtok[k]

                st.session_state["reports"]       = all_reports
                st.session_state["total_tokens"]  = total_tokens
                progress_placeholder.empty()

            except RuntimeError as e:
                progress_placeholder.empty()
                st.error(f"**Σφάλμα επεξεργασίας:** {e}")
            except Exception as e:
                progress_placeholder.empty()
                st.error(f"**Απρόσμενο σφάλμα:** {e}")

        # ── Render Saved Reports ──────────────────────────────────────────────
        if "reports" in st.session_state:
            reports      = st.session_state["reports"]
            total_tokens = st.session_state.get("total_tokens", {})

            # ── Token Usage Bar ───────────────────────────────────────────────
            total_in  = total_tokens.get("vision_prompt", 0)  + total_tokens.get("report_prompt", 0)
            total_out = total_tokens.get("vision_completion", 0) + total_tokens.get("report_completion", 0)
            total_all = total_in + total_out
            # GPT-4.1 pricing: $2/1M input, $8/1M output
            cost_usd  = (total_in * 2 + total_out * 8) / 1_000_000

            st.markdown(f"""
            <div style="background:#17171a;border:1px solid #2a2a30;border-radius:12px;
                        padding:14px 18px;margin-bottom:24px;display:flex;align-items:center;
                        flex-wrap:wrap;gap:16px;">
              <div style="font-size:11px;font-weight:600;letter-spacing:0.1em;
                          text-transform:uppercase;color:#555568;">Χρήση Tokens</div>
              <div style="display:flex;gap:20px;flex-wrap:wrap;flex:1;">
                <div style="font-size:12px;color:#888897;">
                  Εισόδου <strong style="color:#5bc4fa;">{total_in:,}</strong>
                </div>
                <div style="font-size:12px;color:#888897;">
                  Εξόδου <strong style="color:#6bcb77;">{total_out:,}</strong>
                </div>
                <div style="font-size:12px;color:#888897;">
                  Σύνολο <strong style="color:#f0ede8;">{total_all:,}</strong>
                </div>
                <div style="font-size:12px;color:#888897;">
                  Εκτιμ. κόστος <strong style="color:#e8c547;">${cost_usd:.4f}</strong>
                </div>
                <div style="font-size:12px;color:#555568;">
                  {len(reports)} φωτογραφί{"ες" if len(reports) > 1 else "α"} αναλύθηκ{"αν" if len(reports) > 1 else "ε"}
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

            # ── One report per photo ───────────────────────────────────────────
            if len(reports) == 1:
                render_report(reports[0]["report"], reports[0]["rules_used"])
            else:
                tabs = st.tabs([f"📸 {r['filename']}" for r in reports])
                for tab, r in zip(tabs, reports):
                    with tab:
                        render_report(r["report"], r["rules_used"])

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()