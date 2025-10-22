'''
@copyright: Copyright (C) Aidan McCaffrey - All Rights Reserved
'''

import random
from datetime import datetime
from zoneinfo import ZoneInfo
import re
import html
import math
import numpy as np
import streamlit as st
import gensim.downloader as api
import streamlit.components.v1 as components
from PIL import Image, ImageDraw
from typing import Optional, Tuple

# -----------------------------
# Logo generation (Pillow)
# -----------------------------

def make_logo(display_size: int = 128, bg: Optional[Tuple[int, int, int]] = None) -> Image.Image:
    """
    Generate a crisp square logo with 3 axes and arrowheads.
    - Oversamples at 4× and downsamples with LANCZOS for sharp small rendering.
    - Ensures arrow tips stay inside a safe margin so nothing is clipped in the UI.
    - Overlaps the line into the arrowhead to avoid visible seams after scaling.
    - bg: optional RGB background (e.g., (17,17,17) for dark). None = transparent.
    """
    scale = 4  # draw big, then downscale
    size = display_size * scale

    # Background (transparent by default)
    if bg is None:
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    else:
        img = Image.new("RGBA", (size, size), (*bg, 255))
    draw = ImageDraw.Draw(img)

    # Styling (proportional)
    thick = max(2, int(size * 0.06))
    head = max(6, int(size * 0.11))
    overlap = int(head * 0.55)   # shaft overlaps into head to hide seam
    head_w = int(head * 0.65)    # half-width of arrowhead

    # Safe canvas margin so arrow tips never touch the edge
    margin = max(6, int(size * 0.06))

    # Nominal layout
    ox, oy = int(size * 0.38), int(size * 0.66)
    L_nom = int(size * 0.50)

    # Constrain L so all tips (including arrowhead) remain within [margin, size - margin]
    L_max_x = (size - margin - head) - ox                       # X tip (right)
    L_max_y = oy - (margin + head)                              # Y tip (up)
    L_max_z_x = (ox - (margin + head)) / 0.75                   # Z tip x (left)
    L_max_z_y = ((size - margin - head) - oy) / 0.45            # Z tip y (down)
    L = min(L_nom, L_max_x, L_max_y, L_max_z_x, L_max_z_y)
    L = max(10, int(L))

    def arrow(x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int, int]) -> None:
        # Unit vector along the shaft
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy) or 1.0
        ux, uy = dx / length, dy / length
        # Perpendicular
        px, py = -uy, ux

        # Arrow head polygon (draw first)
        hx, hy = x2 - ux * head, y2 - uy * head
        p1 = (int(x2), int(y2))
        p2 = (int(hx + px * head_w), int(hy + py * head_w))
        p3 = (int(hx - px * head_w), int(hy - py * head_w))
        draw.polygon([p1, p2, p3], fill=color)

        # Shaft (overlap slightly into the head)
        sx2, sy2 = int(x2 - ux * overlap), int(y2 - uy * overlap)
        draw.line((int(x1), int(y1), sx2, sy2), fill=color, width=thick)

        # Round cap at origin for nicer small-size look
        r = int(thick / 2)
        draw.ellipse((int(x1 - r), int(y1 - r), int(x1 + r), int(y1 + r)), fill=color)

    # Axes (tips now guaranteed inside the safe area)
    arrow(ox, oy, int(ox + L), oy, (220, 60, 60, 255))                               # X (red)
    arrow(ox, oy, ox, int(oy - L), (60, 180, 60, 255))                                # Y (green)
    arrow(ox, oy, int(ox - 0.75 * L), int(oy + 0.45 * L), (70, 120, 220, 255))        # Z (blue)

    # Origin dot (above caps)
    r0 = max(2, int(size * 0.02))
    draw.ellipse((ox - r0, oy - r0, ox + r0, oy + r0), fill=(30, 30, 30, 255))

    # Downscale with antialiasing
    if display_size != size:
        img = img.resize((display_size, display_size), Image.LANCZOS)
    return img

# Build logo before any Streamlit calls so we can use it for page_icon
page_icon = make_logo(256)
header_logo = make_logo(96)

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Orthogonle", page_icon=page_icon, layout="centered")

# Mobile-friendly CSS and prompt visibility
st.markdown(
    """
    <style>
      .block-container { max-width: 740px; padding-top: 1rem; padding-bottom: 4rem; }
      .prompt-chip {
        display: inline-block; padding: .55rem .85rem; margin: .25rem;
        border-radius: 999px; background: #eef2f7; border: 1px solid #e0e0e0;
        font-weight: 600; color: #111;
      }
      .stButton > button { width: 100%; }
      .small-muted { color: #6c757d; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Scoring config
NEGATIVE_BONUS = 1.0  # full extra credit for negative similarities

# -----------------------------
# Load embeddings (cached)
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_model():
    # Downloads on first run; cached afterwards by Streamlit
    return api.load("glove-wiki-gigaword-50")  # 50d embeddings (~66MB)

model = load_model()

# -----------------------------
# Utilities
# -----------------------------
def cosine_sim(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))

def score_from_similarity(sim: float, neg_bonus: float = NEGATIVE_BONUS) -> float:
    """
    Map cosine similarity [-1,1] to a score:
      - sim >= 0:  score = (1 - sim) * 100      -> 0..100
      - sim <  0:  score = 100 + neg_bonus * (-sim) * 100
    """
    s = clamp(sim, -1.0, 1.0)
    if s >= 0.0:
        return (1.0 - s) * 100.0
    else:
        return 100.0 + neg_bonus * (-s) * 100.0

def score_category(avg_score: float) -> str:
    # Thresholds adapted to average (sum thresholds ÷ 3):
    # <90: Novice, 90–100: Good, >100: Very Good, >133.3: Legend
    if avg_score > 133.3333:
        return "Legend"
    elif avg_score > 100:
        return "Very Good"
    elif avg_score >= 90:
        return "Good"
    else:
        return "Novice"

def category_emoji(label: str) -> str:
    return {
        "Novice": "👶",
        "Good": "🙂",
        "Very Good": "💪",
        "Legend": "🧙‍♂️",
    }.get(label, "")

def in_vocab(word: str) -> bool:
    return word in model.key_to_index

def get_vec(word: str):
    return model[word]

# -----------------------------
# Build a 1000-word pool from model vocab (cached)
# -----------------------------
STOPWORDS = {
    "the","a","an","and","or","but","if","then","with","without","within","into","onto","upon","over","under",
    "in","on","at","by","for","to","from","of","as","is","are","was","were","be","been","being",
    "this","that","these","those","it","its","they","them","their","theirs","he","him","his","she","her","hers",
    "we","us","our","ours","you","your","yours","i","me","my","mine",
    "what","which","who","whom","whose","where","when","why","how",
    "not","no","yes","do","does","did","done","doing","can","could","may","might","must","will","would","shall","should",
    "also","there","here","about","between","through","during","before","after","above","below","again","further",
    "both","each","few","more","most","other","some","such","only","own","same","so","than","too","very"
}
BANNED_SHORT = {"us","uk","u.s","u.s.","u.k","u.k.","etc","e.g","e.g.","i.e","i.e."}
BANNED_CATEGORIES = {"mr","mrs","ms","dr","inc","ltd","co","corp","llc","jr","sr"}
MONTHS_DAYS = {
    "january","february","march","april","may","june","july","august","september","october","november","december",
    "monday","tuesday","wednesday","thursday","friday","saturday","sunday"
}
ALPHA_RE = re.compile(r"^[a-z]+$")  # lowercase ASCII letters only

def looks_like_good_prompt(w: str) -> bool:
    if not ALPHA_RE.match(w):
        return False
    if len(w) < 3 or len(w) > 14:
        return False
    if w in STOPWORDS or w in BANNED_SHORT or w in BANNED_CATEGORIES or w in MONTHS_DAYS:
        return False
    return True

@st.cache_data(show_spinner=True)
def build_word_pool(n_desired=5000, search_span=120000):
    items = sorted(model.key_to_index.items(), key=lambda kv: kv[1])  # by frequency rank
    pool = []
    for word, idx in items:
        if idx >= search_span:
            break
        w = word.lower()
        if looks_like_good_prompt(w):
            pool.append(w)
            if len(pool) >= n_desired:
                break
    return sorted(set(pool))

WORD_POOL = build_word_pool(n_desired=5000, search_span=120000)
if len(WORD_POOL) < 3:
    st.error("Could not build a large enough prompt pool from the model vocabulary.")
    st.stop()

# -----------------------------
# Daily prompts (US Eastern time) and single-play lock
# -----------------------------
today_et = datetime.now(ZoneInfo("America/New_York")).date()
DAILY_SEED = f"orthogonle-{today_et.isoformat()}-v1"
PLAYED_KEY = f"played_et_{today_et.isoformat()}"

rng = random.Random(DAILY_SEED)
prompts = rng.sample(WORD_POOL, 3)

# Initialize defaults for inputs early (before any widgets)
if "g" not in st.session_state:
    st.session_state["g"] = ""

# Clear handler placed BEFORE inputs so changes happen prior to widget instantiation
def clear_inputs():
    # Only clear guess; do not reset the daily lock
    if "g" in st.session_state:
        del st.session_state["g"]
    # Clear any stored result fields
    for k in ("scores", "sims", "guess", "avg", "category", "share_text_with", "share_text_without"):
        st.session_state.pop(k, None)
    st.rerun()

# -----------------------------
# Header with logo and Info
# -----------------------------
cols_header = st.columns([1, 6])
with cols_header[0]:
    st.image(header_logo, width=96)
with cols_header[1]:
    st.title("Orthogonle")
    st.caption("Guess a word that's as far afield as possible from the three prompts.")

st.write(f"Daily date: {today_et.isoformat()}")

info_content = """
- Goal: Enter a single word that is dissimilar to all three prompt words. Higher scores are better.
- Scoring per prompt: 0-100 baseline. Negative similarities earn extra credit (>100).
- Score is based on semantic similarity between the words.
- Words that are near-synonyms or often appear together in a sentence -> low score.
- Words that would never appear close to each other -> high score.
- Your total is the average of the three per-prompt scores.
- Score categories (by average):
  - <90: Novice 👶
  - 90-100: Good 🙂 
  - 100+: Very Good 💪
  - 133.3+: Legend 🧙‍♂️
"""
if hasattr(st, "popover"):
    with st.popover("Info"):
        st.markdown(info_content)
else:
    with st.expander("Info"):
        st.markdown(info_content)

# If user already played today, hide prompt/inputs and show results/share
already_played = st.session_state.get(PLAYED_KEY, False)

if already_played:
    st.info("You’ve already played today (US/Eastern). Come back tomorrow!")
else:
    # Action bar with Clear (runs before inputs)
    cols_actions = st.columns(2)
    with cols_actions[0]:
        st.write("")
    with cols_actions[1]:
        if st.button("Clear", use_container_width=True):
            clear_inputs()

    # Prompts (visible) with fixed readable text color
    st.markdown(
        f"""
        <div>
          <span class="prompt-chip">{prompts[0]}</span>
          <span class="prompt-chip">{prompts[1]}</span>
          <span class="prompt-chip">{prompts[2]}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Single input (one guess for all three prompts)
    g = st.text_input(
        "Your single guess (applies to all three prompts)",
        key="g",
        placeholder="Type a single word..."
    ).strip().lower()

    def valid_guess(guess: str):
        if not guess:
            return False, "Please enter a word."
        if not guess.isalpha():
            return False, "Use letters only (no spaces, hyphens, or numbers)."
        if guess in prompts:
            return False, "Guess must be different from the prompts."
        if not in_vocab(guess):
            return False, "Word not found in the embedding vocabulary. Try a more common word."
        return True, ""

    # Score button
    score_btn = st.button("Score", use_container_width=True)

    # Scoring (only runs when 'Score' is pressed)
    if score_btn:
        ok, msg = valid_guess(g)
        if not ok:
            st.error(msg)
            st.stop()

        # Compute per-prompt similarity and scaled score
        try:
            p_vecs = [get_vec(w) for w in prompts]
            g_vec = get_vec(g)
            sims = [clamp(cosine_sim(g_vec, p_vecs[i])) for i in range(3)]
            scores = [score_from_similarity(s) for s in sims]
        except Exception as e:
            st.error(f"Error scoring: {e}")
            st.stop()

        avg = float(sum(scores) / 3.0)
        category = score_category(avg)

        # Persist results and lock the day
        st.session_state["scores"] = scores
        st.session_state["sims"] = sims
        st.session_state["guess"] = g
        st.session_state["avg"] = avg
        st.session_state["category"] = category

        cat_emoji = category_emoji(category)
        # Build share texts (with and without guess)
        share_lines_with = [
            f"Orthogonle — {today_et.isoformat()}",
            f"Guess: {g}",
            f"1️⃣ {prompts[0]} — {scores[0]:.1f}",
            f"2️⃣ {prompts[1]} — {scores[1]:.1f}",
            f"3️⃣ {prompts[2]} — {scores[2]:.1f}",
            f"Average: {avg:.1f} — {category} {cat_emoji}",
        ]
        share_lines_without = [
            f"Orthogonle — {today_et.isoformat()}",
            f"1️⃣ {prompts[0]} — {scores[0]:.1f}",
            f"2️⃣ {prompts[1]} — {scores[1]:.1f}",
            f"3️⃣ {prompts[2]} — {scores[2]:.1f}",
            f"Average: {avg:.1f} — {category} {cat_emoji}",
        ]
        st.session_state["share_text_with"] = "\n".join(share_lines_with)
        st.session_state["share_text_without"] = "\n".join(share_lines_without)

        st.session_state[PLAYED_KEY] = True
        st.rerun()

# -----------------------------
# Results and Share (shown if the user scored today)
# -----------------------------
if st.session_state.get(PLAYED_KEY, False):
    scores = st.session_state.get("scores", [])
    sims = st.session_state.get("sims", [])
    guess = st.session_state.get("guess", "")
    avg = st.session_state.get("avg", 0.0)
    category = st.session_state.get("category", "Novice")
    cat_emoji = category_emoji(category)
    share_text_with = st.session_state.get("share_text_with", "")
    share_text_without = st.session_state.get("share_text_without", "")

    st.subheader("Results")
    if scores and sims and guess:
        st.write(f"Your guess: {guess}")
        for i in range(3):
            st.write(f"{i+1}. {prompts[i]} — similarity {sims[i]:.3f} → score {scores[i]:.1f}")
        st.success(f"Average Score: {avg:.1f} • Category: {category} {cat_emoji}")

    st.markdown("### Share your Score")
    escaped_with = html.escape(share_text_with)
    escaped_without = html.escape(share_text_without)

    # Two copy buttons (with and without guess)
    components.html(
        f"""
        <div>
          <button id="copy_with" style="padding:0.6rem 1rem; border-radius:8px; border:1px solid #ccc; background:#fff; cursor:pointer; width:100%; margin-bottom:8px;">
            Copy score (with guess)
          </button>
          <button id="copy_without" style="padding:0.6rem 1rem; border-radius:8px; border:1px solid #ccc; background:#fff; cursor:pointer; width:100%;">
            Copy score (no guess)
          </button>
          <textarea id="payload_with" style="position:absolute; left:-9999px; top:-9999px;">{escaped_with}</textarea>
          <textarea id="payload_without" style="position:absolute; left:-9999px; top:-9999px;">{escaped_without}</textarea>
        </div>
        <script>
          async function copyText(idBtn, idTa, defaultLabel) {{
            const btn = document.getElementById(idBtn);
            try {{
              const text = document.getElementById(idTa).value;
              if (navigator.clipboard && navigator.clipboard.writeText) {{
                await navigator.clipboard.writeText(text);
              }} else {{
                const ta = document.getElementById(idTa);
                ta.style.position='fixed'; ta.style.left='-9999px';
                ta.select(); document.execCommand('copy');
                ta.style.position='absolute';
              }}
              btn.innerText = 'Copied!';
              setTimeout(() => btn.innerText = defaultLabel, 2000);
            }} catch (e) {{
              btn.innerText = 'Copy failed';
              setTimeout(() => btn.innerText = defaultLabel, 2000);
            }}
          }}
          document.getElementById('copy_with').addEventListener('click', () => copyText('copy_with','payload_with','Copy score (with guess)'));
          document.getElementById('copy_without').addEventListener('click', () => copyText('copy_without','payload_without','Copy score (no guess)'));
        </script>
        """,
        height=120,
    )

    with st.expander("Show share text (manual copy) — with guess"):
        st.text(share_text_with)
    with st.expander("Show share text (manual copy) — no guess"):
        st.text(share_text_without)