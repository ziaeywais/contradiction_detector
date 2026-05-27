"""
Contradiction Detector — Streamlit UI v2

A clean, modern interface for cross-document contradiction detection.
Run:  streamlit run streamlit_app.py
"""

import os
import sys
import json
import time
import tempfile
import shutil
import traceback
from pathlib import Path
from datetime import timedelta
from collections import defaultdict

import streamlit as st
import plotly.graph_objects as go

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)

from src.ingestion import load_corpus
from src.vectorstore import VectorStore
from src.detector import (
    find_candidate_pairs,
    analyze_pair_with_claude,
    analyze_pair_with_ollama,
    analyze_pair_with_both,
    deduplicate_contradictions,
)
from src.models import (
    AnalysisReport,
    ContradictionEvidence,
    ContradictionType,
    ConfidenceLevel,
    SeverityLevel,
    DocumentChunk,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Contradiction Detector",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,400&display=swap');

/* ─── Tokens ─────────────────────────────────────────────────────────── */
:root {
  --bg-base:   #060b16;
  --bg-1:      #0c1525;
  --bg-2:      #111e33;
  --bg-3:      #162038;
  --border:    rgba(255,255,255,0.07);
  --border-hi: rgba(99,102,241,0.35);
  --text-0:    #f1f5f9;
  --text-1:    #94a3b8;
  --text-2:    #4a5568;
  --accent:    #6366f1;
  --accent-2:  #8b5cf6;
  --cyan:      #22d3ee;
  --purple:    #a78bfa;
  --critical:  #ef4444;
  --major:     #f59e0b;
  --minor:     #3b82f6;
  --success:   #10b981;
  --r-sm: 8px; --r-md: 12px; --r-lg: 16px;
}

/* ─── Global ─────────────────────────────────────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
.main .block-container {
  font-family: 'Inter', sans-serif !important;
  background: var(--bg-base) !important;
  color: var(--text-0) !important;
}
.main .block-container {
  padding: 1.6rem 2.2rem 4rem !important;
  max-width: 1260px !important;
}

/* ─── Sidebar ─────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: #050a14 !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: 'Inter', sans-serif !important; }

/* ─── Icon font fix ──────────────────────────────────────────────────── */
/* The sidebar wildcard rule above breaks Material Symbols Rounded icons  */
/* (the icon text "upload", "keyboard_arrow_down", etc. shows as literal  */
/* text instead of the glyph). Restore the icon font with higher          */
/* specificity so it wins over the !important wildcard.                   */
[data-testid="stSidebar"] [data-testid="stIconMaterial"],
[data-testid="stIconMaterial"] {
  font-family: 'Material Symbols Rounded' !important;
  font-feature-settings: 'liga' !important;
  -webkit-font-feature-settings: 'liga' !important;
  font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24 !important;
  text-transform: none !important;
  letter-spacing: normal !important;
  word-spacing: normal !important;
  white-space: nowrap !important;
}

/* ─── Typography helpers ─────────────────────────────────────────────── */
.label {
  font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.1em; color: var(--text-2); margin-bottom: 0.5rem;
}

/* ─── Divider ─────────────────────────────────────────────────────────── */
hr { border: none; border-top: 1px solid var(--border) !important; margin: 1.2rem 0 !important; }

/* ─── Metric strip ───────────────────────────────────────────────────── */
.metric-strip {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 0.65rem;
  margin-bottom: 1.5rem;
}
@media (max-width: 900px) { .metric-strip { grid-template-columns: repeat(3, 1fr); } }
.metric-tile {
  background: var(--bg-1);
  border: 1px solid var(--border);
  border-radius: var(--r-md);
  padding: 0.9rem 1rem 0.8rem;
  text-align: center;
}
.metric-val {
  font-size: 1.85rem; font-weight: 800; color: var(--text-0);
  line-height: 1; letter-spacing: -0.03em;
}
.metric-key {
  font-size: 0.66rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.08em; color: var(--text-2); margin-top: 0.3rem;
}

/* ─── Panel card ─────────────────────────────────────────────────────── */
.panel {
  background: var(--bg-1);
  border: 1px solid var(--border);
  border-radius: var(--r-lg);
  padding: 1.3rem 1.5rem;
  margin-bottom: 1rem;
}

/* ─── Badges ─────────────────────────────────────────────────────────── */
.badge {
  display: inline-flex; align-items: center; gap: 0.25rem;
  padding: 0.2rem 0.6rem; border-radius: 6px;
  font-size: 0.65rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.06em; white-space: nowrap;
}
.sev-critical { background: rgba(239,68,68,.14); color: #f87171; border: 1px solid rgba(239,68,68,.22); }
.sev-major    { background: rgba(245,158,11,.14); color: #fbbf24; border: 1px solid rgba(245,158,11,.22); }
.sev-minor    { background: rgba(59,130,246,.14);  color: #60a5fa; border: 1px solid rgba(59,130,246,.22); }
.conf-high    { background: rgba(16,185,129,.12);  color: #34d399; border: 1px solid rgba(16,185,129,.2); }
.conf-medium  { background: rgba(245,158,11,.10);  color: #fbbf24; border: 1px solid rgba(245,158,11,.2); }
.conf-low     { background: rgba(100,116,139,.12); color: #94a3b8; border: 1px solid rgba(100,116,139,.2); }
.type-badge   { background: rgba(99,102,241,.1);   color: #a78bfa; border: 1px solid rgba(99,102,241,.2); }
.sim-badge    { background: rgba(255,255,255,.05);  color: var(--text-1); border: 1px solid var(--border); }

/* ─── Contradiction card ─────────────────────────────────────────────── */
.ccard {
  border-radius: var(--r-lg);
  padding: 1.3rem 1.5rem 1.2rem;
  margin-bottom: 0.9rem;
  border: 1px solid;
  transition: transform .17s ease, box-shadow .17s ease, border-color .17s ease;
}
.ccard:hover { transform: translateY(-2px); box-shadow: 0 12px 40px rgba(0,0,0,.45); }
.ccard-critical { background: rgba(239,68,68,.04);  border-color: rgba(239,68,68,.18); }
.ccard-major    { background: rgba(245,158,11,.04); border-color: rgba(245,158,11,.18); }
.ccard-minor    { background: rgba(59,130,246,.04); border-color: rgba(59,130,246,.18); }
.ccard-critical:hover { border-color: rgba(239,68,68,.38); }
.ccard-major:hover    { border-color: rgba(245,158,11,.38); }
.ccard-minor:hover    { border-color: rgba(59,130,246,.38); }

.ccard-top {
  display: flex; align-items: flex-start;
  justify-content: space-between; gap: 1rem;
  margin-bottom: 1rem;
}
.ccard-title-row { display: flex; align-items: center; gap: 0.55rem; flex-wrap: wrap; }
.ccard-num { font-size: 0.72rem; color: var(--text-2); font-weight: 600; }
.ccard-topic {
  font-size: 0.95rem; font-weight: 700; color: var(--text-0);
  letter-spacing: -.01em; text-transform: uppercase;
}
.ccard-badges { display: flex; gap: 0.35rem; flex-wrap: wrap; justify-content: flex-end; }

/* ─── Claims comparison ──────────────────────────────────────────────── */
.claims-row {
  display: grid;
  grid-template-columns: 1fr 36px 1fr;
  gap: 0;
  margin: 0.8rem 0;
  align-items: stretch;
}
.claim-col {
  background: rgba(0,0,0,.22);
  border-radius: var(--r-sm);
  padding: 0.85rem 1rem;
}
.claim-who {
  font-size: 0.69rem; font-weight: 700; letter-spacing: .05em;
  text-transform: uppercase; margin-bottom: 0.45rem;
  display: flex; align-items: center; gap: 0.3rem;
}
.claim-who-a { color: var(--cyan); }
.claim-who-b { color: var(--purple); }
.claim-text { font-size: 0.875rem; line-height: 1.58; color: var(--text-0); font-style: italic; }
.claims-vs {
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  gap: 0.3rem;
}
.vs-line { width: 1px; flex: 1; background: var(--border); }
.vs-label {
  font-size: 0.6rem; font-weight: 800; color: var(--text-2);
  letter-spacing: .07em; text-transform: uppercase;
}

/* ─── Explanation ────────────────────────────────────────────────────── */
.ccard-explain {
  font-size: 0.87rem; line-height: 1.65; color: var(--text-1);
  padding: 0.8rem 1rem;
  background: rgba(99,102,241,.05);
  border: 1px solid rgba(99,102,241,.09);
  border-radius: var(--r-sm);
  margin-top: 0.75rem;
}

/* ─── Document-pair group header ─────────────────────────────────────── */
.pair-header {
  display: flex; align-items: center; gap: 0.65rem;
  padding: 0.7rem 1rem;
  background: var(--bg-2);
  border: 1px solid var(--border);
  border-radius: var(--r-md);
  margin: 1.2rem 0 0.5rem;
}
.pair-doc { font-size: 0.82rem; font-weight: 600; color: var(--text-0); }
.pair-sep { color: var(--text-2); font-size: 0.8rem; }
.pair-count {
  margin-left: auto;
  font-size: 0.7rem; font-weight: 700;
  background: rgba(99,102,241,.12); color: var(--purple);
  border: 1px solid rgba(99,102,241,.2);
  border-radius: 20px; padding: 0.15rem 0.6rem;
}

/* ─── Welcome ────────────────────────────────────────────────────────── */
.welcome {
  max-width: 580px; margin: 3rem auto; text-align: center;
}
.welcome-icon { font-size: 3.8rem; margin-bottom: 1rem; line-height: 1; }
.welcome h2 {
  font-size: 1.55rem; font-weight: 800; color: var(--text-0);
  letter-spacing: -.025em; margin: 0 0 0.6rem;
}
.welcome p { color: var(--text-1); font-size: 0.94rem; line-height: 1.65; }
.how-it-works {
  display: grid; grid-template-columns: repeat(3, 1fr);
  gap: 0.75rem; margin-top: 2rem; text-align: left;
}
.step-card {
  background: var(--bg-1); border: 1px solid var(--border);
  border-radius: var(--r-md); padding: 1rem;
}
.step-num {
  font-size: 0.65rem; font-weight: 800; text-transform: uppercase;
  letter-spacing: .1em; color: var(--accent); margin-bottom: .4rem;
}
.step-title { font-size: 0.85rem; font-weight: 700; color: var(--text-0); margin-bottom: .25rem; }
.step-desc { font-size: 0.78rem; color: var(--text-1); line-height: 1.5; }

/* ─── Empty / no-results ─────────────────────────────────────────────── */
.empty-state { text-align: center; padding: 3rem 2rem; }
.empty-icon { font-size: 2.8rem; margin-bottom: 0.8rem; }
.empty-title { font-size: 1.1rem; font-weight: 700; color: var(--text-1); margin-bottom: .4rem; }
.empty-sub { font-size: 0.88rem; color: var(--text-2); line-height: 1.6; }

/* ─── Success banner ─────────────────────────────────────────────────── */
.success-banner {
  background: rgba(16,185,129,.08);
  border: 1px solid rgba(16,185,129,.2);
  border-radius: var(--r-md);
  padding: 1rem 1.4rem;
  display: flex; align-items: center; gap: 0.8rem;
  margin-bottom: 1rem;
}
.success-banner-icon { font-size: 1.4rem; flex-shrink: 0; }
.success-banner-text { font-size: 0.9rem; color: #6ee7b7; }
.success-banner-text strong { color: #34d399; }

/* ─── Streamlit overrides ────────────────────────────────────────────── */
.stButton > button {
  background: linear-gradient(135deg, #6366f1, #7c3aed) !important;
  color: #fff !important; border: none !important;
  border-radius: var(--r-sm) !important; font-weight: 600 !important;
  font-family: 'Inter', sans-serif !important;
  padding: 0.5rem 1.25rem !important; font-size: 0.88rem !important;
  letter-spacing: .01em !important;
  transition: opacity .15s, transform .15s !important;
}
.stButton > button:hover { opacity: .88 !important; transform: translateY(-1px) !important; }
.stButton > button:disabled {
  background: rgba(255,255,255,.06) !important;
  color: var(--text-2) !important;
  opacity: 1 !important; transform: none !important;
}
.stDownloadButton > button {
  background: rgba(16,185,129,.09) !important; color: #34d399 !important;
  border: 1px solid rgba(16,185,129,.22) !important;
  border-radius: var(--r-sm) !important; font-weight: 600 !important;
  font-family: 'Inter', sans-serif !important;
}
.stDownloadButton > button:hover { background: rgba(16,185,129,.16) !important; }

[data-testid="stExpander"] {
  background: rgba(12,21,37,.7) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important;
}
[data-testid="stTabs"] { gap: 0; }
[data-testid="stTabs"] button {
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important; font-size: 0.83rem !important;
}
[data-testid="stTextInput"] > div > div > input {
  background: var(--bg-2) !important; color: var(--text-0) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-sm) !important;
  font-family: 'Inter', sans-serif !important;
}
[data-testid="stTextInput"] > div > div > input:focus {
  border-color: var(--border-hi) !important;
  box-shadow: 0 0 0 3px rgba(99,102,241,.12) !important;
}

/* ─── Hide Streamlit chrome ──────────────────────────────────────────── */
#MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden; }

/* ─── Scrollbar ──────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,.5); }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
_SEV_ICON  = {"CRITICAL": "🔴", "MAJOR": "🟡", "MINOR": "🔵"}
_SEV_COLOR = {"CRITICAL": "#ef4444", "MAJOR": "#f59e0b", "MINOR": "#3b82f6"}
_SEV_ORDER = {"CRITICAL": 0, "MAJOR": 1, "MINOR": 2}
_CONF_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
_FILE_ICON = {".txt": "📄", ".md": "📝", ".pdf": "📋", ".docx": "📃"}
SAMPLE_CORPUS = PROJECT_ROOT / "sample_corpus"


# ── Tiny helpers ──────────────────────────────────────────────────────────────

def _sev(ev):
    return ev.severity.value if hasattr(ev.severity, "value") else ev.severity

def _conf(ev):
    return ev.confidence.value if hasattr(ev.confidence, "value") else ev.confidence

def _ctype(ev):
    t = ev.contradiction_type.value if hasattr(ev.contradiction_type, "value") else ev.contradiction_type
    return t.capitalize()

def _fmt_bytes(n: int) -> str:
    return f"{n/1024:.1f} KB" if n >= 1024 else f"{n} B"

def _matches_search(ev: ContradictionEvidence, q: str) -> bool:
    if not q:
        return True
    q = q.lower()
    return any(
        q in s.lower()
        for s in [ev.topic, ev.claim_a, ev.claim_b, ev.explanation,
                  ev.chunk_a.document_name, ev.chunk_b.document_name]
    )


# ── Temp corpus ───────────────────────────────────────────────────────────────

def _prepare_temp_corpus(file_cache: dict) -> str:
    tmp = tempfile.mkdtemp(prefix="contradiction_corpus_")
    for name, content in file_cache.items():
        (Path(tmp) / name).write_bytes(content)
    return tmp


# ── Exports ───────────────────────────────────────────────────────────────────

def _report_to_json(report: AnalysisReport) -> str:
    return json.dumps(report.model_dump(mode="json"), indent=2, default=str)


def _report_to_csv(report: AnalysisReport) -> str:
    import csv, io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["#", "Topic", "Type", "Severity", "Confidence",
                "Doc A", "Claim A", "Doc B", "Claim B",
                "Explanation", "Similarity", "Detected By"])
    for i, c in enumerate(report.contradictions_found, 1):
        w.writerow([
            i, c.topic, _ctype(c), _sev(c), _conf(c),
            c.chunk_a.document_name, c.claim_a,
            c.chunk_b.document_name, c.claim_b,
            c.explanation, f"{c.similarity_score:.4f}", c.detected_by,
        ])
    return buf.getvalue()


# ── Chart ─────────────────────────────────────────────────────────────────────

def _severity_donut(contradictions: list):
    counts = {"CRITICAL": 0, "MAJOR": 0, "MINOR": 0}
    for c in contradictions:
        counts[_sev(c)] += 1

    labels, values, colors = [], [], []
    for sev in ["CRITICAL", "MAJOR", "MINOR"]:
        if counts[sev]:
            labels.append(sev)
            values.append(counts[sev])
            colors.append(_SEV_COLOR[sev])
    if not values:
        return None

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.68,
        marker=dict(colors=colors, line=dict(color="#0c1525", width=3)),
        textinfo="label+value",
        textfont=dict(size=11, family="Inter", color="#f1f5f9"),
        hovertemplate="<b>%{label}</b><br>%{value} found · %{percent}<extra></extra>",
    )])
    fig.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=8, b=8), height=210,
        annotations=[dict(
            text=f"<b>{sum(values)}</b>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=34, family="Inter", color="#f1f5f9"),
        )],
    )
    return fig


# ── Contradiction card ────────────────────────────────────────────────────────

def _contradiction_card(ev: ContradictionEvidence, idx: int):
    sev  = _sev(ev)
    conf = _conf(ev)
    typ  = _ctype(ev)

    sim_badge = (
        f'<span class="badge sim-badge">{ev.similarity_score:.0%} similar</span>'
        if ev.similarity_score > 0 else ""
    )

    html = f"""
<div class="ccard ccard-{sev.lower()}">
  <div class="ccard-top">
    <div class="ccard-title-row">
      <span style="font-size:1.25rem;line-height:1">{_SEV_ICON[sev]}</span>
      <span class="ccard-num">#{idx}</span>
      <span class="ccard-topic">{ev.topic}</span>
    </div>
    <div class="ccard-badges">
      <span class="badge sev-{sev.lower()}">{sev}</span>
      <span class="badge conf-{conf.lower()}">{conf}</span>
      <span class="badge type-badge">{typ}</span>
      {sim_badge}
    </div>
  </div>

  <div class="claims-row">
    <div class="claim-col">
      <div class="claim-who claim-who-a">
        📄 {ev.chunk_a.document_name}
        <span style="font-weight:400;opacity:.55;font-size:.65rem">· chunk {ev.chunk_a.chunk_index}</span>
      </div>
      <div class="claim-text">&ldquo;{ev.claim_a}&rdquo;</div>
    </div>
    <div class="claims-vs">
      <div class="vs-line"></div>
      <span class="vs-label">vs</span>
      <div class="vs-line"></div>
    </div>
    <div class="claim-col">
      <div class="claim-who claim-who-b">
        📄 {ev.chunk_b.document_name}
        <span style="font-weight:400;opacity:.55;font-size:.65rem">· chunk {ev.chunk_b.chunk_index}</span>
      </div>
      <div class="claim-text">&ldquo;{ev.claim_b}&rdquo;</div>
    </div>
  </div>

  <div class="ccard-explain">💡 {ev.explanation}</div>
</div>"""
    st.markdown(html, unsafe_allow_html=True)


# ── LLM clients ───────────────────────────────────────────────────────────────

def _claude_client():
    import anthropic
    try:
        return anthropic.Anthropic()
    except Exception as e:
        st.error(f"❌ Failed to initialise Claude: {e}")
        return None

def _ollama_client(host=None):
    try:
        import ollama
    except ImportError:
        st.error("❌ `ollama` package not installed. Run: `pip install ollama`")
        return None
    return ollama.Client(host=host or os.getenv("OLLAMA_HOST", "http://localhost:11434"))


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar():
    """Draw the sidebar and return all scan configuration values."""
    with st.sidebar:
        # ── Brand ──
        st.markdown("""
        <div style="padding:.9rem 0 1.1rem;border-bottom:1px solid rgba(255,255,255,.07);margin-bottom:1rem;">
          <div style="display:flex;align-items:center;gap:.65rem;">
            <div style="width:36px;height:36px;border-radius:10px;
              background:linear-gradient(135deg,#6366f1,#8b5cf6);
              display:flex;align-items:center;justify-content:center;font-size:1.15rem;flex-shrink:0">⚡</div>
            <div>
              <div style="font-size:.95rem;font-weight:800;color:#f1f5f9;letter-spacing:-.02em">
                Contradiction Detector</div>
              <div style="font-size:.68rem;color:#4a5568">RAG + LLM reasoning</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── Documents ──
        st.markdown('<div class="label">📁 Documents</div>', unsafe_allow_html=True)

        source_mode = st.radio(
            "source", ["Upload Files", "Directory Path"],
            horizontal=True, label_visibility="collapsed",
        )

        if "file_cache" not in st.session_state:
            st.session_state.file_cache = {}
        if "uploader_key" not in st.session_state:
            st.session_state.uploader_key = 0

        corpus_path_input = ""

        if source_mode == "Upload Files":
            # Show the sample corpus shortcut ONLY when no files are loaded yet
            # so it can never accidentally overwrite the user's own uploads.
            if SAMPLE_CORPUS.is_dir() and not st.session_state.file_cache:
                if st.button("🧪 Try with sample corpus", use_container_width=True,
                             help="Load 6 demo documents that contain embedded contradictions"):
                    for f in sorted(SAMPLE_CORPUS.iterdir()):
                        if f.suffix.lower() in {".txt", ".md", ".pdf", ".docx"}:
                            st.session_state.file_cache[f.name] = f.read_bytes()
                    st.session_state.uploader_key += 1
                    st.rerun()
                st.caption("— or upload your own files below —")

            uploaded = st.file_uploader(
                "Drop files here",
                type=["txt", "md", "pdf", "docx"],
                accept_multiple_files=True,
                key=f"up_{st.session_state.uploader_key}",
                label_visibility="collapsed",
                help="Select all files at once in the picker (Shift-click / Cmd-click). Supported: .txt  .md  .pdf  .docx",
            )
            # Accumulate uploaded files into the persistent cache.
            # Key = original filename; same name overwrites (same doc re-uploaded).
            if uploaded:
                for f in uploaded:
                    new_bytes = f.getvalue()
                    if st.session_state.file_cache.get(f.name) != new_bytes:
                        st.session_state.file_cache[f.name] = new_bytes

            cache = st.session_state.file_cache
            if cache:
                for fname, content in list(cache.items()):
                    # fname IS the original filename — no stripping needed
                    ext  = Path(fname).suffix.lower()
                    icon = _FILE_ICON.get(ext, "📄")
                    c1, c2 = st.columns([9, 1])
                    c1.markdown(
                        f'<div style="display:flex;align-items:center;gap:.45rem;'
                        f'padding:.38rem .6rem;border-radius:7px;'
                        f'background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);'
                        f'margin-bottom:.3rem;">'
                        f'<span style="font-size:.85rem">{icon}</span>'
                        f'<span style="font-size:.78rem;color:#f1f5f9;flex:1;'
                        f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="{fname}">'
                        f'{fname}</span>'
                        f'<span style="font-size:.67rem;color:#4a5568;flex-shrink:0">'
                        f'{_fmt_bytes(len(content))}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if c2.button("✕", key=f"rm_{fname}", help="Remove"):
                        del st.session_state.file_cache[fname]
                        st.session_state.uploader_key += 1
                        st.rerun()

                n = len(cache)
                if n < 2:
                    st.caption("⚠️ Add one more document to scan")
                else:
                    st.caption(f"✅ {n} document{'s' if n>1 else ''} ready")

                _, c_clear = st.columns([2, 1])
                if c_clear.button("Clear all", use_container_width=True,
                                  help="Remove all loaded documents"):
                    st.session_state.file_cache.clear()
                    st.session_state.uploader_key += 1
                    st.rerun()
        else:
            corpus_path_input = st.text_input(
                "Directory path",
                value="./sample_corpus",
                label_visibility="collapsed",
                placeholder="./my_documents",
            )

        st.divider()

        # ── Provider ──
        st.markdown('<div class="label">⚡ LLM Provider</div>', unsafe_allow_html=True)
        provider = st.selectbox(
            "provider", ["claude", "ollama", "both"],
            format_func=lambda x: {
                "claude": "☁️  Claude Opus 4.7  —  best accuracy",
                "ollama": "🏠  Ollama  —  free, fully local",
                "both":   "⚡  Both  —  Ollama filters → Claude verifies",
            }[x],
            label_visibility="collapsed",
        )

        if provider in ("claude", "both"):
            api_key = os.getenv("ANTHROPIC_API_KEY", "")
            # Show input when key is absent, a placeholder, or obviously wrong format
            key_looks_valid = bool(api_key and not api_key.startswith("your") and api_key.startswith("sk-ant-"))
            if not key_looks_valid:
                typed = st.text_input(
                    "Anthropic API key",
                    type="password", placeholder="sk-ant-...",
                    help="Or set ANTHROPIC_API_KEY in .env",
                )
                if typed:
                    os.environ["ANTHROPIC_API_KEY"] = typed
                    api_key = typed
                    key_looks_valid = True
            if key_looks_valid:
                st.caption("✅ API key loaded — will validate on first scan")

        st.divider()

        # ── Advanced ──
        with st.expander("⚙️ Advanced settings"):
            min_similarity = st.slider(
                "Min similarity", 0.0, 1.0, 0.65, 0.05,
                help="Lower = more pairs checked, but more noise. Default 0.65 works well.",
            )
            max_pairs = st.number_input("Max pairs to analyse", 10, 500, 50, 10)
            n_similar  = st.number_input("Similar chunks per query", 1, 20, 5)
            if provider in ("ollama", "both"):
                ollama_model = st.text_input("Ollama model", os.getenv("OLLAMA_MODEL", "llama3.2"))
                ollama_host  = st.text_input("Ollama host",  os.getenv("OLLAMA_HOST", "http://localhost:11434"))
            else:
                ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
                ollama_host  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            reset_db = st.checkbox("Reset vector DB", False,
                                   help="Clear cached embeddings before this scan.")

        st.divider()

        # ── Scan button ──
        can_scan = bool(
            (source_mode == "Upload Files" and len(st.session_state.file_cache) >= 2)
            or (source_mode == "Directory Path" and corpus_path_input.strip())
        )
        scan_clicked = st.button(
            "⚡  Scan for Contradictions",
            use_container_width=True,
            disabled=not can_scan,
            type="primary",
        )
        if not can_scan and source_mode == "Upload Files":
            st.caption("Upload at least 2 valid documents to begin.")

    return dict(
        source_mode=source_mode,
        corpus_path_input=corpus_path_input,
        provider=provider,
        ollama_model=ollama_model,
        ollama_host=ollama_host,
        min_similarity=min_similarity,
        max_pairs=int(max_pairs),
        n_similar=int(n_similar),
        reset_db=reset_db,
        scan_clicked=scan_clicked,
        can_scan=can_scan,
    )


# ── Welcome screen ────────────────────────────────────────────────────────────

def _render_welcome():
    st.markdown("""
    <div class="welcome">
      <div class="welcome-icon">⚡</div>
      <h2>Find where your documents disagree</h2>
      <p>
        Upload two or more documents — contracts, policies, guidelines, research papers —
        and this tool will surface every place where they make incompatible claims.
      </p>
    </div>
    <div class="how-it-works">
      <div class="step-card">
        <div class="step-num">Step 1 · Retrieve</div>
        <div class="step-title">Semantic Search</div>
        <div class="step-desc">
          Sentence-transformers embed every passage locally.
          ChromaDB finds cross-document pairs that cover the same topic.
        </div>
      </div>
      <div class="step-card">
        <div class="step-num">Step 2 · Reason</div>
        <div class="step-title">LLM Analysis</div>
        <div class="step-desc">
          Claude Opus 4.7 with adaptive thinking inspects each candidate pair
          and judges whether they genuinely contradict each other.
        </div>
      </div>
      <div class="step-card">
        <div class="step-num">Step 3 · Report</div>
        <div class="step-title">Structured Results</div>
        <div class="step-desc">
          Contradictions are ranked by severity, grouped by document pair,
          and exported to JSON or CSV for downstream workflows.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if SAMPLE_CORPUS.is_dir():
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(
            "**Try the sample corpus** — click *Load sample corpus* in the sidebar. "
            "It contains 6 documents with 8 embedded contradictions across HR policies, "
            "software contracts, and medical guidelines.",
            icon="🧪",
        )


# ── Scan execution ────────────────────────────────────────────────────────────

def _run_scan(cfg: dict):
    """Execute the full scan pipeline, updating st.session_state.report on success."""
    source_mode   = cfg["source_mode"]
    provider      = cfg["provider"]
    ollama_model  = cfg["ollama_model"]
    ollama_host   = cfg["ollama_host"]
    min_similarity = cfg["min_similarity"]
    max_pairs     = cfg["max_pairs"]
    n_similar     = cfg["n_similar"]
    reset_db      = cfg["reset_db"]

    tmp_dir = None
    st.session_state.scan_error = None

    try:
        # Resolve corpus directory
        if source_mode == "Upload Files":
            tmp_dir = _prepare_temp_corpus(st.session_state.file_cache)
            corpus_path = tmp_dir
        else:
            corpus_path = str(Path(cfg["corpus_path_input"]).resolve())

        with st.status("Scanning…", expanded=True) as status:
            start = time.time()

            # ── 1. Load documents ──────────────────────────────────
            st.write("📂 Loading documents…")
            try:
                all_chunks = load_corpus(corpus_path)
            except ValueError as e:
                st.error(str(e))
                return

            doc_names = sorted({c.document_name for c in all_chunks})
            st.write(f"✅ **{len(all_chunks)}** chunks from **{len(doc_names)}** document(s)")

            # Detect which uploaded files produced no chunks (empty / unreadable)
            if source_mode == "Upload Files":
                attempted_names = set(st.session_state.file_cache.keys())
                failed_names    = sorted(attempted_names - set(doc_names))
                if failed_names:
                    # Build a tailored hint based on the failing file types
                    _exts = {Path(n).suffix.lower() for n in failed_names}
                    if _exts <= {".docx"}:
                        _hint = (
                            "**Common causes for .docx:** the file may be an old binary `.doc` "
                            "renamed to `.docx`, password-protected, or corrupted.\n\n"
                            "**Fix:** open in Microsoft Word / LibreOffice, then *Save As* → `.docx` "
                            "or export as `.txt`."
                        )
                    elif _exts <= {".pdf"}:
                        _hint = (
                            "**Common causes:** scanned / image-only PDF (no OCR text layer) or "
                            "password-protected PDF.\n\n"
                            "**Fix:** export your PDF as a *text-based* PDF, or use `.docx` / `.txt` instead."
                        )
                    else:
                        _hint = (
                            "**Common causes:** scanned image PDF, password-protected file, old binary "
                            "`.doc` renamed to `.docx`, or a corrupted file.\n\n"
                            "**Fix:** re-export as a text-based PDF or `.docx` / `.txt`."
                        )
                    st.warning(
                        f"⚠️ **{len(failed_names)} file(s) produced no extractable text** and were skipped: "
                        + ", ".join(f"`{n}`" for n in failed_names)
                        + f"\n\n{_hint}"
                    )
            else:
                failed_names = []

            if len(doc_names) < 2:
                n_att = len(st.session_state.file_cache) if source_mode == "Upload Files" else "?"
                st.error(
                    f"Need at least 2 documents with extractable text to detect contradictions.\n\n"
                    + (
                        f"Uploaded **{n_att}** file(s), but only **{len(doc_names)}** produced readable content. "
                        if source_mode == "Upload Files" else
                        f"Only **{len(doc_names)}** document(s) could be loaded from the directory. "
                    )
                    + "See the warning above for which files failed and how to fix them."
                )
                return

            # ── 2. Embeddings ──────────────────────────────────────
            st.write("🧠 Building embeddings…")
            db_path    = str(Path(corpus_path) / ".contradiction_db")
            vectorstore = VectorStore(persist_directory=db_path)
            if reset_db:
                vectorstore.clear()

            prog = st.progress(0, text="Embedding…")
            batch_size = 50
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i : i + batch_size]
                vectorstore.add_chunks(batch)
                pct = min((i + len(batch)) / len(all_chunks), 1.0)
                prog.progress(pct, text=f"Embedding… {min(i+len(batch), len(all_chunks))}/{len(all_chunks)}")
            prog.progress(1.0, text="✅ Embeddings done")

            # ── 3. Candidate pairs ─────────────────────────────────
            st.write("🔗 Finding semantically similar cross-document pairs…")
            pairs = find_candidate_pairs(
                vectorstore, all_chunks,
                n_similar=n_similar, min_similarity=min_similarity,
            )

            if not pairs:
                st.warning(
                    f"No candidate pairs found above similarity **{min_similarity:.0%}**. "
                    "Try lowering the *Min similarity* threshold in Advanced settings."
                )
                st.session_state.report = AnalysisReport(
                    corpus_path=corpus_path, documents_analyzed=doc_names,
                    total_chunks=len(all_chunks), candidate_pairs_examined=0,
                    contradictions_found=[], analysis_duration_seconds=time.time()-start,
                    model_used=ollama_model if provider=="ollama" else "claude-opus-4-7",
                    provider=provider,
                )
                status.update(label="✅ Scan complete — no candidate pairs found", state="complete")
                return

            if len(pairs) > max_pairs:
                pairs = pairs[:max_pairs]
            st.write(f"✅ **{len(pairs)}** candidate pair(s) to analyse")

            # ── 4. Initialise LLM clients ──────────────────────────
            anthropic_client = ollama_client = None
            if provider in ("claude", "both"):
                st.write("☁️ Connecting to Claude…")
                anthropic_client = _claude_client()
                if not anthropic_client:
                    return
            if provider in ("ollama", "both"):
                st.write("🏠 Connecting to Ollama…")
                ollama_client = _ollama_client(ollama_host)
                if not ollama_client:
                    return

            # ── 5. Analyse pairs ───────────────────────────────────
            st.write(f"🧪 Analysing pairs with **{provider}**…")
            tokens_used  = [0]
            contradictions = []
            failed = 0

            bar = st.progress(0, text="Analysing…")
            auth_error_shown = False
            for i, (chunk_a, chunk_b, sim) in enumerate(pairs):
                try:
                    if provider == "claude":
                        result = analyze_pair_with_claude(
                            anthropic_client, chunk_a, chunk_b, tokens_used, sim)
                    elif provider == "ollama":
                        result = analyze_pair_with_ollama(
                            ollama_client, ollama_model, chunk_a, chunk_b, sim)
                    else:
                        result = analyze_pair_with_both(
                            anthropic_client, ollama_client, ollama_model,
                            chunk_a, chunk_b, tokens_used, sim)

                    if result is not None:
                        contradictions.append(result)

                except Exception as exc:
                    exc_str = str(exc)
                    # Check for authentication errors (invalid/expired API key)
                    if "authentication_error" in exc_str or "401" in exc_str or "invalid x-api-key" in exc_str.lower():
                        if not auth_error_shown:
                            auth_error_shown = True
                            st.error(
                                "❌ **Invalid Anthropic API key.**\n\n"
                                "Your API key was rejected (401). Please update `ANTHROPIC_API_KEY` "
                                "in your `.env` file with a valid key from "
                                "[console.anthropic.com](https://console.anthropic.com/), "
                                "then restart the app."
                            )
                        # Don't count each pair as a separate failure
                        failed += len(pairs) - i
                        break  # No point retrying remaining pairs
                    else:
                        failed += 1
                        print(f"[contradiction-detector] pair error: {exc}\n{traceback.format_exc()}")

                bar.progress(
                    (i + 1) / len(pairs),
                    text=f"Analysing… {i+1}/{len(pairs)} · {len(contradictions)} found",
                )

            bar.progress(1.0, text=f"✅ {len(contradictions)} contradiction(s) found")

            if failed:
                st.warning(
                    f"⚠️ {failed} pair(s) failed during analysis — check the terminal for details."
                )

            # ── 6. Deduplicate ─────────────────────────────────────
            if contradictions:
                contradictions = deduplicate_contradictions(contradictions)

            # ── 7. Build report ────────────────────────────────────
            st.session_state.report = AnalysisReport(
                corpus_path=corpus_path,
                documents_analyzed=doc_names,
                total_chunks=len(all_chunks),
                candidate_pairs_examined=len(pairs),
                contradictions_found=contradictions,
                analysis_duration_seconds=time.time() - start,
                model_used=ollama_model if provider=="ollama" else "claude-opus-4-7",
                tokens_used=tokens_used[0],
                provider=provider,
            )
            label = (f"✅ Scan complete — {len(contradictions)} contradiction(s) found"
                     if contradictions else "✅ Scan complete — no contradictions found")
            status.update(label=label, state="complete")

    except Exception as exc:
        st.session_state.scan_error = str(exc)
        st.error(f"❌ Scan failed: {exc}")
        print(traceback.format_exc())
    finally:
        st.session_state.scanning = False
        if tmp_dir and Path(tmp_dir).exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Results view ──────────────────────────────────────────────────────────────

def _render_results(report: AnalysisReport):
    contradictions = report.contradictions_found

    # ── Metric strip ──
    dur = str(timedelta(seconds=int(report.analysis_duration_seconds)))
    n_crit  = sum(1 for c in contradictions if _sev(c) == "CRITICAL")
    n_major = sum(1 for c in contradictions if _sev(c) == "MAJOR")
    n_minor = sum(1 for c in contradictions if _sev(c) == "MINOR")

    st.markdown(f"""
    <div class="metric-strip">
      <div class="metric-tile">
        <div class="metric-val">{len(report.documents_analyzed)}</div>
        <div class="metric-key">Documents</div>
      </div>
      <div class="metric-tile">
        <div class="metric-val">{report.total_chunks}</div>
        <div class="metric-key">Chunks</div>
      </div>
      <div class="metric-tile">
        <div class="metric-val">{report.candidate_pairs_examined}</div>
        <div class="metric-key">Pairs checked</div>
      </div>
      <div class="metric-tile">
        <div class="metric-val">{len(contradictions)}</div>
        <div class="metric-key">Contradictions</div>
      </div>
      <div class="metric-tile">
        <div class="metric-val">{dur}</div>
        <div class="metric-key">Duration</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Summary row: chart + breakdown + meta ──
    if contradictions:
        col_chart, col_breakdown = st.columns([1, 1.3])

        with col_chart:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("**Severity Distribution**")
            fig = _severity_donut(contradictions)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with col_breakdown:
            tok_line = (
                f'<div style="font-size:.78rem;color:#4a5568;margin-top:.4rem">'
                f'Tokens used: <b style="color:#94a3b8">{report.tokens_used:,}</b></div>'
            ) if report.tokens_used else ""
            st.markdown(f"""
            <div class="panel" style="height:100%;box-sizing:border-box">
              <p style="font-weight:700;margin:0 0 .9rem;font-size:.9rem">Breakdown</p>
              <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:.6rem;margin-bottom:1rem">
                <div style="text-align:center;padding:.8rem .4rem;
                  background:rgba(239,68,68,.07);border-radius:10px;
                  border:1px solid rgba(239,68,68,.15)">
                  <div style="font-size:1.9rem;font-weight:800;color:#ef4444">{n_crit}</div>
                  <div style="font-size:.65rem;color:#f87171;text-transform:uppercase;
                    font-weight:700;letter-spacing:.08em">Critical</div>
                </div>
                <div style="text-align:center;padding:.8rem .4rem;
                  background:rgba(245,158,11,.07);border-radius:10px;
                  border:1px solid rgba(245,158,11,.15)">
                  <div style="font-size:1.9rem;font-weight:800;color:#f59e0b">{n_major}</div>
                  <div style="font-size:.65rem;color:#fbbf24;text-transform:uppercase;
                    font-weight:700;letter-spacing:.08em">Major</div>
                </div>
                <div style="text-align:center;padding:.8rem .4rem;
                  background:rgba(59,130,246,.07);border-radius:10px;
                  border:1px solid rgba(59,130,246,.15)">
                  <div style="font-size:1.9rem;font-weight:800;color:#3b82f6">{n_minor}</div>
                  <div style="font-size:.65rem;color:#60a5fa;text-transform:uppercase;
                    font-weight:700;letter-spacing:.08em">Minor</div>
                </div>
              </div>
              <div style="display:flex;gap:.5rem;flex-wrap:wrap;font-size:.78rem;color:#94a3b8">
                <span>Provider: <b style="color:#a78bfa">{report.provider.capitalize()}</b></span>
                <span>·</span>
                <span>Model: <b style="color:#a78bfa">{report.model_used}</b></span>
              </div>
              {tok_line}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

    # ── No-contradiction banner ──
    if not contradictions:
        st.markdown("""
        <div class="panel" style="text-align:center;border-color:rgba(16,185,129,.25);
          background:rgba(16,185,129,.05)">
          <div style="font-size:2.5rem;margin-bottom:.5rem">✅</div>
          <div style="font-size:1.05rem;font-weight:700;color:#10b981;margin-bottom:.3rem">
            No Contradictions Detected</div>
          <div style="font-size:.88rem;color:#94a3b8;max-width:480px;margin:0 auto;line-height:1.6">
            Every semantically similar passage pair was examined.
            All were found to be consistent, complementary, or contextually distinct.
          </div>
        </div>
        """, unsafe_allow_html=True)
        _render_export(report, [])
        return

    # ── Tabs ──
    tab_all, tab_by_pair, tab_table = st.tabs(
        [f"⚡ All ({len(contradictions)})", "📂 By Document Pair", "📊 Summary Table"]
    )

    # ── TAB 1: All contradictions ──
    with tab_all:
        # Filter controls
        fc1, fc2, fc3, fc4, fc_search = st.columns([1.2, 1.2, 1.2, 1.4, 2])
        with fc1:
            sev_filter = st.multiselect(
                "Severity", ["CRITICAL", "MAJOR", "MINOR"],
                default=["CRITICAL", "MAJOR", "MINOR"], label_visibility="collapsed",
                placeholder="Severity…",
            )
        with fc2:
            conf_filter = st.multiselect(
                "Confidence", ["HIGH", "MEDIUM", "LOW"],
                default=["HIGH", "MEDIUM", "LOW"], label_visibility="collapsed",
                placeholder="Confidence…",
            )
        with fc3:
            all_types = sorted({_ctype(c) for c in contradictions})
            type_filter = st.multiselect(
                "Type", all_types, default=all_types,
                label_visibility="collapsed", placeholder="Type…",
            )
        with fc4:
            all_docs = sorted({
                doc for c in contradictions
                for doc in (c.chunk_a.document_name, c.chunk_b.document_name)
            })
            doc_filter = st.multiselect(
                "Documents", all_docs, default=all_docs,
                label_visibility="collapsed", placeholder="Documents…",
            )
        with fc_search:
            search_q = st.text_input(
                "Search", placeholder="🔍  Search topic, claim, explanation…",
                label_visibility="collapsed",
            )

        # Apply filters
        filtered = [
            c for c in contradictions
            if _sev(c) in sev_filter
            and _conf(c) in conf_filter
            and _ctype(c) in type_filter
            and (c.chunk_a.document_name in doc_filter or c.chunk_b.document_name in doc_filter)
            and _matches_search(c, search_q)
        ]

        # Sort: critical first, then high-confidence
        filtered.sort(key=lambda c: (_SEV_ORDER.get(_sev(c), 9), _CONF_ORDER.get(_conf(c), 9)))

        st.markdown(
            f'<p style="font-size:.82rem;color:#4a5568;margin:.6rem 0 .8rem">'
            f'Showing <b style="color:#94a3b8">{len(filtered)}</b> of '
            f'<b style="color:#94a3b8">{len(contradictions)}</b></p>',
            unsafe_allow_html=True,
        )

        if not filtered:
            st.markdown("""
            <div class="empty-state">
              <div class="empty-icon">🔍</div>
              <div class="empty-title">No matches</div>
              <div class="empty-sub">Try adjusting your filters or clearing the search.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for i, ev in enumerate(filtered, 1):
                _contradiction_card(ev, i)

    # ── TAB 2: By document pair ──
    with tab_by_pair:
        # Group by frozenset of the two document names
        groups: dict[tuple, list] = defaultdict(list)
        for c in contradictions:
            key = tuple(sorted([c.chunk_a.document_name, c.chunk_b.document_name]))
            groups[key].append(c)

        # Sort groups by max severity
        def _group_priority(g):
            return min(_SEV_ORDER.get(_sev(c), 9) for c in g)

        sorted_groups = sorted(groups.items(), key=lambda x: _group_priority(x[1]))

        for (doc_a, doc_b), items in sorted_groups:
            n_c = sum(1 for c in items if _sev(c) == "CRITICAL")
            n_mj = sum(1 for c in items if _sev(c) == "MAJOR")
            n_mn = sum(1 for c in items if _sev(c) == "MINOR")

            sev_chips = ""
            if n_c:  sev_chips += f'<span class="badge sev-critical">{n_c} critical</span> '
            if n_mj: sev_chips += f'<span class="badge sev-major">{n_mj} major</span> '
            if n_mn: sev_chips += f'<span class="badge sev-minor">{n_mn} minor</span>'

            st.markdown(f"""
            <div class="pair-header">
              <span style="font-size:1rem">📄</span>
              <span class="pair-doc">{doc_a}</span>
              <span class="pair-sep">↔</span>
              <span class="pair-doc">{doc_b}</span>
              <span style="display:flex;gap:.3rem;margin-left:auto;flex-wrap:wrap">
                {sev_chips}
              </span>
            </div>
            """, unsafe_allow_html=True)

            items_sorted = sorted(items, key=lambda c: _SEV_ORDER.get(_sev(c), 9))
            for j, ev in enumerate(items_sorted, 1):
                _contradiction_card(ev, j)

    # ── TAB 3: Table ──
    with tab_table:
        rows = []
        for i, c in enumerate(contradictions, 1):
            rows.append({
                "#":          i,
                "Topic":      c.topic,
                "Type":       _ctype(c),
                "Severity":   _sev(c),
                "Confidence": _conf(c),
                "Doc A":      c.chunk_a.document_name,
                "Doc B":      c.chunk_b.document_name,
                "Similarity": f"{c.similarity_score:.0%}" if c.similarity_score else "—",
                "Detected by": c.detected_by,
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

    _render_export(report, contradictions)


# ── Export section ────────────────────────────────────────────────────────────

def _render_export(report: AnalysisReport, contradictions: list):
    st.markdown("---")
    st.markdown('<div class="label">📥 Export</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 3])
    with c1:
        st.download_button(
            "Download JSON",
            data=_report_to_json(report),
            file_name="contradiction_report.json",
            mime="application/json",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "Download CSV",
            data=_report_to_csv(report),
            file_name="contradiction_report.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ── App header ────────────────────────────────────────────────────────────────

def _render_header():
    st.markdown("""
    <div style="padding:.5rem 0 1.4rem;border-bottom:1px solid rgba(255,255,255,.07);
      margin-bottom:1.6rem;display:flex;align-items:center;gap:1rem">
      <div style="width:42px;height:42px;border-radius:12px;flex-shrink:0;
        background:linear-gradient(135deg,#6366f1,#8b5cf6);
        display:flex;align-items:center;justify-content:center;font-size:1.3rem">⚡</div>
      <div>
        <div style="font-size:1.45rem;font-weight:800;color:#f1f5f9;letter-spacing:-.025em">
          Contradiction Detector</div>
        <div style="font-size:.82rem;color:#4a5568">
          Find where your documents disagree — powered by RAG + LLM reasoning</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Session-state defaults
    for key, default in [("report", None), ("scanning", False), ("scan_error", None)]:
        if key not in st.session_state:
            st.session_state[key] = default

    # Sidebar
    cfg = _render_sidebar()

    # Header
    _render_header()

    # Trigger scan
    if cfg["scan_clicked"] and cfg["can_scan"]:
        st.session_state.scanning = True
        st.session_state.report   = None
        st.session_state.scan_error = None
        _run_scan(cfg)

    # Render main area
    report = st.session_state.report

    if report is None:
        if st.session_state.scanning:
            # Shouldn't normally be visible, but guard against it
            st.info("Scan in progress…")
        else:
            _render_welcome()
        return

    _render_results(report)

    # Offer new scan
    st.markdown("---")
    if st.button("🔄 Start a new scan", help="Clear results and scan again"):
        st.session_state.report = None
        st.session_state.scan_error = None
        st.rerun()


if __name__ == "__main__":
    main()
