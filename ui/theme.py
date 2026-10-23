"""Editorial "resort de luxo" theme for the Streamlit UI.

Pure presentation: Streamlit cannot restyle itself fully, so we inject a
small CSS layer and Google Fonts via ``unsafe_allow_html``. No business logic
here — everything testable lives in ``ui/logic.py``.
"""

import streamlit as st

from ui.logic import confidence_label, confidence_percent

FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700"
    "&family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500"
    "&display=swap');"
)

CSS = """
<style>
{font_import}

:root {{
  --bg: #FAF6EF;
  --surface: #FFFDF8;
  --ink: #1E1A14;
  --muted: #6B6257;
  --accent: #B4552D;
  --trust: #2E5B4B;
  --abstain: #C9A24B;
  --line: #E7DCC8;
}}

/* Fonts */
html, body, [data-testid="stAppViewContainer"] {{
  font-family: "Instrument Sans", system-ui, sans-serif;
  color: var(--ink);
  background: var(--bg);
}}
h1, h2, h3, [data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2 {{
  font-family: "Sora", system-ui, sans-serif;
}}

/* Editorial masthead */
.rag-eyebrow {{
  font-family: "Sora", sans-serif;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.28em;
  text-transform: uppercase;
  color: var(--accent);
  margin: 0.1rem 0 0.25rem;
}}
.rag-title {{
  font-family: "Sora", sans-serif;
  font-weight: 600;
  font-size: 1.9rem;
  letter-spacing: -0.01em;
  margin: 0;
}}
.rag-subtitle {{
  color: var(--muted);
  font-size: 0.95rem;
  margin-top: 0.4rem;
}}
.rag-rule {{
  border: 0;
  border-top: 2px solid var(--ink);
  margin: 1rem 0 0.25rem;
}}
.rag-rule.thin {{
  border-top: 1px solid var(--line);
  margin-top: 0.2rem;
}}

/* SOP citation cards */
.sop-card {{
  background: var(--surface);
  border: 1px solid var(--line);
  border-left: 3px solid var(--accent);
  border-radius: 8px;
  padding: 0.6rem 0.8rem;
  margin: 0.4rem 0;
}}
.sop-head {{
  display: flex;
  align-items: baseline;
  gap: 0.5rem;
}}
.sop-badge {{
  font-family: "Sora", sans-serif;
  font-weight: 700;
  color: #fff;
  background: var(--accent);
  border-radius: 999px;
  padding: 0.1rem 0.5rem;
  font-size: 0.72rem;
}}
.sop-title {{
  font-family: "Sora", sans-serif;
  font-weight: 600;
  font-size: 0.95rem;
}}
.sop-meta {{
  font-size: 0.78rem;
  color: var(--muted);
  margin-top: 0.15rem;
}}
.sop-excerpt {{
  font-family: "JetBrains Mono", ui-monospace, monospace;
  font-size: 0.78rem;
  line-height: 1.45;
  background: #F4EEE2;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 0.5rem;
  margin: 0.5rem 0 0;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--ink);
}}

/* Confidence meter */
.confidence {{
  display: flex;
  align-items: center;
  gap: 0.6rem;
  margin: 0.5rem 0 0.25rem;
  font-size: 0.8rem;
}}
.confidence .track {{
  flex: 1;
  height: 8px;
  border-radius: 999px;
  background: #EDE3D3;
  overflow: hidden;
}}
.confidence .fill {{ display: block; height: 100%; width: 0; border-radius: 999px; }}
.confidence .label {{
  font-family: "Sora", sans-serif;
  font-weight: 600;
  white-space: nowrap;
}}

/* Abstention state */
.abstain-card {{
  background: #FBF4E3;
  border: 1px dashed var(--abstain);
  border-radius: 8px;
  padding: 0.7rem 0.9rem;
  margin: 0.5rem 0;
  color: #6b5a23;
}}

/* Example chips */
.chips {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.5rem 0 1rem; }}
.stButton > button {{
  font-family: "Instrument Sans", sans-serif;
  font-size: 0.86rem;
  font-weight: 500;
  color: var(--ink);
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 0.4rem 0.9rem;
  height: auto;
  transition: border-color 0.15s, color 0.15s;
}}
.stButton > button:hover {{
  border-color: var(--accent);
  color: var(--accent);
  background: #fff;
}}
.stButton > button:focus {{ border-color: var(--accent); color: var(--accent); box-shadow: none; }}

/* De-emphasise Streamlit chrome */
header[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{
  background: var(--surface);
  border-right: 1px solid var(--line);
}}
</style>
"""


def inject() -> None:
    """Write the theme CSS + font import into the app."""
    st.markdown(CSS.format(font_import=FONT_IMPORT), unsafe_allow_html=True)


def confidence_meter(confidence: float) -> str:
    """HTML for the confidence bar; color follows the trust bucket."""
    pct = confidence_percent(confidence)
    label = confidence_label(confidence)
    colors = {
        "Alta": "#2E5B4B",
        "Média": "#B4552D",
        "Baixa": "#C9A24B",
        "Sem evidência": "#9C8F7A",
    }
    return (
        f'<div class="confidence">'
        f'<span class="track"><span class="fill" style="width:{pct}%;'
        f'background:{colors[label]}"></span></span>'
        f'<span class="label">{label} · {pct}%</span>'
        f"</div>"
    )
