import gradio as gr

THEME = gr.themes.Soft(
    primary_hue="amber",
    neutral_hue="slate",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#070A12",
    body_background_fill_dark="#070A12",
    button_primary_background_fill="#d4af37",
    button_primary_background_fill_dark="#d4af37",
    button_primary_text_color="#070A12",
    button_primary_text_color_dark="#070A12",
)

CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

:root{
  --bg0:#070A12;
  --bg1:#0B1020;
  --card: rgba(17, 24, 39, .58);
  --card2: rgba(17, 24, 39, .42);
  --stroke: rgba(148, 163, 184, .18);
  --stroke2: rgba(148, 163, 184, .12);
  --text: rgba(226, 232, 240, .94);
  --muted: rgba(226, 232, 240, .70);
  --muted2: rgba(226, 232, 240, .56);
  --accent: #d4af37;
  --accent2: #f4d87a;
  --good: #34d399;
  --warn: #fb923c;
  --bad:  #f87171;
  --shadow: 0 18px 60px rgba(0,0,0,.45);
}

.gradio-container{
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif !important;
  background:
    radial-gradient(900px circle at 18% -8%, rgba(212,175,55,.18) 0%, rgba(7,10,18,0) 52%),
    radial-gradient(1200px circle at 90% 0%, rgba(96,165,250,.14) 0%, rgba(7,10,18,0) 55%),
    radial-gradient(1200px circle at 70% 120%, rgba(16,185,129,.12) 0%, rgba(7,10,18,0) 58%),
    linear-gradient(180deg, var(--bg0) 0%, #070A12 45%, #050710 100%) !important;
  color: var(--text) !important;
}

#app{
  max-width: 1240px;
  margin: 0 auto;
  padding: 18px 16px 28px;
}

.glass{
  background: var(--card) !important;
  border: 1px solid var(--stroke) !important;
  border-radius: 18px !important;
  box-shadow: var(--shadow);
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
}

.glass.soft{
  background: var(--card2) !important;
  border: 1px solid var(--stroke2) !important;
  box-shadow: 0 12px 40px rgba(0,0,0,.35);
}

.hero{
  position: relative;
  overflow: hidden;
  padding: 18px 18px 16px;
}

.heroGrid{
  display:flex;
  align-items:flex-end;
  justify-content:space-between;
  gap:18px;
}

.brandTitle{
  font-size: 22px;
  font-weight: 900;
  letter-spacing: .2px;
  line-height: 1.1;
}

.brandSub{
  margin-top: 8px;
  font-size: 13px;
  color: var(--muted);
  max-width: 72ch;
}

.pillRow{ display:flex; gap:10px; flex-wrap:wrap; justify-content:flex-end; }
.pill{
  display:inline-flex;
  align-items:center;
  gap:8px;
  padding: 8px 12px;
  border-radius: 999px;
  border: 1px solid rgba(148,163,184,.22);
  background: rgba(2,6,23,.34);
  color: var(--muted);
  font-size: 12px;
  user-select:none;
}
.dot{ width:8px; height:8px; border-radius:999px; background: var(--accent); box-shadow:0 0 0 3px rgba(212,175,55,.12); }

.kbd{
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  border: 1px solid rgba(148,163,184,.22);
  background: rgba(2,6,23,.35);
  border-bottom-color: rgba(148,163,184,.35);
  padding: 2px 8px;
  border-radius: 10px;
  color: var(--muted);
  font-size: 12px;
}

.sectionTitle{
  font-size: 13px;
  letter-spacing: .18em;
  text-transform: uppercase;
  color: rgba(226,232,240,.62);
  margin-bottom: 10px;
}

textarea, input[type="text"], input[type="number"]{
  background: rgba(2,6,23,.35) !important;
  border: 1px solid rgba(148,163,184,.24) !important;
  color: var(--text) !important;
  border-radius: 14px !important;
}
textarea:focus, input[type="text"]:focus, input[type="number"]:focus{
  border-color: rgba(212,175,55,.65) !important;
  box-shadow: 0 0 0 3px rgba(212,175,55,.14) !important;
}

button.primary{
  background: linear-gradient(135deg, var(--accent2), var(--accent)) !important;
  color: #070A12 !important;
  font-weight: 900 !important;
  border: none !important;
  letter-spacing: .3px;
}
button.primary:hover{ filter: brightness(1.04); transform: translateY(-1px); }

button.secondary{
  border: 1px solid rgba(148,163,184,.24) !important;
  background: rgba(2,6,23,.25) !important;
}

hr{
  border: none;
  height: 1px;
  background: rgba(148,163,184,.14);
  margin: 14px 0;
}

.smallHint{
  font-size: 12px;
  color: var(--muted2);
  line-height: 1.35;
}

.cardPad{ padding: 14px; }
.tightMd p{ margin: 0.55em 0; }
.tightMd ul{ margin: 0.4em 0 0.6em 1.1em; }

.imageFrame img{
  border-radius: 16px !important;
  border: 1px solid rgba(148,163,184,.18);
}

.badgeGood{ color: rgba(52,211,153,.92); }
.badgeWarn{ color: rgba(251,146,60,.92); }
.badgeInfo{ color: rgba(96,165,250,.92); }

@media (max-width: 980px){
  #app{ padding: 14px 10px 18px; }
  .heroGrid{ flex-direction: column; align-items:flex-start; }
  .pillRow{ justify-content:flex-start; }
}
"""

HERO_HTML = r"""
<div class="glass hero">
  <div class="heroGrid">
    <div>
      <div class="brandTitle">SatQuery AI</div>
      <div class="brandSub">
        A human-friendly remote-sensing assistant. Ask in plain language — the agent plans tools, validates inputs, and returns evidence-backed answers with an auditable trace.
      </div>
      <div style="margin-top:10px; display:flex; gap:10px; flex-wrap:wrap;">
        <span class="kbd">Place → Fetch → Analyze</span>
        <span class="kbd">Upload → Validate → Answer</span>
        <span class="kbd">Evidence + Trace</span>
      </div>
    </div>

    <div class="pillRow">
      <div class="pill"><span class="dot"></span> Designed for non-experts</div>
      <div class="pill"><span class="dot"></span> Optical • SAR • Change</div>
      <div class="pill"><span class="dot"></span> Report-ready outputs</div>
    </div>
  </div>
</div>
"""
