import gradio as gr
import numpy as np
from io_utils import read_geotiff, to_rgb_preview, check_pair_compatible
from controller import route_query

# Create a visible space-gray placeholder to prevent broken image icons on load[cite: 1]
dummy_placeholder = np.zeros((450, 450, 3), dtype=np.uint8)
# Lighter dark-blue to make the "globe" visible before image upload
dummy_placeholder[:] = (20, 25, 35) 

def run_satquery(mode, img1_file, img2_file, query, progress=gr.Progress(track_tqdm=False)):
    base_trace = {"mode": mode, "query": query, "tools_used": []}

    if img1_file is None:
        return ("System Error: Primary acquisition missing.", dummy_placeholder, dummy_placeholder, None, {"error": "missing_img1", **base_trace})

    if mode != "Single" and img2_file is None:
        arr1, meta1 = read_geotiff(img1_file.name)
        preview1 = to_rgb_preview(arr1)
        return (
            "System Error: Secondary acquisition required for this mode.",
            preview1,
            preview1,
            None,
            {"error": "missing_img2", **base_trace},
        )

    progress(0.15, desc="Ingesting Primary Raster")
    arr1, meta1 = read_geotiff(img1_file.name)
    preview1 = to_rgb_preview(arr1)
    arr2 = meta2 = preview2 = None

    if img2_file is not None and mode != "Single":
        progress(0.35, desc="Ingesting Secondary Raster")
        arr2, meta2 = read_geotiff(img2_file.name)
        preview2 = to_rgb_preview(arr2)

        if mode in ["Change Pair", "Optical+SAR Pair"]:
            compat = check_pair_compatible(arr1, meta1, arr2, meta2)
            if not (compat["ok_shape"] and compat["ok_crs"]):
                msg = "System Error: Co-registration failed. Images must match in dimensions and CRS."
                return (msg, preview1, preview1, preview2, {"error": "incompatible_pair", **compat, **base_trace})

    progress(0.70, desc="Executing Analysis Pipeline")
    answer, evidence, exec_summary = route_query(mode, img1_file.name, arr1, meta1, arr2, meta2, query)
    
    progress(0.92, desc="Rendering Visual Evidence")
    if evidence is None:
        evidence = preview1

    return answer, evidence, preview1, preview2, exec_summary

def update_ui_for_mode(mode):
    if mode == "Single":
        return (
            gr.update(visible=False, value=None),
            gr.update(value="Identify major infrastructure and describe the prevailing land-cover.", placeholder="Type your message for SatQuery AI..."),
            gr.update(visible=False),
        )
    if mode == "Change Pair":
        return (
            gr.update(visible=True),
            gr.update(value="Identify major structural changes, deforested regions, or new infrastructure between these timestamps.", placeholder="Type your message for SatQuery AI..."),
            gr.update(visible=True),
        )
    return (
        gr.update(visible=True),
        gr.update(value="Isolate hydro-features and structured build-ups using fused reflectance and backscatter.", placeholder="Type your message for SatQuery AI..."),
        gr.update(visible=True),
    )

# ----------------------------
# FUTURISTIC UI DASHBOARD CSS
# ----------------------------
theme = gr.themes.Base(
    primary_hue="amber",
    neutral_hue="zinc",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

:root {
  --bg-dark: #09090b;
  --panel-bg: #121216;
  --panel-border: rgba(212, 175, 55, 0.25);
  --gold-accent: #d4af37;
  --gold-glow: 0px 4px 24px rgba(212, 175, 55, 0.15);
  --text-main: #f4f4f5;
  --text-muted: #a1a1aa;
}

body, .gradio-container {
  background-color: var(--bg-dark) !important;
  color: var(--text-main) !important;
  font-family: 'Inter', sans-serif !important;
}

#dashboard-container {
  max-width: 1600px;
  margin: 0 auto;
  padding: 2vh 2vw;
}

/* Glassmorphic Panels */
.glass-panel {
  background: var(--panel-bg) !important;
  border: 1px solid var(--panel-border) !important;
  border-radius: 16px !important;
  box-shadow: var(--gold-glow) !important;
  padding: 20px !important;
}

/* Typography & Headers */
h1, h2, h3, h4, .markdown-text p {
  color: var(--text-main) !important;
  margin-bottom: 8px;
}
.sidebar-title {
  font-size: 20px;
  font-weight: 600;
  color: var(--gold-accent);
  letter-spacing: 1px;
  text-transform: uppercase;
  margin-bottom: 24px;
}

/* Inputs & Textareas */
textarea, input[type="text"], input[type="file"] {
  background: rgba(255, 255, 255, 0.03) !important;
  border: 1px solid var(--panel-border) !important;
  color: var(--text-main) !important;
  border-radius: 12px !important;
}
textarea:focus, input[type="text"]:focus {
  border-color: var(--gold-accent) !important;
  box-shadow: 0 0 0 1px var(--gold-accent) !important;
}

/* Buttons */
button.primary {
  background: transparent !important;
  color: var(--gold-accent) !important;
  border: 1px solid var(--gold-accent) !important;
  border-radius: 12px !important;
  font-weight: 500 !important;
  transition: all 0.3s ease;
}
button.primary:hover {
  background: var(--gold-accent) !important;
  color: var(--bg-dark) !important;
  box-shadow: var(--gold-glow) !important;
}

button.secondary {
  background: rgba(255,255,255,0.05) !important;
  color: var(--text-muted) !important;
  border: 1px solid transparent !important;
  border-radius: 12px !important;
}
button.secondary:hover {
  background: rgba(255,255,255,0.1) !important;
  color: var(--text-main) !important;
}

/* Center Stage Stabilization */
.center-column {
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  justify-content: flex-start !important;
  min-height: 580px !important;
  position: relative;
}

/* Chat Prompt Box Area */
.chat-container {
  background: linear-gradient(180deg, rgba(30,30,35,0.8) 0%, rgba(15,15,18,0.95) 100%) !important;
  border: 1px solid var(--panel-border) !important;
  border-radius: 16px !important;
  padding: 16px !important;
  width: 90% !important;
  max-width: 600px !important;
  transform: translateY(-60px) !important; /* Safely overlaps the globe */
  position: relative !important;
  z-index: 10 !important;
  backdrop-filter: blur(12px) !important;
  box-shadow: 0px 10px 30px rgba(0,0,0,0.6) !important;
}

/* Image masking for a globe effect */
.output-img {
  border-radius: 50% !important; 
  border: 1px solid var(--panel-border) !important;
  box-shadow: var(--gold-glow) !important;
  background: radial-gradient(circle at 30% 30%, #2a2a35 0%, #050505 80%) !important;
  width: 450px !important;
  height: 450px !important;
  max-width: 450px !important;
  max-height: 450px !important;
  margin: 0 auto !important;
  overflow: hidden !important;
}

/* Target internal Gradio elements to enforce the circular mask */
.output-img > div, .output-img img {
  border-radius: 50% !important;
  object-fit: cover !important;
  width: 100% !important;
  height: 100% !important;
  aspect-ratio: 1 / 1 !important;
}

/* Hide download buttons on the globe to keep it clean */
.output-img button {
  display: none !important;
}

.side-img {
  border-radius: 12px !important;
  border: 1px solid var(--panel-border) !important;
  background-color: #050505 !important;
}
.side-img img {
  object-fit: cover !important;
}

/* Radio Buttons (Left Nav) */
.gr-radio {
  background: transparent !important;
  border: none !important;
}
"""

with gr.Blocks(theme=theme, css=css, title="SatQuery Dashboard", elem_id="dashboard-container") as demo:
    
    with gr.Row(equal_height=False):
        
        # ==========================================
        # LEFT SIDEBAR: Navigation & Configuration
        # ==========================================
        with gr.Column(scale=2, min_width=250):
            gr.HTML("<div class='sidebar-title'>UI Dashboard</div>")
            
            with gr.Group(elem_classes=["glass-panel"]):
                mode_dropdown = gr.Radio(
                    choices=["Single", "Change Pair", "Optical+SAR Pair"],
                    value="Single",
                    label="Operation Modes",
                    elem_classes=["gr-radio"]
                )
            
            gr.HTML("<br>")
            
            with gr.Group(elem_classes=["glass-panel"]):
                gr.Markdown("#### Data Payload")
                img1_upload = gr.File(label="Primary Acquisition", file_count="single", file_types=[".tif", ".tiff"])
                img2_upload = gr.File(label="Secondary Acquisition", file_count="single", file_types=[".tif", ".tiff"], visible=False)
                reset_btn = gr.Button("Reset State", variant="secondary")

        # ==========================================
        # CENTER STAGE: Visual Output & Chat Prompt
        # ==========================================
        with gr.Column(scale=5, min_width=500, elem_classes=["center-column"]):
            
            # Floating Globe initialized with a dummy array
            evidence_img = gr.Image(
                value=dummy_placeholder,
                show_label=False, 
                interactive=False, 
                type="numpy",
                elem_classes=["output-img"]
            )
            
            # Glassmorphic Chat Input Area
            with gr.Group(elem_classes=["chat-container"]):
                query_input = gr.Textbox(
                    show_label=False,
                    value="Identify major infrastructure and describe the prevailing land-cover.",
                    placeholder="Type your message for SatQuery AI...",
                    lines=2
                )
                submit_btn = gr.Button("Send Request ↗", variant="primary")

        # ==========================================
        # RIGHT SIDEBAR: Recent Views & Insights
        # ==========================================
        with gr.Column(scale=3, min_width=300):
            with gr.Group(elem_classes=["glass-panel"]):
                gr.Markdown("### Recent Earth Views")
                # Initialize previews with dummy arrays to prevent broken icons
                preview1_img = gr.Image(value=dummy_placeholder, label="Source 1", interactive=False, type="numpy", height=150, elem_classes=["side-img"])
                preview2_img = gr.Image(value=dummy_placeholder, label="Source 2", interactive=False, type="numpy", height=150, elem_classes=["side-img"], visible=False)
                
            gr.HTML("<br>")
            
            with gr.Group(elem_classes=["glass-panel"]):
                gr.Markdown("### AI Insights")
                ai_answer = gr.Textbox(show_label=False, lines=4, interactive=False, placeholder="Insights will appear here...")
                
            gr.HTML("<br>")
                
            with gr.Accordion("Telemetry Network", open=False, elem_classes=["glass-panel"]):
                exec_summary = gr.JSON(show_label=False)

    # --- Event Binding
    mode_dropdown.change(
        fn=update_ui_for_mode,
        inputs=[mode_dropdown],
        outputs=[img2_upload, query_input, preview2_img],
    )

    submit_btn.click(
        fn=run_satquery,
        inputs=[mode_dropdown, img1_upload, img2_upload, query_input],
        outputs=[ai_answer, evidence_img, preview1_img, preview2_img, exec_summary],
    )

    def _reset(mode):
        ui = update_ui_for_mode(mode)
        return (ui[0], ui[1], ui[2], None, None, dummy_placeholder, dummy_placeholder, None, {})

    reset_btn.click(
        fn=_reset,
        inputs=[mode_dropdown],
        outputs=[img2_upload, query_input, preview2_img, img1_upload, ai_answer, evidence_img, preview1_img, exec_summary],
    )

if __name__ == "__main__":
    demo.launch(share=True)
