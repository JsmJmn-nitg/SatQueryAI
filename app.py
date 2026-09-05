import gradio as gr
from io_utils import read_geotiff, to_rgb_preview, check_pair_compatible
from controller import route_query

def run_satquery(mode, img1_file, img2_file, query, progress=gr.Progress(track_tqdm=False)):
    base_trace = {"mode": mode, "query": query, "tools_used": []}

    if img1_file is None:
        return ("System Error: Primary acquisition missing.", None, None, None, {"error": "missing_img1", **base_trace})

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

/* Chat Prompt Box Area */
.chat-container {
  background: linear-gradient(180deg, rgba(30,30,35,0.6) 0%, rgba(15,15,18,0.9) 100%) !important;
  border: 1px solid var(--panel-border) !important;
  border-radius: 16px !important;
  padding: 16px !important;
  margin-top: -40px; /* Overlaps the visual evidence slightly */
  position: relative;
  z-index: 10;
  backdrop-filter: blur(12px);
}

/* Images */
.output-img {
  border-radius: 50% !important; /* Forces a globe-like circular mask for center image if square */
  border: 1px solid var(--panel-border);
  box-shadow: var(--gold-glow);
  background-color: #000;
  overflow: hidden;
}
.side-img {
  border-radius: 12px !important;
  border: 1px solid var(--panel-border);
  background-color: #000;
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
        with gr.Column(scale=5, min_width=500):
            # Floating Globe / Main Evidence
            with gr.Row(elem_classes=["output-img"]):
                evidence_img = gr.Image(show_label=False, interactive=False, type="numpy", height=500, elem_classes=["output-img"])
            
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
                preview1_img = gr.Image(label="Source 1", interactive=False, type="numpy", height=150, elem_classes=["side-img"])
                preview2_img = gr.Image(label="Source 2", interactive=False, type="numpy", height=150, elem_classes=["side-img"], visible=False)
                
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
        return (ui[0], ui[1], ui[2], None, None, None, None, None, {})

    reset_btn.click(
        fn=_reset,
        inputs=[mode_dropdown],
        outputs=[img2_upload, query_input, preview2_img, img1_upload, ai_answer, evidence_img, preview1_img, exec_summary],
    )

if __name__ == "__main__":
    demo.launch(share=True)
