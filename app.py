import gradio as gr
from io_utils import read_geotiff, to_rgb_preview, check_pair_compatible
from controller import route_query

def run_satquery(mode, img1_file, img2_file, query, progress=gr.Progress(track_tqdm=False)):
    base_trace = {"mode": mode, "query": query, "tools_used": []}

    # 0) Validate inputs
    if img1_file is None:
        return ("Error: Please upload Image 1.", None, None, None, {"error": "missing_img1", **base_trace})

    if mode != "Single" and img2_file is None:
        arr1, meta1 = read_geotiff(img1_file.name)
        preview1 = to_rgb_preview(arr1)
        return (
            "Error: Please upload Image 2 for this mode.",
            preview1,
            preview1,
            None,
            {"error": "missing_img2", **base_trace},
        )

    # 1) Read Image 1
    progress(0.15, desc="Ingesting Primary Raster")
    arr1, meta1 = read_geotiff(img1_file.name)
    preview1 = to_rgb_preview(arr1)

    arr2 = meta2 = preview2 = None

    # 2) Read Image 2
    if img2_file is not None and mode != "Single":
        progress(0.35, desc="Ingesting Secondary Raster")
        arr2, meta2 = read_geotiff(img2_file.name)
        preview2 = to_rgb_preview(arr2)

        if mode in ["Change Pair", "Optical+SAR Pair"]:
            compat = check_pair_compatible(arr1, meta1, arr2, meta2)
            if not (compat["ok_shape"] and compat["ok_crs"]):
                msg = "System Error: Co-registration failed. Images must match in dimensions and CRS for paired analysis."
                return (msg, preview1, preview1, preview2, {"error": "incompatible_pair", **compat, **base_trace})

    # 3) Route to agent
    progress(0.70, desc="Executing Analysis Pipeline")
    answer, evidence, exec_summary = route_query(mode, img1_file.name, arr1, meta1, arr2, meta2, query)
    
    # 4) Resolve rendering
    progress(0.92, desc="Rendering Visual Evidence")
    if evidence is None:
        evidence = preview1

    return answer, evidence, preview1, preview2, exec_summary


def update_ui_for_mode(mode):
    if mode == "Single":
        return (
            gr.update(visible=False, value=None),
            gr.update(value="Identify major infrastructure and describe the prevailing land-cover.", placeholder="Query primary image..."),
            gr.update(visible=False),
            gr.update(value="**Single-Image VQA**: Upload one GeoTIFF and input a natural language query for scene understanding.")
        )
    if mode == "Change Pair":
        return (
            gr.update(visible=True),
            gr.update(value="Identify major structural changes, deforested regions, or new infrastructure between these timestamps.", placeholder="Query temporal change..."),
            gr.update(visible=True),
            gr.update(value="**Bi-Temporal Change**: Upload two co-registered GeoTIFFs (matching CRS and pixel footprint) for change tracking.")
        )
    return (
        gr.update(visible=True),
        gr.update(value="Isolate hydro-features and structured build-ups using fused reflectance and backscatter.", placeholder="Query optical-SAR fusion..."),
        gr.update(visible=True),
        gr.update(value="**Multi-Modal Fusion**: Upload Optical and SAR passes of the same footprint to overcome atmospheric occlusion.")
    )


# ----------------------------
# ENTERPRISE THEME + CSS
# ----------------------------
theme = gr.themes.Base(
    primary_hue="teal",
    neutral_hue="slate",
    radius_size="md",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --bg-main: #0a0e17;
  --bg-panel: #111827;
  --border: #1f2937;
  --text-main: #f3f4f6;
  --text-muted: #9ca3af;
  --accent: #0ea5e9;
  --accent-hover: #0284c7;
}

body, .gradio-container {
  background-color: var(--bg-main) !important;
  color: var(--text-main) !important;
  font-family: 'Inter', sans-serif !important;
}

#app-container {
  max-width: 1400px;
  margin: 0 auto;
  padding: 30px 20px;
}

.panel-card {
  background: var(--bg-panel) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
  padding: 20px !important;
}

.app-header {
  border-bottom: 1px solid var(--border);
  padding-bottom: 20px;
  margin-bottom: 24px;
}

.app-title {
  font-size: 28px;
  font-weight: 700;
  color: #fff;
  letter-spacing: -0.5px;
}

.app-subtitle {
  font-size: 14px;
  color: var(--text-muted);
  margin-top: 6px;
}

textarea, input[type="text"], input[type="file"] {
  background-color: #0f141f !important;
  border: 1px solid var(--border) !important;
  color: var(--text-main) !important;
}

textarea:focus, input[type="text"]:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 1px var(--accent) !important;
}

button.primary {
  background-color: var(--accent) !important;
  color: #fff !important;
  border: none !important;
  font-weight: 600 !important;
  transition: all 0.2s;
}

button.primary:hover {
  background-color: var(--accent-hover) !important;
}

.output-image {
  border: 1px solid var(--border);
  border-radius: 8px;
  background-color: #000;
}
"""

with gr.Blocks(theme=theme, css=css, title="SatQuery Enterprise", elem_id="app-container") as demo:
    gr.HTML(
        """
        <div class="app-header">
            <div class="app-title">SatQuery Geospatial Intelligence</div>
            <div class="app-subtitle">Enterprise multispectral, SAR, and temporal image analysis engine with auditable telemetry.</div>
        </div>
        """
    )

    with gr.Row(equal_height=False):
        # Configuration Sidebar
        with gr.Column(scale=3, min_width=320, elem_classes=["panel-card"]):
            gr.Markdown("### Analysis Configuration")
            mode_dropdown = gr.Radio(
                choices=["Single", "Change Pair", "Optical+SAR Pair"],
                value="Single",
                label="Operation Mode"
            )

            mode_help = gr.Markdown(
                "**Single-Image VQA**: Upload one GeoTIFF and input a natural language query for scene understanding."
            )

            gr.Markdown("---")
            
            img1_upload = gr.File(label="Primary Acquisition (GeoTIFF)", file_count="single", file_types=[".tif", ".tiff"])
            img2_upload = gr.File(label="Secondary Acquisition (GeoTIFF)", file_count="single", file_types=[".tif", ".tiff"], visible=False)

            query_input = gr.Textbox(
                label="Analytical Prompt",
                value="Identify major infrastructure and describe the prevailing land-cover.",
                lines=4
            )

            with gr.Row():
                submit_btn = gr.Button("Execute Pipeline", variant="primary")
                reset_btn = gr.Button("Reset State", variant="secondary")

        # Output Main Panel
        with gr.Column(scale=7, min_width=600):
            with gr.Group(elem_classes=["panel-card"]):
                gr.Markdown("### Intelligence Briefing")
                ai_answer = gr.Textbox(show_label=False, lines=4, interactive=False, placeholder="System awaiting payload...")

            with gr.Row(elem_classes=["panel-card"]):
                with gr.Column():
                    gr.Markdown("#### Spatial Evidence (Overlay)")
                    evidence_img = gr.Image(show_label=False, interactive=False, type="numpy", height=380, elem_classes=["output-image"])
                
                with gr.Column():
                    with gr.Tabs():
                        with gr.TabItem("Primary Source"):
                            preview1_img = gr.Image(show_label=False, interactive=False, type="numpy", height=340, elem_classes=["output-image"])
                        with gr.TabItem("Secondary Source", visible=False) as img2_container:
                            preview2_img = gr.Image(show_label=False, interactive=False, type="numpy", height=340, elem_classes=["output-image"])

            with gr.Accordion("System Telemetry & Auditable Trace", open=False, elem_classes=["panel-card"]):
                exec_summary = gr.JSON(show_label=False)

    # --- Event Binding
    mode_dropdown.change(
        fn=update_ui_for_mode,
        inputs=[mode_dropdown],
        outputs=[img2_upload, query_input, img2_container, mode_help],
    )

    submit_btn.click(
        fn=run_satquery,
        inputs=[mode_dropdown, img1_upload, img2_upload, query_input],
        outputs=[ai_answer, evidence_img, preview1_img, preview2_img, exec_summary],
    )

    def _reset(mode):
        ui = update_ui_for_mode(mode)
        return (ui[0], ui[1], ui[2], ui[3], None, None, None, None, None, {})

    reset_btn.click(
        fn=_reset,
        inputs=[mode_dropdown],
        outputs=[img2_upload, query_input, img2_container, mode_help, img1_upload, ai_answer, evidence_img, preview1_img, preview2_img, exec_summary],
    )

if __name__ == "__main__":
    demo.launch(share=True)
