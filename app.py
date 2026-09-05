import gradio as gr
from io_utils import read_geotiff, to_rgb_preview, check_pair_compatible
from controller import route_query


def run_satquery(mode, img1_file, img2_file, query, progress=gr.Progress(track_tqdm=False)):
    # Standardized trace shape (always return something)
    base_trace = {"mode": mode, "query": query, "tools_used": []}

    # 0) Validate inputs (functional-first)
    if img1_file is None:
        return (
            "Error: Please upload Image 1.",
            None,
            None,
            None,
            {"error": "missing_img1", **base_trace},
        )

    if mode != "Single" and img2_file is None:
        # Still show preview1 so the UI doesn't feel broken
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
    progress(0.15, desc="Reading Image 1")
    arr1, meta1 = read_geotiff(img1_file.name)
    preview1 = to_rgb_preview(arr1)

    arr2 = meta2 = preview2 = None

    # 2) Read Image 2 (if applicable)
    if img2_file is not None and mode != "Single":
        progress(0.35, desc="Reading Image 2")
        arr2, meta2 = read_geotiff(img2_file.name)
        preview2 = to_rgb_preview(arr2)

        # Enforce compatibility check for paired modes
        if mode in ["Change Pair", "Optical+SAR Pair"]:
            compat = check_pair_compatible(arr1, meta1, arr2, meta2)
            if not (compat["ok_shape"] and compat["ok_crs"]):
                msg = "Error: Images must match in dimensions and CRS for paired analysis."
                return (
                    msg,
                    preview1,   # show something in Evidence
                    preview1,
                    preview2,
                    {"error": "incompatible_pair", **compat, **base_trace},
                )

    # 3) Route to controller
    progress(0.70, desc="Running agent pipeline")
    answer, evidence, exec_summary = route_query(mode, img1_file.name, arr1, meta1, arr2, meta2, query)
    
    # 4) Fallback evidence
    progress(0.92, desc="Rendering outputs")
    if evidence is None:
        evidence = preview1

    return answer, evidence, preview1, preview2, exec_summary


def update_ui_for_mode(mode):
    """Dynamically updates the UI based on the selected mode."""
    if mode == "Single":
        return (
            gr.update(visible=False, value=None),  # hide + clear Image 2 upload
            gr.update(
                value="Describe the land-cover and major objects.",
                placeholder="Ask about Image 1…",
            ),
            gr.update(visible=False),              # hide Image 2 preview panel
            gr.update(
                value=(
                    "Single-image understanding (captioning / VQA). "
                    "Upload one GeoTIFF and ask a question."
                )
            ),
        )

    if mode == "Change Pair":
        return (
            gr.update(visible=True),
            gr.update(
                value="What changed between these two dates? Identify major differences.",
                placeholder="Ask about change (construction, deforestation, flooding)…",
            ),
            gr.update(visible=True),
            gr.update(
                value=(
                    "Bi-temporal change analysis. "
                    "Upload two co-registered GeoTIFFs (same CRS + pixel dimensions)."
                )
            ),
        )

    # Optical+SAR Pair
    return (
        gr.update(visible=True),
        gr.update(
            value="Use both optical and SAR data to identify built-up areas and water bodies.",
            placeholder="Ask about fusion results (water / built-up / roads / etc.)…",
        ),
        gr.update(visible=True),
        gr.update(
            value=(
                "Optical + SAR fusion. "
                "Upload both views of the same area to improve robustness (clouds/night)."
            )
        ),
    )


# ----------------------------
# THEME + CSS (modern, clean)
# ----------------------------
theme = gr.themes.Soft(
    primary_hue="amber",
    neutral_hue="slate",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

:root{
  --bg0:#0b0f19;
  --bg1:#0f172a;
  --panel:rgba(15, 23, 42, .55);
  --stroke:rgba(148, 163, 184, .18);
  --text:rgba(226, 232, 240, .92);
  --muted:rgba(226, 232, 240, .70);
  --accent:#d4af37;
}

.gradio-container{
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif !important;
  background: radial-gradient(1100px circle at 20% -10%, #1f2a44 0%, var(--bg0) 55%) !important;
  color: var(--text) !important;
}

#app{
  max-width: 1240px;
  margin: 0 auto;
  padding: 20px 16px 28px;
}

.card{
  background: var(--panel) !important;
  border: 1px solid var(--stroke) !important;
  border-radius: 16px !important;
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  box-shadow: 0 10px 30px rgba(0,0,0,.35);
}

.header{
  display:flex;
  align-items:flex-end;
  justify-content:space-between;
  gap:14px;
  margin: 4px 0 14px;
}
.title{
  font-size: 22px;
  font-weight: 800;
  letter-spacing: .2px;
  line-height: 1.2;
}
.subtitle{
  margin-top: 6px;
  font-size: 13px;
  color: var(--muted);
}

.badge{
  display:inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid rgba(148,163,184,.22);
  background: rgba(2,6,23,.35);
  color: var(--muted);
  font-size: 12px;
}

textarea, input[type="text"]{
  background: rgba(2,6,23,.35) !important;
  border: 1px solid rgba(148,163,184,.25) !important;
  color: var(--text) !important;
  border-radius: 12px !important;
}
textarea:focus, input[type="text"]:focus{
  border-color: rgba(212,175,55,.65) !important;
  box-shadow: 0 0 0 3px rgba(212,175,55,.14) !important;
}

button.primary{
  background: linear-gradient(135deg, var(--accent), #9a7b1a) !important;
  color: #0b0f19 !important;
  font-weight: 800 !important;
  border: none !important;
  letter-spacing: .6px;
}
button.primary:hover{
  filter: brightness(1.06);
  transform: translateY(-1px);
}

img{
  border-radius: 14px;
}

@media (max-width: 900px){
  #app{ padding: 14px 10px 18px; }
}
"""


with gr.Blocks(theme=theme, css=css, title="SatQuery AI", elem_id="app") as demo:
    gr.HTML(
        """
        <div class="header">
          <div>
            <div class="title">SatQuery AI</div>
            <div class="subtitle">
              Upload GeoTIFFs, ask a question, get an answer + visual evidence + an auditable trace.
            </div>
          </div>
          <div class="badge">Single · Change Pair · Optical+SAR</div>
        </div>
        """
    )

    with gr.Row(equal_height=True):
        # Sidebar (inputs)
        with gr.Column(scale=4, min_width=320, elem_classes=["card"]):
            gr.Markdown("### Configure")
            mode_dropdown = gr.Radio(
                choices=["Single", "Change Pair", "Optical+SAR Pair"],
                value="Single",
                label="Mode",
            )

            mode_help = gr.Markdown(
                "Single-image understanding (captioning / VQA). Upload one GeoTIFF and ask a question.",
                elem_classes=[],
            )

            img1_upload = gr.File(
                label="Image 1 (GeoTIFF/TIFF)",
                file_count="single",
                file_types=[".tif", ".tiff"],
            )
            img2_upload = gr.File(
                label="Image 2 (required for paired modes)",
                file_count="single",
                file_types=[".tif", ".tiff"],
                visible=False,
            )

            query_input = gr.Textbox(
                label="Query",
                value="Describe the land-cover and major objects.",
                lines=4,
                placeholder="Ask about Image 1…",
            )

            with gr.Row():
                submit_btn = gr.Button("Run", variant="primary")
                reset_btn = gr.Button("Reset", variant="secondary")

        # Main (outputs)
        with gr.Column(scale=8, min_width=520):
            with gr.Group(elem_classes=["card"]):
                gr.Markdown("### Agent response")
                ai_answer = gr.Textbox(
                    show_label=False,
                    lines=5,
                    interactive=False,
                    placeholder="Your answer will appear here…",
                )

            with gr.Row():
                with gr.Column(elem_classes=["card"]):
                    gr.Markdown("**Evidence / overlay**")
                    evidence_img = gr.Image(
                        show_label=False,
                        interactive=False,
                        type="numpy",
                        height=320,
                    )

                with gr.Column(elem_classes=["card"]):
                    gr.Markdown("**Image 1 preview**")
                    preview1_img = gr.Image(
                        show_label=False,
                        interactive=False,
                        type="numpy",
                        height=320,
                    )

                with gr.Column(elem_classes=["card"], visible=False) as img2_container:
                    gr.Markdown("**Image 2 preview**")
                    preview2_img = gr.Image(
                        show_label=False,
                        interactive=False,
                        type="numpy",
                        height=320,
                    )

            with gr.Accordion("Execution summary (JSON trace)", open=False, elem_classes=["card"]):
                exec_summary = gr.JSON(show_label=False)

    # --- Events
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

    # Reset clears inputs + outputs (keeps mode selection)
    def _reset(mode):
        # preserve mode value; clear everything else
        ui = update_ui_for_mode(mode)
        return (
            ui[0],               # img2_upload update
            ui[1],               # query_input update
            ui[2],               # img2_container update
            ui[3],               # mode_help update
            None,                # img1_upload value
            None,                # ai_answer
            None, None, None,    # images
            {},                  # exec_summary
        )

    reset_btn.click(
        fn=_reset,
        inputs=[mode_dropdown],
        outputs=[
            img2_upload, query_input, img2_container, mode_help,
            img1_upload, ai_answer, evidence_img, preview1_img, preview2_img, exec_summary
        ],
    )


if __name__ == "__main__":
    demo.launch(share=True)
