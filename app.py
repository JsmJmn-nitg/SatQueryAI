import gradio as gr
import numpy as np

from ui_styles import THEME, CSS, HERO_HTML
from mock_backend import run_place_workflow, run_upload_workflow

def safe_preview_from_file(file_obj):
    if file_obj is None:
        return None
    path = getattr(file_obj, "name", None)
    if not path:
        return None

    # Try GeoTIFF first
    try:
        from io_utils import read_geotiff, to_rgb_preview
        arr, _ = read_geotiff(path)
        return to_rgb_preview(arr)
    except Exception:
        pass

    # Fallback: standard images
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        return np.array(img)
    except Exception:
        return None

def update_ui_for_upload_mode(mode):
    if mode == "Single":
        return (
            gr.update(visible=False, value=None),
            gr.update(visible=False),
            gr.update(value="Describe the land-cover, major objects, and any notable environmental features."),
        )
    if mode == "Change Pair":
        return (
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(value="What changed between these two dates, and where did the change occur?"),
        )
    return (
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(value="Use the optical and SAR images together to identify water and built-up regions."),
    )

def on_run_place(place, lat, lon, start_date, end_date, goal, query):
    try:
        lat_f = float(lat) if lat not in (None, "") else 28.6139
        lon_f = float(lon) if lon not in (None, "") else 77.2090
    except Exception:
        lat_f, lon_f = 28.6139, 77.2090

    place = place.strip() if place else "User-provided location"
    query = query.strip() if query else "Describe what is happening in this area."

    return run_place_workflow(
        place=place,
        lat=lat_f,
        lon=lon_f,
        start_date=start_date,
        end_date=end_date,
        goal=goal,
        query=query,
    )

def on_run_upload(mode, img1, img2, query):
    p1 = safe_preview_from_file(img1)
    p2 = safe_preview_from_file(img2) if mode != "Single" else None

    if mode != "Single" and img2 is None:
        answer = (
            "## Upload missing\n\n"
            "This workflow needs **two images**.\n\n"
            "- Upload **Image 1** and **Image 2**\n"
            "- Ensure they are co-registered (same CRS + pixel dimensions)\n\n"
            "Tip: If you don’t have data, use **Search by Place** to auto-fetch scenes (demo)."
        )
        return answer, p1, p1, None, {"error": "missing_image_2"}, None

    answer_md, evidence, prev1, prev2, exec_summary, report_path = run_upload_workflow(
        mode=mode, query=query, preview1=p1, preview2=p2
    )

    return answer_md, evidence, prev1, prev2, exec_summary, report_path


with gr.Blocks(theme=THEME, css=CSS, title="SatQuery AI", elem_id="app") as demo:
    gr.HTML(HERO_HTML)

    with gr.Tabs() as tabs:
        # TAB 1 — Search by Place
        with gr.Tab("Search by Place", id="place"):
            with gr.Row():
                with gr.Column(scale=5, min_width=340, elem_classes=["glass", "cardPad"]):
                    gr.Markdown('<div class="sectionTitle">Step 1 — Describe your place</div>', elem_classes=["tightMd"])

                    place = gr.Textbox(
                        label="Place / landmark / description",
                        placeholder="e.g., 'near Chilika Lake, Odisha' or 'Haldwani, Uttarakhand'",
                    )

                    with gr.Row():
                        lat = gr.Textbox(label="Latitude (optional)", placeholder="e.g., 19.81")
                        lon = gr.Textbox(label="Longitude (optional)", placeholder="e.g., 85.31")

                    gr.Markdown("<hr>")
                    gr.Markdown('<div class="sectionTitle">Step 2 — Pick time & goal</div>', elem_classes=["tightMd"])

                    with gr.Row():
                        start_date = gr.Textbox(label="Start date", value="2024-01-01")
                        end_date = gr.Textbox(label="End date", value="2024-12-31")

                    goal = gr.Radio(
                        ["Understand scene", "Water / flooding", "Wildfire", "Urban growth", "Detect change", "Custom"],
                        value="Understand scene",
                        label="What do you want to know?",
                    )

                    query = gr.Textbox(
                        label="Your question (plain language)",
                        lines=4,
                        placeholder="e.g., 'Is there flooding? show where.'",
                        value="What is happening in this area? Summarize land-cover and any notable activity.",
                    )

                    gr.Markdown(
                        "<div class='smallHint'>For PPT/demo: this mode returns a realistic UI + trace. Later you can wire SentinelHub/Google Earth here.</div>"
                    )

                    with gr.Row():
                        run_place = gr.Button("Run Analysis", variant="primary")
                        reset_place = gr.Button("Reset", variant="secondary")

                with gr.Column(scale=7, min_width=420):
                    with gr.Group(elem_classes=["glass", "cardPad"]):
                        gr.Markdown('<div class="sectionTitle">Agent response</div>')
                        answer_place = gr.Markdown(value="*", elem_classes=["tightMd"])

                    with gr.Row():
                        with gr.Column(elem_classes=["glass", "cardPad", "imageFrame"]):
                            gr.Markdown('<div class="sectionTitle">Evidence overlay</div>')
                            evidence_place = gr.Image(type="numpy", height=280, show_label=False)

                        with gr.Column(elem_classes=["glass", "cardPad", "imageFrame"]):
                            gr.Markdown('<div class="sectionTitle">Optical (preview)</div>')
                            optical_place = gr.Image(type="numpy", height=280, show_label=False)

                    with gr.Row():
                        with gr.Column(elem_classes=["glass", "cardPad", "imageFrame"]):
                            gr.Markdown('<div class="sectionTitle">SAR (preview)</div>')
                            sar_place = gr.Image(type="numpy", height=260, show_label=False)

                        with gr.Column():
                            map_html = gr.HTML(value="", elem_classes=[])

                    with gr.Accordion("Execution Trace (auditable JSON)", open=False, elem_classes=["glass", "cardPad"]):
                        trace_place = gr.JSON(show_label=False)

                    report_place = gr.File(label="Download report", interactive=False)

            def _reset_place():
                return (
                    "", "", "",
                    "2024-01-01", "2024-12-31",
                    "Understand scene",
                    "What is happening in this area? Summarize land-cover and any notable activity.",
                    "", None, None, None, "", {}, None
                )

            run_place.click(
                fn=on_run_place,
                inputs=[place, lat, lon, start_date, end_date, goal, query],
                outputs=[answer_place, evidence_place, optical_place, sar_place, map_html, trace_place, report_place],
            )
            reset_place.click(
                fn=_reset_place,
                inputs=[],
                outputs=[place, lat, lon, start_date, end_date, goal, query, answer_place, evidence_place, optical_place, sar_place, map_html, trace_place, report_place],
            )

        # TAB 2 — Upload Images
        with gr.Tab("Upload Images", id="upload"):
            with gr.Row():
                with gr.Column(scale=5, min_width=340, elem_classes=["glass", "cardPad"]):
                    gr.Markdown('<div class="sectionTitle">Step 1 — Choose workflow</div>', elem_classes=["tightMd"])

                    mode = gr.Radio(
                        choices=["Single", "Change Pair", "Optical+SAR Pair"],
                        value="Single",
                        label="Workflow",
                    )

                    gr.Markdown("<div class='smallHint'>Single → caption/VQA. Change Pair → bi-temporal. Optical+SAR → cross-modal fusion.</div>")
                    gr.Markdown("<hr>")

                    gr.Markdown('<div class="sectionTitle">Step 2 — Upload data</div>', elem_classes=["tightMd"])
                    img1 = gr.File(
                        label="Image 1 (GeoTIFF / TIFF / PNG / JPG)",
                        file_count="single",
                        file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg"],
                    )
                    img2 = gr.File(
                        label="Image 2 (required for paired modes)",
                        file_count="single",
                        file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg"],
                        visible=False,
                    )

                    img2_hint = gr.Markdown(
                        "<div class='smallHint'>Paired modes expect co-registered scenes (same CRS + pixel dimensions).</div>",
                        visible=False,
                    )

                    gr.Markdown("<hr>")
                    gr.Markdown('<div class="sectionTitle">Step 3 — Ask</div>', elem_classes=["tightMd"])

                    query_u = gr.Textbox(
                        label="Query",
                        lines=4,
                        value="Describe the land-cover, major objects, and any notable environmental features.",
                        placeholder="Ask in plain language…",
                    )

                    with gr.Row():
                        run_upload = gr.Button("Run Analysis", variant="primary")
                        reset_upload = gr.Button("Reset", variant="secondary")

                with gr.Column(scale=7, min_width=420):
                    with gr.Group(elem_classes=["glass", "cardPad"]):
                        gr.Markdown('<div class="sectionTitle">Agent response</div>')
                        answer_u = gr.Markdown(value="*", elem_classes=["tightMd"])

                    with gr.Row():
                        with gr.Column(elem_classes=["glass", "cardPad", "imageFrame"]):
                            gr.Markdown('<div class="sectionTitle">Evidence overlay</div>')
                            evidence_u = gr.Image(type="numpy", height=280, show_label=False)

                        with gr.Column(elem_classes=["glass", "cardPad", "imageFrame"]):
                            gr.Markdown('<div class="sectionTitle">Image 1 preview</div>')
                            prev1_u = gr.Image(type="numpy", height=280, show_label=False)

                    with gr.Row():
                        with gr.Column(elem_classes=["glass", "cardPad", "imageFrame"], visible=False) as img2_prev_col:
                            gr.Markdown('<div class="sectionTitle">Image 2 preview</div>')
                            prev2_u = gr.Image(type="numpy", height=260, show_label=False)

                        with gr.Column(elem_classes=["glass", "cardPad"]):
                            gr.Markdown('<div class="sectionTitle">Deliverables</div>')
                            report_u = gr.File(label="Download report", interactive=False)

                    with gr.Accordion("Execution Trace (auditable JSON)", open=False, elem_classes=["glass", "cardPad"]):
                        trace_u = gr.JSON(show_label=False)

            mode.change(fn=update_ui_for_upload_mode, inputs=[mode], outputs=[img2, img2_prev_col, query_u]).then(
                fn=lambda m: gr.update(visible=(m != "Single")),
                inputs=[mode],
                outputs=[img2_hint],
            )

            run_upload.click(
                fn=on_run_upload,
                inputs=[mode, img1, img2, query_u],
                outputs=[answer_u, evidence_u, prev1_u, prev2_u, trace_u, report_u],
            )

            def _reset_upload():
                return (
                    "Single",
                    None,
                    gr.update(visible=False, value=None),
                    gr.update(visible=False),
                    gr.update(value="Describe the land-cover, major objects, and any notable environmental features."),
                    "",
                    None, None, None, {}, None
                )

            reset_upload.click(
                fn=_reset_upload,
                inputs=[],
                outputs=[mode, img1, img2, img2_prev_col, query_u, answer_u, evidence_u, prev1_u, prev2_u, trace_u, report_u],
            )


if __name__ == "__main__":
    demo.launch(share=True)
