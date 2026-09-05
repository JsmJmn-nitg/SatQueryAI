import gradio as gr
from io_utils import read_geotiff, to_rgb_preview, check_pair_compatible
from controller import route_query

def run_satquery(mode, img1_file, img2_file, query, progress=gr.Progress(track_tqdm=False)):
    import traceback
    
    base_trace = {"mode": mode, "query": query, "tools_used": []}

    try:
        # 0) Validate inputs
        if img1_file is None:
            return ("Error: Please upload Image 1.", None, None, None, {"error": "missing_img1", **base_trace})

        if mode != "Single" and img2_file is None:
            arr1, meta1 = read_geotiff(img1_file.name)
            preview1 = to_rgb_preview(arr1)
            return ("Error: Please upload Image 2 for this mode.", preview1, preview1, None, {"error": "missing_img2", **base_trace})

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

            if mode in ["Change Pair", "Optical+SAR Pair"]:
                compat = check_pair_compatible(arr1, meta1, arr2, meta2)
                if not (compat["ok_shape"] and compat["ok_crs"]):
                    return ("Error: Images must match in dimensions and CRS for paired analysis.", preview1, preview1, preview2, {"error": "incompatible_pair", **compat, **base_trace})

        # 3) Route to controller
        progress(0.60, desc="Agent Orchestrating Tools...")
        
        answer, evidence, exec_summary = route_query(
            mode, arr1, meta1, arr2, meta2, query, preview1
        )
        
        progress(0.95, desc="Rendering outputs")
        if evidence is None:
            evidence = preview1

        return answer, evidence, preview1, preview2, exec_summary
        
    except Exception as e:
        traceback.print_exc()
        error_msg = f"❌ **Error in processing:**\n\n```\n{type(e).__name__}: {str(e)}\n```"
        return (error_msg, None, None, None, {"error": str(e), "type": type(e).__name__, **base_trace})

def update_ui_for_mode(mode):
    if mode == "Single":
        return (gr.update(visible=False, value=None), gr.update(value="Describe the land-cover types and any notable environmental features visible in this satellite image.", placeholder="Ask about Image 1…"), gr.update(visible=False))
    if mode == "Change Pair":
        return (gr.update(visible=True), gr.update(value="What changed between these two dates? Identify the differences.", placeholder="Ask about change..."), gr.update(visible=True))
    return (gr.update(visible=True), gr.update(value="Use both optical and SAR data to identify water bodies. Explain the findings.", placeholder="Ask about fusion results..."), gr.update(visible=True))

theme = gr.themes.Soft(primary_hue="amber", neutral_hue="slate", radius_size="lg")

css = """
body { font-family: 'Inter', sans-serif; background: #0b0f19; color: #e2e8f0; }
.card { background: rgba(15, 23, 42, 0.7) !important; border: 1px solid rgba(148, 163, 184, 0.2) !important; border-radius: 12px !important; }
"""

with gr.Blocks(theme=theme, css=css, title="SatQuery AI") as demo:
    gr.Markdown("# 🛰️ SatQuery AI: Agentic Remote Sensing Assistant")
    
    with gr.Row():
        with gr.Column(scale=4, elem_classes=["card"]):
            mode_dropdown = gr.Radio(["Single", "Change Pair", "Optical+SAR Pair"], value="Single", label="Agent Mode")
            img1_upload = gr.File(label="Image 1 (GeoTIFF)", file_types=[".tif", ".tiff"])
            img2_upload = gr.File(label="Image 2 (Required for Pairs)", visible=False, file_types=[".tif", ".tiff"])
            query_input = gr.Textbox(label="Query", lines=4)
            submit_btn = gr.Button("Execute Agent Workflow", variant="primary")

        with gr.Column(scale=8):
            ai_answer = gr.Textbox(label="🤖 Agent Response", lines=6, interactive=False, elem_classes=["card"])
            with gr.Row():
                evidence_img = gr.Image(label="📊 Visual Evidence", type="numpy", height=300)
                preview1_img = gr.Image(label="🖼️ Image 1 Preview", type="numpy", height=300)
                img2_container = gr.Image(label="🖼️ Image 2 Preview", type="numpy", height=300, visible=False)
            exec_summary = gr.JSON(label="🔍 Auditable Execution Trace (JSON)")

    mode_dropdown.change(fn=update_ui_for_mode, inputs=[mode_dropdown], outputs=[img2_upload, query_input, img2_container])
    submit_btn.click(fn=run_satquery, inputs=[mode_dropdown, img1_upload, img2_upload, query_input], outputs=[ai_answer, evidence_img, preview1_img, img2_container, exec_summary])

if __name__ == "__main__":
    demo.launch(share=True)
