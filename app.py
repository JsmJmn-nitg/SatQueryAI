import gradio as gr
from io_utils import read_geotiff, to_rgb_preview, check_pair_compatible
from controller import route_query

def run_satquery(mode, img1_file, img2_file, query):
    # 0) Safety check
    if img1_file is None:
        return "⚠️ Error: Please upload at least Image 1.", None, None, None, {"error": "Missing input"}

    # 1) Read Image 1
    arr1, meta1 = read_geotiff(img1_file.name)
    preview1 = to_rgb_preview(arr1)
    
    arr2 = meta2 = preview2 = None
    
    # 2) Read Image 2 (If provided)
    if img2_file is not None and mode != "Single":
        arr2, meta2 = read_geotiff(img2_file.name)
        preview2 = to_rgb_preview(arr2)
        
        # Enforce compatibility check for paired modes
        if mode in ["Change Pair", "Optical+SAR Pair"]:
            compat = check_pair_compatible(arr1, meta1, arr2, meta2)
            if not compat["ok_shape"]:
                return "⚠️ Error: Images are not compatible (different dimensions).", preview1, preview1, preview2, compat

    # 3) Route to Agentic Controller
    answer, evidence, exec_summary = route_query(mode, arr1, meta1, arr2, meta2, query)
    
    # 4) Fallback to Image 1 preview if no overlay is generated
    if evidence is None:
        evidence = preview1

    return answer, evidence, preview1, preview2, exec_summary

# ==========================================
# DYNAMIC UI LOGIC
# ==========================================
def update_ui_for_mode(mode):
    """Dynamically updates the UI based on the dropdown selection."""
    if mode == "Single":
        return [
            gr.update(visible=False), # Hide Image 2 input
            gr.update(value="Describe the land-cover and major objects.", placeholder="Ask about this image..."),
            gr.update(visible=False)  # Hide Image 2 output preview
        ]
    elif mode == "Change Pair":
        return [
            gr.update(visible=True),  # Show Image 2 input
            gr.update(value="What changed between these two dates? Identify major differences.", placeholder="Ask about changes..."),
            gr.update(visible=True)   # Show Image 2 output preview
        ]
    else: # Optical+SAR Pair
        return [
            gr.update(visible=True),  # Show Image 2 input
            gr.update(value="Use both optical and SAR data to identify built-up areas and water bodies.", placeholder="Ask about fusion..."),
            gr.update(visible=True)   # Show Image 2 output preview
        ]

# ==========================================
# HIGH-END CSS (Glassmorphism & Gold)
# ==========================================
custom_theme = gr.themes.Monochrome(
    neutral_hue="slate",
    radius_size="lg",
).set(
    body_background_fill="#0b0f19", # Deep space dark
    body_background_fill_dark="#0b0f19",
    block_background_fill="transparent",
    block_border_width="0px",
)

custom_css = """
/* Global App Font & Background */
.gradio-container {
    font-family: 'Inter', -apple-system, sans-serif;
    background: radial-gradient(circle at top right, #1a1c29, #0b0f19) !important;
    color: #e2e8f0;
}

/* Glassmorphism Panels with Gold Borders */
.glass-panel {
    background: rgba(20, 25, 40, 0.4) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(212, 175, 55, 0.2) !important; 
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5) !important;
    border-radius: 16px !important;
    padding: 15px;
}

/* Highlighted Gold Data Points / Titles */
.gold-text h1, .gold-text h2, .gold-text h3 {
    background: linear-gradient(135deg, #F3E7E9 0%, #D4AF37 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
}

/* Translucent Chat Box (Query Input & Answer Output) */
.glass-chat textarea {
    background: rgba(0, 0, 0, 0.3) !important;
    border: 1px solid rgba(212, 175, 55, 0.3) !important;
    color: #f8fafc !important;
    box-shadow: inset 0 2px 10px rgba(0,0,0,0.5) !important;
    border-radius: 12px !important;
    font-size: 15px;
}
.glass-chat textarea:focus {
    border: 1px solid #d4af37 !important;
    box-shadow: 0 0 10px rgba(212, 175, 55, 0.4) !important;
}

/* Gold Futuristic Submit Button */
.gold-btn {
    background: linear-gradient(135deg, #d4af37, #997a15) !important;
    color: #000000 !important;
    font-weight: 800 !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(212, 175, 55, 0.4) !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.gold-btn:hover {
    transform: scale(1.02);
    box-shadow: 0 6px 25px rgba(212, 175, 55, 0.6) !important;
}

/* Code block (JSON Trace) styling */
.json-trace {
    background: #0d1117 !important;
    border: 1px solid rgba(212, 175, 55, 0.15) !important;
    border-radius: 8px !important;
}
"""

# ==========================================
# UI LAYOUT
# ==========================================
with gr.Blocks(theme=custom_theme, css=custom_css, title="SatQuery AI") as demo:
    
    # Header
    with gr.Row(elem_classes=["gold-text"]):
        gr.Markdown("# 🛰️ SatQuery AI: Agentic Geospatial Intelligence")

    with gr.Row():
        
        # LEFT COLUMN (Inputs / Configuration)
        with gr.Column(scale=3, elem_classes=["glass-panel"]):
            gr.Markdown("### ⚙️ Configure Your Query", elem_classes=["gold-text"])
            
            mode_dropdown = gr.Dropdown(
                choices=["Single", "Change Pair", "Optical+SAR Pair"], 
                value="Single", 
                label="Input Mode",
            )
            
            img1_upload = gr.File(label="Image 1 (GeoTIFF / TIFF)", file_count="single")
            img2_upload = gr.File(label="Image 2 (Optional)", file_count="single", visible=False) # Hidden by default
                
            query_input = gr.Textbox(
                label="Query", 
                value="Describe the land-cover and major objects.",
                lines=3,
                elem_classes=["glass-chat"]
            )
            
            submit_btn = gr.Button("🚀 Run Agentic Query", variant="primary", elem_classes=["gold-btn"])

        # RIGHT COLUMN (Outputs / Dashboard)
        with gr.Column(scale=7):
            
            # Top Output: AI Answer
            with gr.Row(elem_classes=["glass-panel"]):
                with gr.Column():
                    gr.Markdown("### ✨ Agent Response", elem_classes=["gold-text"])
                    ai_answer = gr.Textbox(show_label=False, lines=3, interactive=False, elem_classes=["glass-chat"])
            
            # Middle Output: 3-Grid Image Viewer
            with gr.Row():
                with gr.Column(elem_classes=["glass-panel"]):
                    gr.Markdown("**Evidence / Overlay**")
                    evidence_img = gr.Image(show_label=False, interactive=False)
                with gr.Column(elem_classes=["glass-panel"]):
                    gr.Markdown("**Image 1 (Preview)**")
                    preview1_img = gr.Image(show_label=False, interactive=False)
                with gr.Column(elem_classes=["glass-panel"], visible=False) as img2_container:
                    # Wrapped in a container so we can hide/show the whole column dynamically
                    gr.Markdown("**Image 2 (Preview)**")
                    preview2_img = gr.Image(show_label=False, interactive=False)
                    
            # Bottom Output: Execution Trace
            with gr.Row(elem_classes=["glass-panel"]):
                with gr.Column():
                    gr.Markdown("### 📜 Execution Summary (JSON Trace)", elem_classes=["gold-text"])
                    exec_summary = gr.JSON(show_label=False, elem_classes=["json-trace"])

    # ==========================================
    # EVENT WIRING (Dynamic Actions)
    # ==========================================
    
    # 1. Update UI when Dropdown changes
    mode_dropdown.change(
        fn=update_ui_for_mode,
        inputs=[mode_dropdown],
        outputs=[img2_upload, query_input, img2_container]
    )

    # 2. Run the actual AI logic when button is clicked
    submit_btn.click(
        fn=run_satquery,
        inputs=[mode_dropdown, img1_upload, img2_upload, query_input],
        outputs=[ai_answer, evidence_img, preview1_img, preview2_img, exec_summary]
    )

if __name__ == "__main__":
    demo.launch(share=True)
