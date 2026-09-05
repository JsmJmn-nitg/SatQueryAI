import gradio as gr
from io_utils import read_geotiff, to_rgb_preview, check_pair_compatible
from controller import route_query

def run_satquery(mode, img1_file, img2_file, query):
    # 0) Safety check
    if img1_file is None:
        return "⚠️ Error: Please upload at least Image 1.", None, None, {"error": "Missing input"}

    # 1) Read Image 1
    arr1, meta1 = read_geotiff(img1_file.name)
    preview1 = to_rgb_preview(arr1)
    
    arr2 = meta2 = preview2 = None
    
    # 2) Read Image 2 (If provided)
    if img2_file is not None:
        arr2, meta2 = read_geotiff(img2_file.name)
        preview2 = to_rgb_preview(arr2)
        
        # Enforce compatibility check for paired modes
        if mode in ["Change Pair", "Optical+SAR Pair"]:
            compat = check_pair_compatible(arr1, meta1, arr2, meta2)
            if not compat["ok_shape"]:
                return "⚠️ Error: Images are not compatible (different dimensions).", preview1, preview2, compat

    # 3) Route to Agentic Controller
    answer, evidence, exec_summary = route_query(mode, arr1, meta1, arr2, meta2, query)
    
    # 4) Fallback to Image 1 preview if no overlay is generated
    if evidence is None:
        evidence = preview1

    return answer, evidence, preview2, exec_summary

# ==========================================
# CUSTOM THEME & CSS FOR MODERN UI
# ==========================================
custom_theme = gr.themes.Soft(
    primary_hue="amber",       # Warm gold/orange accents like the reference image
    neutral_hue="slate",       # Deep blue-grays for the background
    radius_size="lg",          # Highly rounded corners
    spacing_size="md",
).set(
    body_background_fill="*neutral_950",
    body_background_fill_dark="#0a0a0f", # Very dark background
    block_background_fill="*neutral_900",
    block_background_fill_dark="#13131a", # Slightly lighter panels
    block_border_width="1px",
    block_border_color_dark="#2a2a35",
    button_primary_background_fill="#d97706",
    button_primary_background_fill_dark="#d97706",
    button_primary_text_color="white",
)

custom_css = """
/* Make the whole app look like a dashboard */
.gradio-container {
    font-family: 'Inter', system-ui, sans-serif;
}
/* Glassmorphism effect for main panels */
.glass-panel {
    background: rgba(19, 19, 26, 0.7) !important;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
}
/* Glow effect on the primary button */
.primary-btn {
    box-shadow: 0 0 15px rgba(217, 119, 6, 0.4) !important;
    transition: all 0.3s ease-in-out !important;
}
.primary-btn:hover {
    box-shadow: 0 0 25px rgba(217, 119, 6, 0.7) !important;
    transform: translateY(-2px);
}
/* Style the header */
.dash-header {
    text-align: center;
    color: #f3f4f6;
    margin-bottom: 1rem;
}
.dash-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    background: -webkit-linear-gradient(45deg, #f59e0b, #d97706);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
"""

# ==========================================
# UI LAYOUT USING GR.BLOCKS
# ==========================================
with gr.Blocks(theme=custom_theme, css=custom_css) as demo:
    
    # Header Section
    with gr.Column(elem_classes=["dash-header"]):
        gr.Markdown("# 🛰️ SatQuery AI Dashboard")
        gr.Markdown("Upload a satellite image (or a pair), select an agentic mode, and ask questions in plain language.")

    # Main Dashboard Grid
    with gr.Row():
        
        # LEFT COLUMN: Inputs & Controls (The "Cockpit")
        with gr.Column(scale=3, elem_classes=["glass-panel"]):
            gr.Markdown("### 🎛️ Control Panel")
            
            mode_dropdown = gr.Dropdown(
                choices=["Single", "Change Pair", "Optical+SAR Pair"], 
                value="Single", 
                label="Agentic Task Mode",
                info="Select the type of analysis you want the agent to perform."
            )
            
            with gr.Row():
                img1_upload = gr.File(label="Image 1 (GeoTIFF)", file_count="single")
                img2_upload = gr.File(label="Image 2 (GeoTIFF - Optional)", file_count="single")
                
            query_input = gr.Textbox(
                label="Query / Instructions", 
                value="Describe the land-cover and major objects.",
                placeholder="Ask the AI something about the map...",
                lines=3
            )
            
            submit_btn = gr.Button("🚀 Execute AI Analysis", variant="primary", elem_classes=["primary-btn"])

        # RIGHT COLUMN: Outputs & Visuals (The "Display")
        with gr.Column(scale=7):
            
            # Top Output: AI Answer
            with gr.Row(elem_classes=["glass-panel"]):
                ai_answer = gr.Textbox(label="✨ AI Insights & Answer", lines=2, interactive=False)
            
            # Middle Output: Image Visualizers
            with gr.Row():
                with gr.Column(elem_classes=["glass-panel"]):
                    evidence_img = gr.Image(label="Processed Overlay / Evidence", interactive=False)
                with gr.Column(elem_classes=["glass-panel"]):
                    preview2_img = gr.Image(label="Image 2 Preview", interactive=False)
                    
            # Bottom Output: Execution Trace
            with gr.Row(elem_classes=["glass-panel"]):
                exec_summary = gr.JSON(label="Agentic Execution Summary (Traceability)")

    # ==========================================
    # EVENT WIRING (Connecting the button to the function)
    # ==========================================
    submit_btn.click(
        fn=run_satquery,
        inputs=[mode_dropdown, img1_upload, img2_upload, query_input],
        outputs=[ai_answer, evidence_img, preview2_img, exec_summary]
    )

if __name__ == "__main__":
    demo.launch(share=True)
