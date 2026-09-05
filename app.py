# ===== ./app.py =====
import gradio as gr
import numpy as np
from PIL import Image
from mock_backend import run_place_workflow, run_upload_workflow

# ==========================================
# 1. CUSTOM CSS FOR DASHBOARD LOOK
# ==========================================
CSS = """
/* Import font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Base styling overrides */
.gradio-container {
    font-family: 'Inter', sans-serif !important;
    background-color: #0b0e14 !important; /* Deep dark background */
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Hide default footer and margins */
footer { display: none !important; }
.contain { padding: 0 !important; max-width: 100% !important; }

/* Wrapper for the whole dashboard */
#dashboard-wrapper {
    display: flex;
    flex-direction: row;
    height: 100vh;
    width: 100vw;
}

/* SIDEBAR STYLING */
#sidebar {
    width: 260px !important;
    min-width: 260px !important;
    max-width: 260px !important;
    background-color: #0b0e14 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    padding: 24px 16px !important;
    display: flex;
    flex-direction: column;
    height: 100vh;
}

.brand-title { color: white; font-weight: 700; font-size: 18px; margin: 0; display: flex; align-items: center; gap: 8px;}
.brand-sub { color: #6b7280; font-size: 11px; margin-top: -2px; margin-left: 28px; margin-bottom: 30px;}

.nav-btn {
    width: 100%; text-align: left; padding: 12px 16px; border-radius: 8px;
    font-size: 13px; font-weight: 500; border: none; cursor: pointer;
    margin-bottom: 8px; background: transparent; color: #9ca3af;
}
.nav-btn.primary {
    background: linear-gradient(90deg, #8b5cf6, #6d28d9);
    color: white; font-weight: 600;
}
.nav-btn:hover:not(.primary) { background: rgba(255,255,255,0.05); color: white; }
.nav-btn.active { background: rgba(255,255,255,0.05); color: white; }

/* MAIN CONTENT AREA */
#main-content {
    flex-grow: 1;
    background-color: #111520 !important; /* Slightly lighter than sidebar */
    padding: 40px 60px !important;
    overflow-y: auto;
    height: 100vh;
    border-top-left-radius: 20px;
}

/* Header & Search */
.greeting-text { color: white; font-size: 24px; font-weight: 600; margin-bottom: 24px;}
.greeting-text span { color: #6b7280; font-size: 14px; font-weight: 400; }

#search-container {
    background: #171b29 !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 12px !important;
    padding: 4px 4px 4px 16px !important;
}
#search-container textarea {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: white !important;
    font-size: 14px !important;
}

/* Send Button */
#send-btn {
    background: #8b5cf6 !important;
    color: white !important;
    border-radius: 8px !important;
    width: 40px !important; height: 40px !important;
    min-width: 40px !important; padding: 0 !important;
    display: flex; justify-content: center; align-items: center;
}

/* Mode Tabs */
.mode-tabs { margin-top: 24px; margin-bottom: 24px; gap: 12px; }
.mode-btn {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #9ca3af !important;
    border-radius: 20px !important;
    padding: 8px 16px !important;
    font-size: 13px !important;
}
.mode-btn.selected {
    background: rgba(139, 92, 246, 0.1) !important;
    border-color: #8b5cf6 !important;
    color: #8b5cf6 !important;
}

/* Dynamic areas (Upload vs Autofetch) */
.dynamic-area {
    background: #171b29 !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 12px !important;
    padding: 24px !important;
}

/* Results Card */
#results-card {
    background: #171b29 !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 16px !important;
    padding: 32px !important;
    margin-top: 32px !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2) !important;
}

/* Custom HTML styling for results list */
.icon-list { list-style: none; padding: 0; margin-top: 20px; }
.icon-list-item { display: flex; align-items: flex-start; margin-bottom: 24px; }
.icon-circle {
    width: 32px; height: 32px; border-radius: 50%; display: flex; justify-content: center;
    align-items: center; margin-right: 16px; font-size: 14px; flex-shrink: 0; color: white;
}
.ic-red { background: #ef4444; } .ic-blue { background: #3b82f6; }
.ic-green { background: #10b981; } .ic-yellow { background: #f59e0b; } .ic-purple { background: #8b5cf6; }
.item-text h4 { margin: 0 0 4px 0; font-size: 14px; color: white; font-weight: 600;}
.item-text p { margin: 0; font-size: 13px; color: #9ca3af; line-height: 1.4;}

/* Upload boxes override */
.gradio-image { background: transparent !important; border: 1px dashed rgba(255,255,255,0.1) !important; }
"""

# ==========================================
# 2. LOGIC FUNCTIONS
# ==========================================
def change_mode(mode_name):
    # Determine visibility for upload areas based on the mode clicked
    show_up1 = mode_name in ["Single Image", "Optical + SAR", "Change Detection"]
    show_up2 = mode_name in ["Optical + SAR", "Change Detection"]
    show_auto = mode_name == "Autofetch"
    
    # Update button styles
    modes = ["Single Image", "Optical + SAR", "Change Detection", "Autofetch"]
    btn_updates = [gr.update(elem_classes=["mode-btn", "selected"] if m == mode_name else ["mode-btn"]) for m in modes]
    
    # Update area visibility
    area_updates = [
        gr.update(visible=show_up1 or show_up2),  # The whole upload group
        gr.update(visible=show_up1),              # Image 1
        gr.update(visible=show_up2),              # Image 2
        gr.update(visible=show_auto)              # Autofetch text area
    ]
    
    return btn_updates + area_updates + [mode_name]

def process_query(mode, query, img1, img2, place_text):
    query = (query or "").strip() or "What are the main land cover types in this image?"
    
    if mode == "Autofetch":
        ans_html, evidence, _, _, _, exec_s, _ = run_place_workflow(
            place=place_text or "Coastal Region", lat=0, lon=0, start_date="", end_date="", goal="Understand scene", query=query
        )
    else:
        ans_html, evidence, _, _, exec_s, _ = run_upload_workflow(
            mode=mode, query=query, preview1=img1, preview2=img2
        )
        
    return gr.update(visible=True), ans_html, evidence, exec_s

# ==========================================
# 3. UI LAYOUT
# ==========================================
with gr.Blocks(title="SatQuery AI") as demo:
    
    # Using a row to act as our full-screen flex container
    with gr.Row(elem_id="dashboard-wrapper"):
        
        # --- LEFT SIDEBAR ---
        with gr.Column(elem_id="sidebar", scale=0):
            gr.HTML("""
            <div class="brand-title">☄️ SatQuery AI</div>
            <div class="brand-sub">Vision-Language Assistant</div>
            
            <button class="nav-btn primary">+ New Query</button>
            <button class="nav-btn active">🏠 Home</button>
            <button class="nav-btn">⏱ History</button>
            
            <div style="margin-top: auto;">
                <div style="background: #111520; border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 12px; display: flex; align-items: center; gap: 10px;">
                    <div style="width: 8px; height: 8px; background: #10b981; border-radius: 50%;"></div>
                    <div>
                        <div style="color: white; font-size: 12px; font-weight: 500;">System Status</div>
                        <div style="color: #6b7280; font-size: 10px;">All systems operational</div>
                    </div>
                </div>
            </div>
            """)
            
        # --- MAIN CONTENT ---
        with gr.Column(elem_id="main-content", scale=1):
            
            # Header
            gr.HTML('<div class="greeting-text">Good morning! 👋 <span>Ask anything about your remote sensing imagery.</span></div>')
            
            # Search Bar
            with gr.Row(elem_id="search-container", equal_height=True):
                query_input = gr.Textbox(
                    placeholder='Try: "What are the main land cover types in this image?"',
                    show_label=False, container=False, lines=1, scale=1
                )
                submit_btn = gr.Button("↗", elem_id="send-btn")
            
            # Mode Tabs
            with gr.Row(elem_classes="mode-tabs"):
                btn_single = gr.Button("🖼️ Single Image", elem_classes=["mode-btn"])
                btn_fusion = gr.Button("🎯 Optical + SAR", elem_classes=["mode-btn"])
                btn_change = gr.Button("⚡ Change Detection", elem_classes=["mode-btn"])
                btn_auto = gr.Button("✨ Autofetch", elem_classes=["mode-btn", "selected"])
            
            # State tracker for current mode
            current_mode = gr.State("Autofetch")
            
            # Dynamic Input Areas
            with gr.Group(elem_classes="dynamic-area"):
                
                # State 1: Autofetch
                with gr.Column(visible=True) as area_auto:
                    gr.HTML("<h4 style='color:#8b5cf6; margin-top:0;'>✨ Autofetch Mode</h4><p style='color:#9ca3af; font-size: 13px; margin-bottom: 12px;'>Describe your area of interest, and we'll automatically fetch the best available satellite data and provide insights.</p>")
                    place_input = gr.Textbox(placeholder="E.g., Coastal Region, San Francisco", show_label=False)
                
                # State 2: Uploads
                with gr.Row(visible=False) as area_upload:
                    img1 = gr.Image(label="Image 1", type="numpy", visible=True)
                    img2 = gr.Image(label="Image 2 (SAR/Post)", type="numpy", visible=False)

            # Results Area (Hidden by default)
            with gr.Group(visible=False, elem_id="results-card") as results_area:
                gr.HTML('<div style="background: rgba(139, 92, 246, 0.1); color: #8b5cf6; border: 1px solid rgba(139, 92, 246, 0.2); padding: 4px 12px; border-radius: 20px; display: inline-block; font-size: 12px; font-weight: 600; margin-bottom: 20px;">💬 Answer</div>')
                
                with gr.Row():
                    # Text side
                    with gr.Column(scale=1):
                        answer_html = gr.HTML()
                        gr.HTML('<div style="margin-top: 20px; background: rgba(16, 185, 129, 0.1); color: #10b981; padding: 6px 16px; border-radius: 20px; display: inline-block; font-size: 13px; font-weight: 600;">Confidence Score &nbsp;&nbsp; 0.88</div>')
                    
                    # Image Side
                    with gr.Column(scale=1):
                        result_img = gr.Image(show_label=False, interactive=False)
                
                with gr.Accordion("🛠️ View Agent Execution Trace", open=False):
                    trace_json = gr.JSON()

    # --- Events ---
    mode_btns = [btn_single, btn_fusion, btn_change, btn_auto]
    ui_areas = [area_upload, img1, img2, area_auto, current_mode]

    # Map buttons to the change_mode logic
    btn_single.click(fn=lambda: change_mode("Single Image"), outputs=mode_btns + ui_areas)
    btn_fusion.click(fn=lambda: change_mode("Optical + SAR"), outputs=mode_btns + ui_areas)
    btn_change.click(fn=lambda: change_mode("Change Detection"), outputs=mode_btns + ui_areas)
    btn_auto.click(fn=lambda: change_mode("Autofetch"), outputs=mode_btns + ui_areas)

    # Trigger processing on Send button
    submit_btn.click(
        fn=process_query,
        inputs=[current_mode, query_input, img1, img2, place_input],
        outputs=[results_area, answer_html, result_img, trace_json]
    )

# Note: In Gradio 6, css and theme are passed to launch()
if __name__ == "__main__":
    demo.launch(css=CSS, share=True)
