# ===== ./app.py =====
import gradio as gr
from ui_styles import THEME, CSS
from mock_backend import run_place_workflow, run_upload_workflow

# ================= HTML TEMPLATES =================
SIDEBAR_HTML = """
<div class="brand">
    <div class="brand-icon">☄️</div>
    <div class="brand-text">
        <h1>SatQuery AI</h1>
        <p>Vision-Language Assistant</p>
    </div>
</div>
<button class="nav-btn btn-new">＋ New Query</button>
<button class="nav-btn btn-nav active">🏠 Home</button>
<button class="nav-btn btn-nav">⏱ History</button>

<div class="sidebar-bottom">
    <div class="status-card">
        <div class="status-dot"></div>
        <div>System Status<br><span style="color:var(--text-muted);font-size:10px;">All systems operational</span></div>
    </div>
</div>
"""

HEADER_HTML = """
<div class="top-nav">
    <div></div>
    <div class="top-nav-right">
        <span>❓ Help</span>
        <span id="theme-toggle" style="cursor:pointer; font-size: 18px;">☀️</span>
        <div class="user-avatar">U</div>
        <span>User ⌄</span>
    </div>
</div>
"""

HERO_ART_HTML = """
<div class="hero-bg">
    <svg width="600" height="400" viewBox="0 0 600 400" xmlns="http://www.w3.org/2000/svg" style="opacity: 0.05;">
        <ellipse cx="400" cy="150" rx="150" ry="60" fill="none" stroke="currentColor" stroke-width="1" transform="rotate(-20 400 150)"/>
        <ellipse cx="400" cy="150" rx="200" ry="80" fill="none" stroke="currentColor" stroke-width="1" transform="rotate(-20 400 150)"/>
        <ellipse cx="400" cy="150" rx="250" ry="100" fill="none" stroke="currentColor" stroke-width="1" transform="rotate(-20 400 150)"/>
    </svg>
    <div class="planet-orb"></div>
</div>
"""

# ================= APP LOGIC =================
def handle_run(mode, query, img1, img2, place_text):
    query = (query or "").strip() or "What are the main land cover types in this image?"
    
    if mode == "Autofetch":
        answer_html, evidence, _, _, _, _, _ = run_place_workflow(
            place=place_text or "Coastal Region", lat=0, lon=0, start_date="", end_date="", goal="Understand scene", query=query
        )
    else:
        answer_html, evidence, _, _, _, _ = run_upload_workflow(
            mode=mode, query=query, preview1=img1, preview2=img2
        )
        
    return gr.update(visible=True), answer_html, evidence

def toggle_mode(btn_name):
    # Returns updates for the 4 mode buttons, the upload area, and autofetch area
    modes = ["Single Image", "Optical + SAR", "Change Detection", "Autofetch"]
    btn_updates = [gr.update(elem_classes=["mode-btn", "selected"] if m == btn_name else ["mode-btn"]) for m in modes]
    
    show_upload = btn_name in ["Single Image", "Optical + SAR", "Change Detection"]
    show_second_upload = btn_name in ["Optical + SAR", "Change Detection"]
    show_autofetch = btn_name == "Autofetch"
    
    area_updates = [
        gr.update(visible=show_upload),         # upload_area group
        gr.update(visible=show_second_upload),  # img_upload_2
        gr.update(visible=show_autofetch)       # autofetch_area
    ]
    return btn_updates + area_updates

# ================= UI LAYOUT =================
with gr.Blocks(theme=THEME, css=CSS, title="SatQuery AI") as demo:
    with gr.Row(elem_id="app-container"):
        
        # --- LEFT SIDEBAR ---
        with gr.Column(elem_id="sidebar-col"):
            gr.HTML(SIDEBAR_HTML)
                    
        # --- MAIN CONTENT ---
        with gr.Column(elem_id="main-col"):
            gr.HTML(HEADER_HTML)
            gr.HTML(HERO_ART_HTML)
            
            with gr.Column(elem_classes=["content-wrapper"]):
                
                # Header Texts
                gr.HTML("""
                    <div class="greeting">Good morning! 👋 <span>Ask anything about your remote sensing imagery.</span></div>
                """)
                
                # Chat Input Box
                with gr.Group(elem_classes=["search-box"]):
                    with gr.Row():
                        query_input = gr.Textbox(
                            placeholder='Try: "What are the main land cover types in this image?"', 
                            show_label=False, lines=1, max_lines=3, scale=1
                        )
                        submit_btn = gr.Button("↗", elem_classes=["send-btn-wrap"])
                
                # Mode Selectors
                with gr.Row(elem_classes=["mode-tabs"]):
                    m_single = gr.Button("🖼 Single Image", elem_classes=["mode-btn"])
                    m_fusion = gr.Button("🎯 Optical + SAR", elem_classes=["mode-btn"])
                    m_change = gr.Button("⚡ Change Detection", elem_classes=["mode-btn"])
                    m_auto   = gr.Button("✨ Autofetch", elem_classes=["mode-btn", "selected"])
                
                # Dynamic Input Area
                with gr.Group(elem_classes=["dynamic-area"]):
                    # Upload Area State
                    with gr.Row(visible=False, elem_classes=["upload-grid"]) as upload_area:
                        with gr.Column(elem_classes=["upload-box"]):
                            img_upload_1 = gr.Image(type="numpy", label="Upload Primary Image", elem_id="img1")
                        with gr.Column(elem_classes=["upload-box"], visible=False) as upload_area_2:
                            img_upload_2 = gr.Image(type="numpy", label="Upload Secondary Image (SAR/Post)", elem_id="img2")
                    
                    # Autofetch Area State
                    with gr.Row(visible=True, elem_classes=["autofetch-ui"]) as autofetch_area:
                        with gr.Column():
                            gr.HTML("<h3>✨ Autofetch Mode</h3><p>Describe your area of interest, and we'll automatically fetch the best available satellite data and provide insights.</p>")
                            place_input = gr.Textbox(placeholder="E.g., Coastal Region, San Francisco", show_label=False, container=False)
                            
                # Selected mode state tracking
                current_mode = gr.State("Autofetch")

                # Results Area (Hidden until run)
                with gr.Group(visible=False, elem_classes=["results-card"]) as results_area:
                    gr.HTML("<div class='answer-badge'>💬 Answer</div>")
                    
                    with gr.Row(elem_classes=["results-grid"]):
                        # Left Text Analysis
                        with gr.Column(elem_classes=["result-text-area"]):
                            answer_html = gr.HTML()
                            gr.HTML("""
                                <div class="confidence">
                                    Confidence Score <span class="conf-score">0.88</span>
                                </div>
                            """)
                            
                        # Right Image Analysis
                        with gr.Column(elem_classes=["result-image-area"]):
                            result_img = gr.Image(show_label=False, interactive=False)
                            gr.HTML("""
                                <div class="image-controls">
                                    <div class="img-btn">🔍</div><div class="img-btn">➕</div><div class="img-btn">🔲</div>
                                </div>
                                <div class="legend-box">
                                    <h5>Detected Objects</h5>
                                    <div class="legend-item"><div class="legend-color ic-red"></div> Built-up Area</div>
                                    <div class="legend-item"><div class="legend-color ic-blue"></div> Water Body</div>
                                    <div class="legend-item"><div class="legend-color ic-green"></div> Vegetation</div>
                                    <div class="legend-item"><div class="legend-color ic-yellow"></div> Roads</div>
                                    <div class="legend-item"><div class="legend-color ic-purple"></div> Bare Land</div>
                                </div>
                            """)

    # Events
    mode_btns = [m_single, m_fusion, m_change, m_auto]
    
    m_single.click(lambda: "Single Image", None, current_mode).then(toggle_mode, inputs=[gr.State("Single Image")], outputs=mode_btns + [upload_area, upload_area_2, autofetch_area])
    m_fusion.click(lambda: "Optical + SAR", None, current_mode).then(toggle_mode, inputs=[gr.State("Optical + SAR")], outputs=mode_btns + [upload_area, upload_area_2, autofetch_area])
    m_change.click(lambda: "Change Detection", None, current_mode).then(toggle_mode, inputs=[gr.State("Change Detection")], outputs=mode_btns + [upload_area, upload_area_2, autofetch_area])
    m_auto.click(lambda: "Autofetch", None, current_mode).then(toggle_mode, inputs=[gr.State("Autofetch")], outputs=mode_btns + [upload_area, upload_area_2, autofetch_area])

    submit_btn.click(
        fn=handle_run,
        inputs=[current_mode, query_input, img_upload_1, img_upload_2, place_input],
        outputs=[results_area, answer_html, result_img]
    )

    # Optional UI script to sync the sun/moon icon with Gradio's internal theme state
    demo.load(None, None, None, js="""
        () => {
            const toggle = document.getElementById('theme-toggle');
            toggle.addEventListener('click', () => {
                document.querySelector('.gradio-container').classList.toggle('dark');
            });
        }
    """)

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, share=True)
