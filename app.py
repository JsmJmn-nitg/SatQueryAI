# ===== ./app.py =====
import streamlit as st
import numpy as np
from PIL import Image
from mock_backend import run_place_workflow, run_upload_workflow

# 1. Page Configuration (Must be first)
st.set_page_config(
    page_title="SatQuery AI", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# 2. Inject Custom CSS for the Result Icons & Cards
st.markdown("""
<style>
    /* Styling for the custom HTML result list */
    .icon-list { list-style: none; padding: 0; margin-top: 15px; }
    .icon-list-item { display: flex; align-items: flex-start; margin-bottom: 20px; }
    .icon-circle { 
        width: 32px; height: 32px; border-radius: 50%; 
        display: flex; justify-content: center; align-items: center; 
        margin-right: 15px; color: white; font-size: 14px; flex-shrink: 0;
    }
    .ic-red { background-color: #ef4444; }
    .ic-blue { background-color: #3b82f6; }
    .ic-green { background-color: #10b981; }
    .ic-yellow { background-color: #f59e0b; }
    .ic-purple { background-color: #8b5cf6; }
    .item-text h4 { margin: 0 0 4px 0; font-size: 15px; font-weight: 600;}
    .item-text p { margin: 0; font-size: 14px; color: #9CA3AF; line-height: 1.4;}
    
    /* Hide Streamlit default top branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Make buttons look better */
    .stButton>button { border-radius: 8px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# 3. Sidebar Setup
with st.sidebar:
    st.markdown("## ☄️ SatQuery AI")
    st.markdown("<p style='color:gray; font-size:13px; margin-top:-10px;'>Vision-Language Assistant</p>", unsafe_allow_html=True)
    
    st.write("")
    if st.button("＋ New Query", type="primary", use_container_width=True):
        st.session_state.results = None
        st.rerun()
        
    st.button("🏠 Home", use_container_width=True)
    st.button("⏱ History", use_container_width=True)
    
    st.markdown("<br>" * 10, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
        <div style='display: flex; align-items: center; gap: 10px;'>
            <div style='width: 10px; height: 10px; border-radius: 50%; background-color: #10b981;'></div>
            <div>
                <div style='font-weight: 600; font-size: 14px;'>System Status</div>
                <div style='font-size: 12px; color: gray;'>All systems operational</div>
            </div>
        </div>
    """, unsafe_allow_html=True)


# 4. Main Content Area
st.markdown("<h1>Good morning! 👋 <span style='font-size: 18px; color: gray; font-weight: normal;'>Ask anything about your remote sensing imagery.</span></h1>", unsafe_allow_html=True)

# Search Input
query = st.text_input("Query", placeholder='Try: "What are the main land cover types in this image?"', label_visibility="collapsed")

# 5. Mode Selection using Native Streamlit Tabs
tab_single, tab_fusion, tab_change, tab_auto = st.tabs(["🖼️ Single Image", "🎯 Optical + SAR", "⚡ Change Detection", "✨ Autofetch Mode"])

img1, img2, place_input = None, None, ""
active_mode = "Autofetch"

with tab_single:
    st.markdown("#### Upload Image\n<small style='color:gray;'>GeoTIFF / TIFF / PNG (Max 200MB)</small>", unsafe_allow_html=True)
    file1 = st.file_uploader("Upload Primary Image", type=['tif', 'tiff', 'png', 'jpg'], label_visibility="collapsed")
    if file1: img1 = np.array(Image.open(file1))
    if st.button("Run Analysis", key="b1", type="primary"):
        active_mode, st.session_state.run = "Single Image", True

with tab_fusion:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Optical Image**")
        f1 = st.file_uploader("Opt", type=['tif', 'png', 'jpg'], key="f1_f", label_visibility="collapsed")
        if f1: img1 = np.array(Image.open(f1))
    with col2:
        st.markdown("**SAR Image**")
        f2 = st.file_uploader("SAR", type=['tif', 'png', 'jpg'], key="f2_f", label_visibility="collapsed")
        if f2: img2 = np.array(Image.open(f2))
    if st.button("Run Fusion Analysis", key="b2", type="primary"):
        active_mode, st.session_state.run = "Optical+SAR Pair", True

with tab_change:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Pre-Event Image**")
        c1 = st.file_uploader("Pre", type=['tif', 'png', 'jpg'], key="c1_c", label_visibility="collapsed")
        if c1: img1 = np.array(Image.open(c1))
    with col2:
        st.markdown("**Post-Event Image**")
        c2 = st.file_uploader("Post", type=['tif', 'png', 'jpg'], key="c2_c", label_visibility="collapsed")
        if c2: img2 = np.array(Image.open(c2))
    if st.button("Run Change Detection", key="b3", type="primary"):
        active_mode, st.session_state.run = "Change Pair", True

with tab_auto:
    st.info("Describe your area of interest, and we'll automatically fetch the best available satellite data and provide insights.")
    place_input = st.text_input("Location", placeholder="E.g., Coastal Region, San Francisco", label_visibility="collapsed")
    if st.button("Run Autofetch", key="b4", type="primary"):
        active_mode, st.session_state.run = "Autofetch", True

# 6. Execution Logic
if getattr(st.session_state, 'run', False):
    st.session_state.run = False
    with st.spinner("Agentic pipeline running..."):
        if active_mode == "Autofetch":
            ans_html, ev, opt, sar, map_h, exec_s, rep = run_place_workflow(
                place=place_input, lat=0, lon=0, start_date="", end_date="", goal="Understand scene", query=query
            )
        else:
            ans_html, ev, _, _, exec_s, rep = run_upload_workflow(
                mode=active_mode, query=query, preview1=img1, preview2=img2
            )
        st.session_state.results = (ans_html, ev, exec_s)

# 7. Results Section
if getattr(st.session_state, 'results', None):
    st.markdown("---")
    ans_html, evidence_img, trace = st.session_state.results
    
    st.markdown("### 💬 Answer")
    
    # Split into 2 columns just like the mockup (Text on left, Image on right)
    res_col1, res_col2 = st.columns([1, 1.2])
    
    with res_col1:
        st.markdown(ans_html, unsafe_allow_html=True)
        st.markdown("<br><div style='background:rgba(16,185,129,0.15); color:#10b981; padding:6px 14px; border-radius:20px; display:inline-block; font-weight:600; font-size:14px;'>Confidence Score &nbsp; 0.88</div>", unsafe_allow_html=True)
        
    with res_col2:
        st.image(evidence_img, use_container_width=True)
        
    with st.expander("🛠️ View Agent Execution Trace & Report"):
        st.json(trace)
