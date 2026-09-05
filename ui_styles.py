# ===== ./ui_styles.py =====
import gradio as gr

THEME = gr.themes.Default().set(
    body_background_fill="var(--bg-main)",
    body_background_fill_dark="var(--bg-main)",
    block_background_fill="transparent",
    block_border_width="0px",
    color_accent_soft="transparent",
)

CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Base Variables for Light/Dark Theme */
:root {
    --font-primary: 'Inter', sans-serif;
    
    /* Light Mode Variables */
    --bg-main: #f8f9fc;
    --bg-sidebar: #ffffff;
    --bg-card: #ffffff;
    --bg-card-hover: #f1f3f9;
    --bg-input: #ffffff;
    --border-color: #e2e8f0;
    --text-main: #0f172a;
    --text-muted: #64748b;
    --primary: #7c3aed; /* Purple */
    --primary-hover: #6d28d9;
    --primary-gradient: linear-gradient(135deg, #8b5cf6, #6d28d9);
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.05);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.05);
    --shadow-lg: 0 12px 24px rgba(0,0,0,0.08);
    
    --tag-red: #ef4444;
    --tag-blue: #3b82f6;
    --tag-green: #10b981;
    --tag-yellow: #f59e0b;
    --tag-purple: #8b5cf6;
}

.dark {
    /* Dark Mode Variables */
    --bg-main: #0b0e17; /* Deep Navy */
    --bg-sidebar: #0b0e17;
    --bg-card: #15192b;
    --bg-card-hover: #1e243d;
    --bg-input: #101322;
    --border-color: rgba(255, 255, 255, 0.08);
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
    --shadow-md: 0 8px 32px rgba(0,0,0,0.4);
    --shadow-lg: 0 16px 48px rgba(0,0,0,0.6);
}

/* Hard reset Gradio defaults */
body, .gradio-container {
    background-color: var(--bg-main) !important;
    font-family: var(--font-primary) !important;
    color: var(--text-main) !important;
    padding: 0 !important;
    margin: 0 !important;
    max-width: 100% !important;
}

#app-container {
    display: flex;
    height: 100vh;
    width: 100vw;
    overflow: hidden;
}

/* ================= SIDEBAR ================= */
#sidebar-col {
    width: 260px;
    min-width: 260px;
    background-color: var(--bg-sidebar);
    border-right: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
    padding: 24px 20px;
    z-index: 10;
}

.brand { display: flex; align-items: center; gap: 12px; margin-bottom: 40px; }
.brand-icon { width: 36px; height: 36px; background: var(--primary-gradient); border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;}
.brand-text h1 { font-size: 18px; font-weight: 700; margin: 0; line-height: 1.2; color: var(--text-main); }
.brand-text p { font-size: 11px; color: var(--text-muted); margin: 0; }

.nav-btn {
    width: 100%; text-align: left; padding: 12px 16px; border-radius: 12px;
    font-size: 14px; font-weight: 500; cursor: pointer; border: none; margin-bottom: 8px;
    display: flex; align-items: center; gap: 12px; transition: all 0.2s;
}
.btn-new { background: var(--primary-gradient) !important; color: white !important; box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3); }
.btn-nav { background: transparent !important; color: var(--text-main) !important; border: 1px solid transparent !important; }
.btn-nav:hover { background: var(--bg-card-hover) !important; }
.btn-nav.active { background: var(--bg-card) !important; border-color: var(--border-color) !important; box-shadow: var(--shadow-sm); }

.sidebar-bottom { margin-top: auto; display: flex; flex-direction: column; gap: 16px; }
.status-card { border: 1px solid var(--border-color); border-radius: 12px; padding: 12px 16px; display: flex; align-items: center; gap: 10px; font-size: 12px; background: var(--bg-card); }
.status-dot { width: 8px; height: 8px; background: var(--tag-green); border-radius: 50%; }

/* ================= MAIN CONTENT ================= */
#main-col {
    flex: 1;
    overflow-y: auto;
    position: relative;
    padding: 0;
}

/* Background Art for Top Area */
.hero-bg {
    position: absolute; top: 0; right: 0; width: 600px; height: 400px;
    background-image: radial-gradient(circle at 70% 30%, rgba(124, 58, 237, 0.08) 0%, transparent 60%);
    pointer-events: none; z-index: 0;
}
.dark .hero-bg { background-image: radial-gradient(circle at 70% 30%, rgba(124, 58, 237, 0.15) 0%, transparent 60%); }
.planet-orb {
    position: absolute; right: 80px; top: 80px; width: 40px; height: 40px;
    background: linear-gradient(135deg, #a78bfa, #5b21b6); border-radius: 50%;
    box-shadow: 0 0 40px rgba(139, 92, 246, 0.5), inset -5px -5px 15px rgba(0,0,0,0.3);
}

.content-wrapper { max-width: 1100px; margin: 0 auto; padding: 32px 40px 60px; position: relative; z-index: 1; }

/* Header */
.top-nav { display: flex; justify-content: space-between; align-items: center; margin-bottom: 40px; padding: 20px 40px; }
.top-nav-right { display: flex; gap: 16px; align-items: center; color: var(--text-muted); font-size: 14px; }
.user-avatar { width: 32px; height: 32px; border-radius: 50%; background: var(--primary); color: white; display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: bold;}

/* Query Section */
.greeting { font-size: 28px; font-weight: 600; margin-bottom: 24px; color: var(--text-main); display: flex; align-items: center; gap: 12px;}
.greeting span { font-size: 16px; color: var(--text-muted); font-weight: 400; }

.search-box {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 8px 8px 8px 20px;
    display: flex;
    align-items: center;
    box-shadow: var(--shadow-md);
    margin-bottom: 24px;
}
/* Override Gradio Textbox inside search-box */
.search-box .gradio-textbox { border: none !important; background: transparent !important; box-shadow: none !important; }
.search-box textarea { background: transparent !important; border: none !important; color: var(--text-main) !important; font-size: 15px !important; resize: none; padding-top: 12px; }
.search-box textarea:focus { box-shadow: none !important; }

.send-btn-wrap button {
    background: var(--primary) !important; color: white !important; border-radius: 12px !important;
    width: 44px !important; height: 44px !important; min-width: 44px !important; border: none !important;
}

/* Mode Selectors */
.mode-tabs { display: flex; gap: 12px; margin-bottom: 24px; }
.mode-btn {
    background: var(--bg-card) !important; border: 1px solid var(--border-color) !important;
    color: var(--text-muted) !important; border-radius: 12px !important; padding: 10px 20px !important;
    font-size: 14px !important; transition: all 0.2s;
}
.mode-btn:hover { background: var(--bg-card-hover) !important; color: var(--text-main) !important; }
.mode-btn.selected { border-color: var(--primary) !important; color: var(--primary) !important; background: rgba(124, 58, 237, 0.05) !important; }

/* Dynamic Content Area (Upload vs Autofetch) */
.dynamic-area {
    background: var(--bg-card); border: 1px solid var(--border-color);
    border-radius: 16px; padding: 24px; margin-bottom: 40px; box-shadow: var(--shadow-md);
}
.autofetch-ui h3 { display: flex; align-items: center; gap: 8px; color: var(--primary); font-size: 15px; margin: 0 0 8px 0; }
.autofetch-ui p { color: var(--text-muted); font-size: 14px; margin: 0; }

.upload-grid { display: flex; gap: 20px; }
.upload-box {
    border: 1px dashed var(--border-color); border-radius: 12px; padding: 30px;
    text-align: center; color: var(--text-muted); background: var(--bg-card-hover);
    flex: 1; cursor: pointer; transition: all 0.2s;
}
.upload-box:hover { border-color: var(--primary); background: rgba(124, 58, 237, 0.05); }

/* Override Gradio Image Upload */
.upload-grid .gradio-image { border: none !important; background: transparent !important; }

/* ================= RESULTS SECTION ================= */
.results-card {
    background: var(--bg-card); border: 1px solid var(--border-color);
    border-radius: 20px; padding: 32px; box-shadow: var(--shadow-lg);
    position: relative;
}

.answer-badge {
    position: absolute; top: -14px; left: 32px;
    background: rgba(124, 58, 237, 0.1); color: var(--primary); border: 1px solid rgba(124, 58, 237, 0.2);
    padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; display: flex; align-items: center; gap: 6px;
    backdrop-filter: blur(8px);
}
.dark .answer-badge { background: rgba(124, 58, 237, 0.2); }

.results-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 40px; }

/* Left text side formatted by HTML output */
.result-text-area h2 { font-size: 20px; margin-top: 0; margin-bottom: 12px; color: var(--text-main); }
.result-text-area > p { font-size: 14px; color: var(--text-muted); line-height: 1.6; margin-bottom: 24px; }

/* Custom Legend List */
.icon-list { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 16px; }
.icon-list-item { display: flex; gap: 16px; align-items: flex-start; }
.icon-circle {
    width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; font-size: 16px; color: white;
}
.ic-red { background: var(--tag-red); }
.ic-blue { background: var(--tag-blue); }
.ic-green { background: var(--tag-green); }
.ic-yellow { background: var(--tag-yellow); }
.ic-purple { background: var(--tag-purple); }

.item-text h4 { margin: 0 0 4px 0; font-size: 14px; color: var(--text-main); }
.item-text p { margin: 0; font-size: 13px; color: var(--text-muted); line-height: 1.4; }

.confidence { margin-top: 30px; display: flex; align-items: center; gap: 12px; font-size: 14px; color: var(--text-main); }
.conf-score { background: rgba(16, 185, 129, 0.15); color: var(--tag-green); padding: 4px 12px; border-radius: 12px; font-weight: 600; }

/* Right Image side */
.result-image-area { position: relative; }
.result-image-area img { border-radius: 12px; width: 100%; border: 1px solid var(--border-color); }
.image-controls { position: absolute; top: 12px; right: 12px; display: flex; gap: 8px; }
.img-btn { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 8px; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; color: var(--text-muted); cursor: pointer; box-shadow: var(--shadow-sm); }

/* Legend Box over image */
.legend-box {
    position: absolute; right: -20px; top: 60px; background: var(--bg-card);
    border: 1px solid var(--border-color); border-radius: 12px; padding: 16px;
    box-shadow: var(--shadow-lg); width: 160px;
}
.legend-box h5 { margin: 0 0 12px 0; font-size: 12px; color: var(--text-muted); font-weight: 500;}
.legend-item { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; font-size: 12px; color: var(--text-main); }
.legend-color { width: 12px; height: 12px; border-radius: 2px; }

/* Utility */
.hidden { display: none !important; }
"""
