import gradio as gr
from io_utils import read_geotiff, to_rgb_preview, check_pair_compatible
from controller import route_query

def run_satquery(mode, img1_file, img2_file, query):
    # 0) Safety check
    if img1_file is None:
        return "Error: Please upload at least Image 1.", None, None, {"error": "Missing input"}

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
                return "Error: Images are not compatible (different dimensions).", preview1, preview2, compat

    # 3) Route to Agentic Controller
    answer, evidence, exec_summary = route_query(mode, arr1, meta1, arr2, meta2, query)
    
    # 4) If AI tool didn't generate a custom overlay yet, fallback to Image 1 preview
    if evidence is None:
        evidence = preview1

    return answer, evidence, preview2, exec_summary

# UI Definition
demo = gr.Interface(
    fn=run_satquery,
    inputs=[
        gr.Dropdown(["Single", "Change Pair", "Optical+SAR Pair"], value="Single", label="Task Mode"),
        gr.File(label="Image 1 (GeoTIFF)"),
        gr.File(label="Image 2 (GeoTIFF - Optional)"),
        gr.Textbox(label="Query", value="Describe the land-cover and major objects.")
    ],
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Image(label="Evidence / Overlay"),
        gr.Image(label="Image 2 Preview"),
        gr.JSON(label="Execution Summary")
    ],
    title="🛰️ SatQuery AI (Hackathon MVP)",
    description="Upload a satellite image (or a pair), select a mode, and ask questions!"
)

if __name__ == "__main__":
    demo.launch(share=True)
