import gradio as gr
from io_utils.geotiff import read_geotiff, to_rgb_preview, check_pair_compatible
from controller.router import route_query

# Main function connected to Gradio UI
def run_satquery(mode, img1_file, img2_file, query):
    # 1) Read inputs
    arr1, meta1 = read_geotiff(img1_file.name)
    preview1 = to_rgb_preview(arr1)
    
    arr2 = meta2 = preview2 = None
    if img2_file is not None:
        arr2, meta2 = read_geotiff(img2_file.name)
        preview2 = to_rgb_preview(arr2)
        
        # Optional: Run compatibility check from Page 5
        # compat = check_pair_compatible(arr1, meta1, arr2, meta2)

    # 2) Route (Send to Controller)
    answer, evidence, exec_summary = route_query(mode, arr1, meta1, arr2, meta2, query)
    
    # Fallback to preview1 if no evidence overlay is generated yet
    if evidence is None:
        evidence = preview1

    return answer, evidence, preview2, exec_summary

# 3) Define the UI Layout
demo = gr.Interface(
    fn=run_satquery,
    inputs=[
        gr.Dropdown(["Single", "Change Pair", "Optical+SAR Pair"], value="Single", label="Input Mode"),
        gr.File(label="Image 1 (GeoTIFF/TIFF)"),
        gr.File(label="Image 2 (optional)"),
        gr.Textbox(label="Query", value="Describe the land-cover and major objects.")
    ],
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Image(label="Evidence / Overlay"),
        gr.Image(label="Image 2 Preview"),
        gr.JSON(label="Execution Summary")
    ],
    title="SatQuery AI (MVP)"
)

if __name__ == "__main__":
    # share=True creates a public link you can give to the hackathon judges!
    demo.launch(share=True)
