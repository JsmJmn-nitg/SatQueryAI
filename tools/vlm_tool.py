from gradio_client import Client, file
import os

def run_vqa(image_path, query):
    try:
        # Connect to a public Hugging Face Space running a VLM
        # (Replace with an actual GeoChat/RS-LLaVA space URL if you find one, 
        # or use a powerful generic open-source VLM like LLaVA as a fallback)
        client = Client("lmms-lab/llava-onevision") 
        
        # Send the image and text to the API
        result = client.predict(
            image=file(image_path),
            text=query,
            api_name="/predict"
        )
        return result, {"vlm_model": "API_Hosted_VLM", "status": "Success"}
    except Exception as e:
        return f"VLM Error: {str(e)}", {"status": "Failed"}
