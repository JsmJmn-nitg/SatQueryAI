from gradio_client import Client, file
import os

def run_vqa(image_path, query):
    try:
        # Connect to a public Hugging Face Space running a VLM
        client = Client("lmms-lab/llava-onevision") 
        
        # Send the image and text to the API
        result = client.predict(
            image=file(image_path),
            text=query,
            api_name="/predict"
        )
        return result, {"vlm_model": "API_Hosted_VLM", "status": "Success"}
    except Exception as e:
        return f"System Integrity Exception (VLM Failed): {str(e)}", {"status": "Failed", "error": str(e)}
