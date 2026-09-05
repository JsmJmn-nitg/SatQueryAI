import os
from gradio_client import Client, handle_file

def run_vqa(image_path, query):
    try:
        print(f"Connecting to GeoChat...")
        client = Client("Bireswar26/GeoChat")
        
        # System prompt for better context
        full_query = (
            "You are an expert Remote Sensing Analyst with specialization in satellite imagery interpretation. "
            "Analyze the provided imagery with focus on:\n"
            "- Land cover classification (vegetation, water bodies, urban fabric, bare soil)\n"
            "- Infrastructure and built environment\n"
            "- Environmental indicators (cloud cover, shadows, seasonal effects)\n"
            "- Spatial patterns and geometric features\n\n"
            "Use precise geospatial terminology. Be concise but thorough."
        )
        
        result = client.predict(
            image=handle_file(image_path),
            prompt=full_query,
            max_new_tokens=256,
            api_name="/geochat"
        )
        
        return result, {
            "vlm_model": "GeoChat",
            "status": "Success",
            "system_prompt_used": True
        }
        
    except Exception as e:
        print(f"GeoChat Error: {e}")
        
        fallback = (
            f"**[GeoChat API Failed - Using Fallback]**\n\n"
            f"Query: {query}\n\n"
            f"Mock Analysis: Scene shows mixed land cover with vegetation, "
            f"potential urban structures, and varied surface types typical of "
            f"satellite imagery. Water bodies may be present in low-reflectance regions.\n\n"
            f"*Error: {str(e)}*"
        
        return fallback, {
            "vlm_model": "Fallback",
            "status": f"Error: {str(e)}"
        }
        )
