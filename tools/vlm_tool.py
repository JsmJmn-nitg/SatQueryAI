import os
from gradio_client import Client, handle_file

def run_vqa(image_path, query):
    import traceback
    
    try:
        print("\n" + "-"*50)
        print("DEBUG: Inside run_vqa")
        print(f"Image path: {image_path}")
        print(f"Query: {query}")
        print("-"*50)
        
        print("DEBUG: Creating GradioClient...")
        client = Client("Bireswar26/GeoChat")
        print("DEBUG: Client created successfully")
        
        full_query = (
            f"You are a remote sensing expert. {query} "
            f"Focus on land cover, infrastructure, water, vegetation, and urban features."
        )
        print(f"DEBUG: Full query: {full_query[:100]}...")
        
        print(f"DEBUG: Calling client.predict...")
        print(f"DEBUG: - image path: {image_path}")
        print(f"DEBUG: - checking if file exists: {os.path.exists(image_path)}")
        
        result = client.predict(
            image=handle_file(image_path),
            prompt=full_query,
            max_new_tokens=256,
            api_name="/geochat"
        )
        
        print(f"DEBUG: Prediction successful!")
        print(f"DEBUG: Result type: {type(result)}")
        print(f"DEBUG: Result: {result[:200] if isinstance(result, str) else result}")
        print("-"*50 + "\n")
        
        return result, {
            "vlm_model": "GeoChat",
            "status": "Success",
            "system_prompt_used": True
        }
        
    except Exception as e:
        print("\n" + "!"*50)
        print("ERROR in run_vqa:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("!"*50 + "\n")
        
        fallback = (
            f"**[GeoChat API Failed]**\n\n"
            f"Error: {type(e).__name__}: {str(e)}\n\n"
            f"Mock Analysis: Scene shows mixed land cover typical of satellite imagery.\n"
        )
        
        return fallback, {
            "vlm_model": "Fallback",
            "status": f"Error: {str(e)}"
        }
