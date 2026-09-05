import os
from gradio_client import Client, file


def run_vqa(image_path, query):
    try:
        hf_token = os.getenv("HF_TOKEN")

        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN environment variable is not set. "
                "Run: export HF_TOKEN='hf_...'"
            )

        # System prompt to enforce remote sensing expertise
        system_prompt = (
            "You are an expert Remote Sensing Analyst with specialization in satellite imagery interpretation. "
            "Analyze the provided imagery with focus on:\n"
            "- Land cover classification (vegetation, water bodies, urban fabric, bare soil)\n"
            "- Infrastructure and built environment\n"
            "- Environmental indicators (cloud cover, shadows, seasonal effects)\n"
            "- Spatial patterns and geometric features\n\n"
            "Use precise geospatial terminology. Be concise but thorough."
        )

        client = Client(
            "lmms-lab/LLaVA-OneVision-1.5",
            hf_token=hf_token
        )

        # Inject system prompt before user query
        full_query = f"{system_prompt}\n\n**User Question:** {query}"

        result = client.predict(
            image=file(image_path),
            text=full_query,
            api_name="/predict"
        )

        return result, {
            "vlm_model": "LLaVA-OneVision-1.5",
            "status": "Success",
            "system_prompt_used": True
        }

    except Exception as e:
        # HACKATHON FALLBACK: If the API is down, gated, or paused,
        # return a realistic dummy string so the demo doesn't crash!
        print(f"Warning: VLM API failed ({e}). Using fallback.")

        fallback_answer = (
            f"**[Mock VLM Response - API Unavailable]**\n\n"
            f"Based on satellite imagery analysis:\n\n"
            f"**Query:** {query}\n\n"
            f"**Observations:**\n"
            f"- The scene displays heterogeneous land cover with visible vegetation patches (indicated by spectral reflectance patterns)\n"
            f"- Built-up areas detected in the central region with characteristic rectangular geometry\n"
            f"- Water bodies potentially present in darker low-reflectance zones\n"
            f"- Cloud cover appears minimal based on visual inspection\n\n"
            f"*Note: This is a demonstration fallback. In production, a fine-tuned RS-VLM (GeoChat/RS-LLaVA) would provide domain-specific analysis.*"
        )

        return fallback_answer, {
            "vlm_model": "Fallback_Mock",
            "status": "API_Failed_Using_Mock",
            "system_prompt_used": False
        }
