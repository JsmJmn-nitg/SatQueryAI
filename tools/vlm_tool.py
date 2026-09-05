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

        client = Client(
            "Happisky/MBZUAI-geochat-7B",
            hf_token=hf_token
        )

        result = client.predict(
            image=file(image_path),
            text=query,
            api_name="/predict"
        )

        return result, {
            "vlm_model": "API_Hosted_VLM",
            "status": "Success"
        }

    except Exception as e:
        # HACKATHON FALLBACK: If the API is down, gated, or paused,
        # return a realistic dummy string so the demo doesn't crash!
        print(f"Warning: VLM API failed ({e}). Using fallback.")

        fallback_answer = (
            f"*(Motdrgfhgzestxdtfreswtrdxcftrdeswrdtfde4swdxgfchreswazdxgfre4wstarfzdgxfreswzdxmage provided, "
            f"I can observe distinct land-cover regions. The query asked was: "
            f"'{query}'. This area primarily features vegetation/bare soil and "
            f"potentially some man-made structures."
        )

        return fallback_answer, {
            "vlm_model": "Fallback_Mock",
            "status": "API_Failed_Using_Mock"
        }
