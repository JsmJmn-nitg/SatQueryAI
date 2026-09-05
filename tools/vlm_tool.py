import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "vikhyatk/moondream2"

print("Loading Agentic VLM (Moondream2). This takes a minute on first run...")
tokenizer = AutoTokenizer.from_pretrained(model_id, revision="2024-08-26")
# Load in float16 for Colab memory safety
moondream = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, torch_dtype=torch.float16, revision="2024-08-26"
).to(device)

def run_agentic_vqa(image_path, query, rs_tags=None, context_stats=None):
    """
    Synthesizes tool outputs and visual data to answer the user query.
    """
    image = Image.open(image_path).convert("RGB")
    enc_image = moondream.encode_image(image)
    
    # Construct the Agent Prompt
    prompt = "You are SatQuery AI, an expert remote sensing assistant. "
    
    if rs_tags:
        prompt += f"A BigEarthNet model analyzed this image and found these features: {', '.join(rs_tags)}. "
    if context_stats:
        prompt += f"Specialist tool analysis: {context_stats}. "
        
    prompt += f"\nBased on this data and the image, answer the user's question clearly and professionally: '{query}'"
    
    answer = moondream.answer_question(enc_image, prompt, tokenizer)
    return answer
