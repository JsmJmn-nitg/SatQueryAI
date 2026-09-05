import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "HuggingFaceTB/SmolVLM-Instruct"

print("Loading Agentic VLM (SmolVLM). This takes a minute...")
processor = AutoProcessor.from_pretrained(model_id)
# Loads natively in float16 to save Colab Memory
model = AutoModelForVision2Seq.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

def run_agentic_vqa(image_path, query, rs_tags=None, context_stats=None):
    """
    Synthesizes tool outputs and visual data to answer the user query.
    """
    image = Image.open(image_path).convert("RGB")
    
    # 1. Construct the Agent's reasoning prompt using the specialized tools
    prompt_text = "You are SatQuery AI, an expert remote sensing assistant.\n"
    
    if rs_tags:
        prompt_text += f"A BigEarthNet model analyzed this image and found: {', '.join(rs_tags)}.\n"
    if context_stats:
        prompt_text += f"Specialist tool analysis: {context_stats}.\n"
        
    prompt_text += f"Based on this verified data and the image, answer the user's question clearly and professionally: '{query}'"
    
    # 2. Format it into the standard Chat format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]
    
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)
    
    # 3. Generate the Answer
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256)
        
    # Trim the prompt tokens out of the output so we only get the answer
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0]
    
    return output_text.strip()
