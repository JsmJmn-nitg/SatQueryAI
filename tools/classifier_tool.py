import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

# Load globally to save time and VRAM
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoImageProcessor.from_pretrained("dima806/bigearthnet_resnet50")
model = AutoModelForImageClassification.from_pretrained("dima806/bigearthnet_resnet50").to(device)

def get_landcover_tags(image_path):
    """Runs a BigEarthNet fine-tuned model to extract high-confidence land cover tags."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        logits = model(**inputs).logits
        
    probs = torch.nn.functional.sigmoid(logits[0])
    top_probs, top_indices = torch.topk(probs, 4) # Get top 4 tags
    
    tags = [model.config.id2label[idx.item()] for idx in top_indices if probs[idx].item() > 0.2]
    return tags if tags else ["Mixed unknown cover"]
