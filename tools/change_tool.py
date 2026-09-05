import numpy as np
import torch
# import your_model_library here (e.g., from transformers import ...)

# Load model globally so it only loads into GPU memory once
device = "cuda" if torch.cuda.is_available() else "cpu"
# change_model = MyChangeFormerModel.from_pretrained("...").to(device)

def run_change_detection(arr1, arr2):
    """
    arr1, arr2: numpy arrays from your GeoTIFFs
    Returns: a binary mask (0 for no change, 1 for change)
    """
    # 1. Preprocess arrays to PyTorch Tensors
    # tensor1 = torch.from_numpy(arr1).unsqueeze(0).to(device)
    # tensor2 = torch.from_numpy(arr2).unsqueeze(0).to(device)
    
    # 2. Run Inference
    # with torch.no_grad():
    #     output_mask = change_model(tensor1, tensor2)
    
    # ---------------------------------------------------------
    # HACKATHON FALLBACK (If your model weights aren't loading):
    # Use simple absolute difference thresholding just to make the UI work!
    # ---------------------------------------------------------
    diff = np.abs(arr1.astype(np.float32) - arr2.astype(np.float32))
    mean_diff = np.mean(diff, axis=0) # Collapse bands
    
    threshold = np.percentile(mean_diff, 90) # Top 10% of changes
    binary_mask = (mean_diff > threshold).astype(np.uint8)
    
    # Calculate stats for the JSON trace
    changed_pixels = np.sum(binary_mask)
    total_pixels = binary_mask.size
    change_pct = round((changed_pixels / total_pixels) * 100, 2)
    
    return binary_mask, change_pct
