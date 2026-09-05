from tools import change_tool, fusion_tool, vlm_tool, classifier_tool
from io_utils import save_temp_png, create_evidence_overlay

def route_query(mode, arr1, meta1, arr2, meta2, query, preview1):
    exec_summary = {"mode": mode, "query": query, "tools_executed": []}
    
    # 1. Convert numpy preview to a temp PNG for the Vision Models
    temp_img_path = save_temp_png(preview1)
    
    if mode == "Single":
        # TOOL 1: BigEarthNet Classifier (MANDATORY REQUIREMENT)
        tags = classifier_tool.get_landcover_tags(temp_img_path)
        exec_summary["tools_executed"].append({
            "name": "BigEarthNet_ResNet50_Classifier",
            "detected_tags": tags
        })
        
        # TOOL 2: Moondream VLM (Agentic synthesis)
        answer = vlm_tool.run_agentic_vqa(temp_img_path, query, rs_tags=tags)
        exec_summary["tools_executed"].append({"name": "Moondream2_VLM", "task": "VQA_Synthesis"})
        
        evidence_img = preview1

    elif mode == "Change Pair":
        # TOOL 1: Math-based Change Detector
        mask, change_pct, stats = change_tool.run_change_detection(arr1, arr2)
        exec_summary["tools_executed"].append({"name": "Pixel_Difference_Change_Extractor", "stats": stats})
        
        evidence_img = create_evidence_overlay(preview1, mask, color=(255, 50, 50), opacity=0.6)
        
        # TOOL 2: Moondream VLM (Agentic synthesis)
        answer = vlm_tool.run_agentic_vqa(temp_img_path, query, context_stats=f"Change detection tool found {change_pct}% of the area changed.")
        exec_summary["tools_executed"].append({"name": "Moondream2_VLM", "task": "Change_VQA"})

    else:  # Optical+SAR Pair
        # TOOL 1: Multi-modal Fusion Extractor
        mask, stats = fusion_tool.run_fusion(arr1, arr2)
        exec_summary["tools_executed"].append({"name": "NDWI_SAR_Fusion_Extractor", "stats": stats})
        
        evidence_img = create_evidence_overlay(preview1, mask, color=(0, 120, 255), opacity=0.55)
        
        # TOOL 2: Moondream VLM (Agentic synthesis)
        water_pct = stats["fused_water_pixels"] / mask.size * 100
        answer = vlm_tool.run_agentic_vqa(temp_img_path, query, context_stats=f"Optical+SAR fusion detected water covering {water_pct:.1f}% of the area.")
        exec_summary["tools_executed"].append({"name": "Moondream2_VLM", "task": "CrossModal_VQA"})
        
    return answer, evidence_img, exec_summary
