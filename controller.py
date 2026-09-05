from tools import vlm_tool, change_tool, fusion_tool
from io_utils import to_rgb_preview, create_overlay

def route_query(mode, img1_file_path, arr1, meta1, arr2, meta2, query):
    exec_summary = {"mode": mode, "query": query, "tools_executed": []}
    evidence_img = None
    
    if mode == "Single":
        answer, stats = vlm_tool.run_vqa(img1_file_path, query)
        exec_summary["tools_executed"].append({"name": "VLM_API", "params": stats})
        
    elif mode == "Change Pair":
        mask, change_pct = change_tool.run_change_detection(arr1, arr2)
        answer = f"Analysis complete. Approximately {change_pct}% of the area experienced significant spatial change between the two temporal captures."
        exec_summary["tools_executed"].append({"name": "ChangeFormer", "outputs": {"changed_pct": change_pct}})
        
        # Generate Red Evidence Overlay
        base_rgb = to_rgb_preview(arr1)
        evidence_img = create_overlay(base_rgb, mask, color=(255, 40, 40), alpha=0.6)
        
    else: 
        # Optical+SAR
        mask, stats = fusion_tool.run_fusion(arr1, arr2)
        answer = "Fused Optical and SAR data successfully. Water bodies and high-moisture/low-backscatter regions are highlighted in the evidence overlay."
        exec_summary["tools_executed"].append({"name": "Optical_SAR_Fusion", "params": stats})
        
        # Generate Cyan Evidence Overlay
        base_rgb = to_rgb_preview(arr1)
        evidence_img = create_overlay(base_rgb, mask, color=(40, 200, 255), alpha=0.6)
        
    return answer, evidence_img, exec_summary
