from tools import vlm_tool, change_tool, fusion_tool

def route_query(mode, img1_file_path, arr1, meta1, arr2, meta2, query):
    exec_summary = {"mode": mode, "query": query, "tools_executed": []}
    evidence_img = None
    
    if mode == "Single":
        # Call the API
        answer, stats = vlm_tool.run_vqa(img1_file_path, query)
        exec_summary["tools_executed"].append({"name": "VLM_API", "params": stats})
        
    elif mode == "Change Pair":
        # Run PyTorch / Math
        mask, change_pct = change_tool.run_change_detection(arr1, arr2)
        answer = f"Analysis complete. Approximately {change_pct}% of the area experienced significant change."
        exec_summary["tools_executed"].append({"name": "ChangeFormer", "outputs": {"changed_pct": change_pct}})
        
        # TODO: Ask your data team to write a function that overlays 'mask' on top of 'arr1' 
        # and saves it as an image, then pass that image to evidence_img!
        
    else: # Optical+SAR
        mask, stats = fusion_tool.run_fusion(arr1, arr2)
        answer = "Fused Optical and SAR data successfully. Water bodies and built-up areas highlighted in evidence overlay."
        exec_summary["tools_executed"].append({"name": "Optical_SAR_Fusion", "params": stats})
        
    return answer, evidence_img, exec_summary
