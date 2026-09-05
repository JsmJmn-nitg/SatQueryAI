from tools import vlm_tool, change_tool, fusion_tool
from io_utils import to_rgb_preview, create_evidence_overlay
import numpy as np

def route_query(mode, img1_file_path, arr1, meta1, arr2, meta2, query):
    import traceback
    
    print("\n" + "="*50)
    print("DEBUG: Inside route_query")
    print(f"Mode: {mode}")
    print(f"Image path: {img1_file_path}")
    print(f"Query: {query}")
    print("="*50 + "\n")
    
    exec_summary = {"mode": mode, "query": query, "tools_executed": []}
    evidence_img = None
    preview1 = to_rgb_preview(arr1)
                
    try:
        if mode == "Single":
            print("DEBUG: Entering Single mode - calling VLM")
            
            # --- NEW FIX: Import necessary modules and save as PNG ---
            import tempfile
            from PIL import Image
            
            # Create a temporary file path
            temp_img_path = tempfile.mktemp(suffix=".png")
            
            # Convert the RGB numpy array into an Image and save it
            Image.fromarray(preview1).save(temp_img_path)
            
            # Call the VLM API using the temporary PNG file instead of the raw .tif
            answer, stats = vlm_tool.run_vqa(temp_img_path, query)
            # ---------------------------------------------------------
            
            print(f"DEBUG: VLM returned: {answer[:100]}...")
            
            exec_summary["tools_executed"].append({
                "name": "VLM_API",
                "model": "GeoChat",
                "params": stats,
            })
            evidence_img = preview1
            
        elif mode == "Change Pair":
            print("DEBUG: Entering Change Pair mode")
            mask, change_pct, stats = change_tool.run_change_detection(arr1, arr2)
            print(f"DEBUG: Change detection complete, {change_pct}% changed")
            
            evidence_img = create_evidence_overlay(preview1, mask, color=(255, 50, 50), opacity=0.6)
            
            answer = (
                f"**Change Detection Analysis Complete**\n\n"
                f"• Changed Area: ~{change_pct}% of total scene\n"
                f"• Changed Pixels: {stats['changed_pixels']:,} / {stats['total_pixels']:,}\n"
            )
            
            exec_summary["tools_executed"].append({
                "name": "ChangeFormer_Fallback",
                "outputs": {"change_percentage": change_pct}
            })
            
        else:  # Optical+SAR Pair
            print("DEBUG: Entering Optical+SAR mode")
            mask, fusion_stats = fusion_tool.run_fusion(arr1, arr2)
            print(f"DEBUG: Fusion complete")
            
            evidence_img = create_evidence_overlay(preview1, mask, color=(0, 120, 255), opacity=0.55)
            
            water_pct = (np.sum(mask) / mask.size) * 100
            answer = f"**Optical + SAR Fusion Complete**\n\nWater Coverage: ~{water_pct:.2f}%"
            
            exec_summary["tools_executed"].append({
                "name": "Optical_SAR_Fusion",
                "outputs": {"water_percentage": round(water_pct, 2)}
            })
        
        print("DEBUG: route_query completed successfully\n")
        return answer, evidence_img, exec_summary
        
    except Exception as e:
        print("\n" + "!"*50)
        print("ERROR in route_query:")
        print(f"Error: {e}")
        traceback.print_exc()
        print("!"*50 + "\n")
        raise  # Re-raise so the outer handler catches it
