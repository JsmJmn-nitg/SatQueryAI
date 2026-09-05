from tools import vlm_tool, change_tool, fusion_tool
from io_utils import to_rgb_preview, create_evidence_overlay
import numpy as np

def route_query(mode, img1_file_path, arr1, meta1, arr2, meta2, query):
    exec_summary = {"mode": mode, "query": query, "tools_executed": []}
    evidence_img = None
    preview1 = to_rgb_preview(arr1)
    
    if mode == "Single":
        # Call the VLM API
        answer, stats = vlm_tool.run_vqa(img1_file_path, query)
        exec_summary["tools_executed"].append({
            "name": "VLM_API",
            "model": "LLaVA-OneVision-1.5 (Remote Sensing Adapted)",
            "params": stats,
            "note": "System prompt injected to enforce geospatial expertise"
        })
        evidence_img = preview1  # Show original image as evidence
        
    elif mode == "Change Pair":
        # Run PyTorch / Math-based change detection
        mask, change_pct, stats = change_tool.run_change_detection(arr1, arr2)
        
        # Create visual evidence overlay (red for changes)
        evidence_img = create_evidence_overlay(preview1, mask, color=(255, 50, 50), opacity=0.6)
        
        # Generate detailed answer
        answer = (
            f"**Change Detection Analysis Complete**\n\n"
            f"• Changed Area: ~{change_pct}% of total scene\n"
            f"• Changed Pixels: {stats['changed_pixels']:,} / {stats['total_pixels']:,}\n"
            f"• Threshold Used: {stats['threshold']:.2f} (90th percentile of absolute difference)\n\n"
            f"Red overlay highlights areas of significant change between the two dates. "
            f"Common causes include: construction, deforestation, flooding, or seasonal vegetation changes."
        )
        
        exec_summary["tools_executed"].append({
            "name": "ChangeFormer_Fallback",
            "logic": "Absolute Difference Thresholding (suitable for co-registered multispectral pairs)",
            "params": {
                "threshold_percentile": 90,
                "threshold_value": float(stats['threshold'])
            },
            "outputs": {
                "changed_pixels": int(stats['changed_pixels']),
                "change_percentage": change_pct,
                "evidence_overlay": "Red = Changed regions"
            }
        })
        
    else:  # Optical+SAR Pair
        mask, fusion_stats = fusion_tool.run_fusion(arr1, arr2)
        
        # Create visual evidence overlay (blue for water)
        evidence_img = create_evidence_overlay(preview1, mask, color=(0, 120, 255), opacity=0.55)
        
        # Generate detailed answer
        water_pct = (np.sum(mask) / mask.size) * 100
        answer = (
            f"**Optical + SAR Fusion Analysis Complete**\n\n"
            f"• Water Coverage: ~{water_pct:.2f}% of scene\n"
            f"• Fusion Method: {fusion_stats['fusion_method']}\n"
            f"• Blue overlay highlights detected water bodies\n\n"
            f"**Why this works:**\n"
            f"- Optical NDWI detects water via spectral signature (Green - NIR)\n"
            f"- SAR detects water via low backscatter (smooth surfaces)\n"
            f"- Combined approach is robust to clouds and illumination issues"
        )
        
        exec_summary["tools_executed"].append({
            "name": "Optical_SAR_Fusion",
            "logic": "NDWI (Optical) OR Low-Backscatter (SAR)",
            "params": {
                "ndwi_threshold": 0.1,
                "sar_percentile_threshold": 15,
                "fusion_operator": "Logical OR (union)"
            },
            "outputs": {
                "water_percentage": round(water_pct, 2),
                "evidence_overlay": "Blue = Water bodies"
            }
        })
        
    return answer, evidence_img, exec_summary
