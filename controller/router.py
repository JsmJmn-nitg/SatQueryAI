from tools import vlm_tool, change_tool, fusion_tool

def route_query(mode, img1_arr, img1_meta, img2_arr, img2_meta, query):
    exec_summary = {"mode": mode, "query": query, "tools_used": []}
    
    if mode == "Single":
        answer, evidence, exec_summary = vlm_tool.run(img1_arr, query, exec_summary)
    elif mode == "Change Pair":
        answer, evidence, exec_summary = change_tool.run(img1_arr, img2_arr, exec_summary)
    else: # "Optical+SAR Pair"
        answer, evidence, exec_summary = fusion_tool.run(img1_arr, img2_arr, exec_summary)
        
    return answer, evidence, exec_summary
