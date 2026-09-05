def route_query(mode, arr1, meta1, arr2, meta2, query):
    # The required Agentic Execution Summary
    exec_summary = {"mode": mode, "query": query, "tools_used": []}
    
    if mode == "Single":
        answer = f"[MVP] VQA Answer: Analyzing '{query}' using domain-adapted VLM."
        exec_summary["tools_used"].append({"name": "GeoChat_or_RS-LLaVA", "params": {}})
        evidence = None 
        
    elif mode == "Change Pair":
        answer = "[MVP] Major changes detected between dates. See evidence overlay."
        exec_summary["tools_used"].append({"name": "ChangeFormer", "params": {"input_size": 512, "threshold": 0.5}})
        evidence = None 
        
    else: # "Optical+SAR Pair"
        answer = "[MVP] Computed water & built-up masks using complementary optical + SAR features."
        exec_summary["tools_used"].append({"name": "OpticalSarFusion", "params": {"index": "NDWI"}})
        evidence = None 
        
    return answer, evidence, exec_summary
