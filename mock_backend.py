from __future__ import annotations
import json
import os
import tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from datetime import datetime

def _rng(seed: int = 7):
    return np.random.default_rng(seed)

def make_placeholder_satellite(kind: str = "optical", H: int = 640, W: int = 960) -> np.ndarray:
    r = _rng(11 if kind == "optical" else 23 if kind == "sar" else 37)

    x = np.linspace(0, 1, W, dtype=np.float32)[None, :]
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None]

    if kind == "sar":
        base = (0.35 + 0.45*(1-x) + 0.25*y)
        noise = r.normal(0, 0.22, (H, W)).astype(np.float32)
        img = np.clip(base + noise, 0, 1)
        rgb = np.stack([img, img, img], axis=-1)
        rgb = (rgb * 255).astype(np.uint8)
        for _ in range(10):
            rr = r.integers(0, H)
            rgb[max(0, rr-1):min(H, rr+1), :, :] = np.clip(rgb[max(0, rr-1):min(H, rr+1), :, :] + r.integers(10, 28), 0, 255)
        return rgb

    g = 0.25 + 0.55*(1 - (x*0.8 + y*0.5))
    b = 0.18 + 0.50*(x*0.6 + (1-y)*0.4)
    rch = 0.15 + 0.45*(x*0.4 + y*0.35)
    base = np.stack([rch, g, b], axis=-1)

    n1 = r.normal(0, 0.08, (H, W, 1)).astype(np.float32)
    n2 = r.normal(0, 0.05, (H, W, 1)).astype(np.float32)
    base = np.clip(base + n1 + (n2*np.sin(x*10)[..., None]), 0, 1)

    rgb = (base * 255).astype(np.uint8)

    pil = Image.fromarray(rgb)
    d = ImageDraw.Draw(pil, "RGBA")
    for _ in range(10):
        x0 = int(r.integers(0, W))
        y0 = int(r.integers(0, H))
        x1 = int(np.clip(x0 + r.normal(0, W*0.25), 0, W-1))
        y1 = int(np.clip(y0 + r.normal(0, H*0.22), 0, H-1))
        col = (230, 230, 230, 90) if r.random() > 0.6 else (40, 120, 255, 70)
        width = int(r.integers(2, 5))
        d.line([(x0, y0), (x1, y1)], fill=col, width=width)

    pil = pil.filter(ImageFilter.GaussianBlur(radius=0.35))
    return np.array(pil)

def make_mask(H: int, W: int, seed: int = 5) -> np.ndarray:
    r = _rng(seed)
    mask = np.zeros((H, W), dtype=np.uint8)
    for _ in range(7):
        cx = int(r.integers(W*0.1, W*0.9))
        cy = int(r.integers(H*0.15, H*0.85))
        rx = int(r.integers(W*0.03, W*0.12))
        ry = int(r.integers(H*0.03, H*0.12))
        y, x = np.ogrid[:H, :W]
        blob = ((x - cx)**2)/(rx**2 + 1e-6) + ((y - cy)**2)/(ry**2 + 1e-6) <= 1.0
        mask[blob] = 1
    return mask

def overlay_mask(base_rgb: np.ndarray, mask: np.ndarray, color=(212, 175, 55), opacity=0.55) -> np.ndarray:
    base = base_rgb.astype(np.float32)
    out = base.copy()
    c = np.array(color, dtype=np.float32)[None, None, :]
    m = mask.astype(bool)
    out[m] = base[m]*(1-opacity) + c*(opacity)
    return np.clip(out, 0, 255).astype(np.uint8)

def render_osm_iframe(lat: float, lon: float) -> str:
    dx = 0.08
    dy = 0.05
    left = lon - dx
    right = lon + dx
    top = lat + dy
    bottom = lat - dy
    src = f"https://www.openstreetmap.org/export/embed.html?bbox={left}%2C{bottom}%2C{right}%2C{top}&layer=mapnik&marker={lat}%2C{lon}"
    return f"""
    <div class="glass soft cardPad">
      <div class="sectionTitle">Map</div>
      <div class="smallHint">Location context preview (OpenStreetMap embed). Sentinel/Google Earth imagery fetch can be wired later.</div>
      <div style="margin-top:10px; border-radius:16px; overflow:hidden; border:1px solid rgba(148,163,184,.18);">
        <iframe width="100%" height="320" frameborder="0" scrolling="no" marginheight="0" marginwidth="0" src="{src}"></iframe>
      </div>
    </div>
    """

def build_report_md(title: str, answer_md: str, exec_summary: dict) -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    return f"""# {title}

**Generated:** {ts}

---

## Answer

{answer_md}

---

## Execution Summary (Auditable Trace)

```json
{json.dumps(exec_summary, indent=2)}
```
"""

def save_report_file(report_md: str, filename: str = "satquery_report.md") -> str:
    path = os.path.join(tempfile.gettempdir(), filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_md)
    return path

def smartish_answer(query: str, mode: str, place: str | None = None) -> str:
    q = (query or "").lower()
    bullets = []

    if any(k in q for k in ["wildfire", "fire", "burn", "smoke"]):
        bullets += [
            "Possible fire-impacted zones are typically indicated by **darkened burn scars**, reduced vegetation vigor, and changes in texture.",
            "A reliable workflow uses **pre/post imagery** and vegetation indices (e.g., NDVI differencing) plus thermal anomalies if available."
        ]
    if any(k in q for k in ["river", "rivers", "stream", "canal"]):
        bullets += [
            "Counting exact rivers from a single medium-resolution image can be ambiguous due to **seasonal flow**, shadows, and braided channels.",
            "A better approach is to delineate the drainage network using **water indices** (NDWI) and/or SAR low-backscatter water masks."
        ]
    if any(k in q for k in ["flood", "water", "inundation"]):
        bullets += [
            "Flood water is best detected with **SAR** (cloud-robust) combined with optical indices when clear.",
            "We would compare **current water extent vs baseline** to estimate newly inundated areas."
        ]
    if any(k in q for k in ["urban", "settlement", "built-up", "construction"]):
        bullets += [
            "Built-up regions are characterized by **rectilinear textures** and persistent high backscatter in SAR.",
            "Change over time is confirmed via **bi-temporal comparison** and morphology of new edges/roads."
        ]
    if not bullets:
        bullets = [
            "We interpret land cover using spectral/texture cues and (when available) cross-sensor confirmation from SAR.",
            "For high-stakes decisions, we recommend validating with multi-date imagery and known ground truth layers."
        ]

    place_line = f"**Area of interest:** {place}\n\n" if place else ""
    return (
        f"{place_line}"
        f"## Result\n\n"
        f"**Query:** {query}\n\n"
        f"### What the agent would do\n"
        f"- Select tools based on intent (VQA / change / fusion)\n"
        f"- Fetch the best available scenes (cloud-filtered optical + matching SAR)\n"
        f"- Run specialist extractors (water, burn-scar, built-up, change)\n"
        f"- Return an evidence overlay + quantified summary\n\n"
        f"### Interpretation (demo)\n"
        + "\n".join([f"- {b}" for b in bullets]) +
        "\n\n"
        f"### Confidence (demo)\n"
        f"- **Medium** — this is a mocked run intended for UI/demo. Real confidence would be derived from data quality (clouds), sensor agreement, and model calibration.\n"
    )

def run_place_workflow(place: str, lat: float, lon: float, start_date: str, end_date: str, goal: str, query: str):
    planned = []
    if goal in ["Understand scene", "Custom"]:
        planned = ["Scene Understanding", "Land-cover tags (BigEarthNet-adapted)", "Report generator"]
    elif goal == "Detect change":
        planned = ["Bi-temporal scene selection", "Change map", "Change description"]
    elif goal == "Water / flooding":
        planned = ["Optical NDWI", "SAR low-backscatter water", "Fusion + inundation estimate"]
    elif goal == "Wildfire":
        planned = ["Pre/post imagery", "Vegetation loss (NDVI delta)", "Burn scar delineation"]
    elif goal == "Urban growth":
        planned = ["Built-up extraction", "Road-edge emergence", "Bi-temporal comparison"]

    exec_summary = {
        "task_router": {
            "mode": "Place Search",
            "goal": goal,
            "query": query,
        },
        "data_acquisition": {
            "status": "mock",
            "intended_sources": [
                "Sentinel-2 L2A (optical)",
                "Sentinel-1 GRD (SAR)",
                "Optional: Google Earth basemap (for context)"
            ],
            "place": place,
            "lat": lat,
            "lon": lon,
            "date_range": {"start": start_date, "end": end_date},
            "filters": {"cloud_cover_max": "20% (planned)"},
        },
        "planned_tools": planned,
        "outputs": {"evidence_overlay": True, "trace": True, "report": True},
    }

    optical = make_placeholder_satellite("optical")
    sar = make_placeholder_satellite("sar")
    mask = make_mask(optical.shape[0], optical.shape[1], seed=9 if goal == "Water / flooding" else 6)
    evidence = overlay_mask(optical, mask, color=(212, 175, 55), opacity=0.58)

    answer_md = smartish_answer(query=query, mode="Place Search", place=place)
    map_html = render_osm_iframe(lat, lon)

    report_md = build_report_md("SatQuery AI Report (Demo)", answer_md, exec_summary)
    report_path = save_report_file(report_md)

    return answer_md, evidence, optical, sar, map_html, exec_summary, report_path

def run_upload_workflow(mode: str, query: str, preview1: np.ndarray | None, preview2: np.ndarray | None):
    exec_summary = {
        "task_router": {"mode": mode, "query": query},
        "data_acquisition": {"status": "local_upload"},
        "tools_executed": [
            {"name": "InputValidator", "status": "ok"},
            {"name": "ToolPlanner", "decision": f"Mock plan for {mode}"},
            {"name": "EvidenceOverlay", "status": "ok"},
            {"name": "ReportGenerator", "status": "ok"},
        ],
        "note": "Mocked backend for demo/PPT. Replace with specialist models when ready."
    }

    if preview1 is None:
        preview1 = make_placeholder_satellite("optical")
    if preview2 is None:
        preview2 = make_placeholder_satellite("sar")

    base = preview1
    mask = make_mask(base.shape[0], base.shape[1], seed=13 if mode == "Change Pair" else 4)
    color = (255, 80, 90) if mode == "Change Pair" else (0, 160, 255) if mode == "Optical+SAR Pair" else (212, 175, 55)
    evidence = overlay_mask(base, mask, color=color, opacity=0.55)

    answer_md = smartish_answer(query=query, mode=mode, place=None)
    report_md = build_report_md(f"SatQuery AI Report (Demo) — {mode}", answer_md, exec_summary)
    report_path = save_report_file(report_md, filename="satquery_report_upload.md")

    return answer_md, evidence, preview1, preview2, exec_summary, report_path
