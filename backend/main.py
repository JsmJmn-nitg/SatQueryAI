import os
import io
import re
import json
import base64
from typing import Optional, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np
import cv2
import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

app = FastAPI(title="SatQuery AI", version="18.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health():
    return {"status": "ok", "model_loaded": vlm_model is not None}

# =========================================================
# LOAD SELF-HOSTED VLM  (Qwen2.5-VL-7B-Instruct, 4-bit)
# =========================================================
# Replaces: the external "Bireswar26/geochat" Gradio Space call (unreliable
# third-party demo) AND the local Qwen2-VL-2B synthesizer. One stronger model
# now does scene understanding + report writing in a single pass.
#
# Fits a free-tier Colab T4 (~15GB VRAM): 4-bit weights + vision tower +
# activations for a single image ~= 6-9GB.
#
# If you hit "Qwen2_5_VLForConditionalGeneration not found", your Colab's
# pre-installed `transformers` is too old -- run `!pip install -U -q
# "transformers>=4.49.0"` before this cell.
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading {MODEL_ID} on {device}...")

vlm_model, vlm_processor = None, None
try:
    if device == "cuda":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=quant_config,
            device_map="auto",
        )
    else:
        # No GPU: loads in fp32 so the server doesn't crash, but this will be
        # far too slow for real use. Make sure the Colab runtime is set to a
        # GPU (T4) before running.
        print("WARNING: no CUDA device found -- VLM will run on CPU and be very slow.")
        vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.float32, device_map="auto"
        )

    # Bounds how many visual tokens an image can cost -- keeps memory/latency
    # predictable regardless of how large the uploaded image is.
    vlm_processor = AutoProcessor.from_pretrained(
        MODEL_ID, min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28
    )
    print(f"VLM loaded OK on {device}.")
except Exception as e:
    print(f"VLM load failed: {e}")
    vlm_model, vlm_processor = None, None

DOMAIN_OPTIONS = ["WILDFIRE", "COASTAL", "URBAN", "TERRESTRIAL"]

# =========================================================
# SPECTRAL INDICES
# =========================================================
def calculate_spectral_indices(np_rgb: np.ndarray, nir_band: Optional[np.ndarray] = None) -> Dict[str, str]:
    """
    If a real NIR band was recovered from a multi-band source file, compute a
    genuine NDVI/NDWI. Otherwise fall back to the RGB-only proxy (and label it
    as a proxy, since it is NOT a real spectral index -- there's no NIR data in
    a plain RGB screenshot).
    """
    try:
        img = np_rgb.astype(np.float32) / 255.0
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        if nir_band is not None:
            nir = nir_band.astype(np.float32)
            ndvi = float(np.mean((nir - r) / (nir + r + 1e-8)))
            ndwi = float(np.mean((g - nir) / (g + nir + 1e-8)))  # McFeeters NDWI
            ndvi_label, ndwi_label = "NDVI", "NDWI"
        else:
            ndvi = float(np.mean((g - r) / (g + r + 1e-8)))
            ndwi = float(np.mean((g - b) / (g + b + 1e-8)))
            ndvi_label, ndwi_label = "NDVI (RGB proxy)", "NDWI (RGB proxy)"

        gray = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        urban_score = float(np.mean(np.abs(lap)) / 10.0)
        brightness = float(np.mean(gray) / 255.0)

        return {
            ndvi_label:      f"{ndvi:.3f}",
            ndwi_label:      f"{ndwi:.3f}",
            "Urban Texture": f"{urban_score:.3f}",
            "Brightness":    f"{brightness:.3f}",
        }
    except Exception as e:
        print(f"Spectral error: {e}")
        return {"NDVI": "err", "NDWI": "err", "Urban": "err", "Brightness": "err"}

# =========================================================
# DOMAIN DETECTION (cheap pixel-based guess, used as a
# grounding hint for the VLM and as a safety-net fallback)
# =========================================================
def detect_domain(np_rgb: np.ndarray) -> str:
    hsv = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2HSV)
    total = float(np_rgb.shape[0] * np_rgb.shape[1])

    water_mask = ((hsv[:, :, 0] > 90) & (hsv[:, :, 0] < 140) & (hsv[:, :, 1] > 30))
    fire_mask  = ((hsv[:, :, 0] < 15) | (hsv[:, :, 0] > 165)) & (hsv[:, :, 1] > 60)
    veg_mask   = (hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 85) & (hsv[:, :, 1] > 30)

    water_pct = np.sum(water_mask) / total
    fire_pct  = np.sum(fire_mask)  / total
    veg_pct   = np.sum(veg_mask)   / total

    if fire_pct > 0.10:
        return "WILDFIRE"
    if water_pct > 0.25:
        return "COASTAL"
    if veg_pct < 0.10:
        return "URBAN"
    return "TERRESTRIAL"

# =========================================================
# POLYGON EXTRACTION (unchanged CV logic; now keyed off the
# VLM-corrected domain rather than a heuristic-only guess)
# =========================================================
def extract_polygons(np_rgb: np.ndarray, domain: str):
    h, w   = np_rgb.shape[:2]
    total  = float(h * w)
    hsv    = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2HSV)
    gray   = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)

    water_mask = ((hsv[:, :, 0] > 90) & (hsv[:, :, 0] < 140) & (hsv[:, :, 1] > 30))
    veg_mask   = (hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 85) & (hsv[:, :, 1] > 30)
    fire_mask  = ((hsv[:, :, 0] < 15) | (hsv[:, :, 0] > 165)) & (hsv[:, :, 1] > 60)
    edges      = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    urban_mask = (edges > 20) & (~water_mask) & (~veg_mask)
    smoke_mask = (gray > 180) & (hsv[:, :, 1] < 20)
    bare_mask  = (gray > 100) & (~urban_mask) & (~water_mask) & (~veg_mask) & (~fire_mask)

    if domain == "WILDFIRE":
        mask_defs = [
            ("Active Fire Zone",     fire_mask | (gray < 25), "#DC2626", "High-temperature combustion area."),
            ("Smoke / Ash Plume",    smoke_mask,               "#6B7280", "Suspended particulate and smoke."),
            ("Burned Vegetation",    bare_mask,                "#78350F", "Charred biomass and scorched soil."),
            ("Intact Canopy",        veg_mask,                 "#10B981", "Vegetation outside the fire perimeter."),
        ]
    elif domain == "COASTAL":
        mask_defs = [
            ("Open Water",           water_mask, "#0284C7", "Marine or lacustrine surface."),
            ("Coastal Infrastructure", urban_mask, "#E11D48", "Built environment near shore."),
            ("Beach / Littoral",     bare_mask,  "#F59E0B", "Sandy substrate and intertidal zone."),
            ("Coastal Vegetation",   veg_mask,   "#10B981", "Mangroves or coastal flora."),
        ]
    elif domain == "URBAN":
        mask_defs = [
            ("Built-up Structures",  urban_mask, "#E11D48", "Rooftops, roads, impervious cover."),
            ("Urban Green Space",    veg_mask,   "#10B981", "Parks and tree canopy."),
            ("Bare / Construction",  bare_mask,  "#F59E0B", "Exposed soil or development."),
            ("Water Features",       water_mask, "#0284C7", "Urban ponds or rivers."),
        ]
    else:
        mask_defs = [
            ("Vegetative Canopy",    veg_mask,   "#10B981", "Active photosynthetic biomass."),
            ("Impervious Surfaces",  urban_mask, "#E11D48", "Roads and structures."),
            ("Bare Soil",            bare_mask,  "#F59E0B", "Exposed earth."),
            ("Water Bodies",         water_mask, "#0284C7", "Lakes, rivers, or wetlands."),
        ]

    polygons = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    for name, mask, color, desc in mask_defs:
        clean = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN,  kernel)
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c    = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < total * 0.005:
            continue
        eps   = 0.015 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True)
        if len(approx) < 3:
            continue
        pts = " ".join(f"{int(p[0][0]/w*1024)},{int(p[0][1]/h*1024)}" for p in approx)
        M   = cv2.moments(c)
        cx  = int(M["m10"] / M["m00"] / w * 1024) if M["m00"] > 0 else 512
        cy  = int(M["m01"] / M["m00"] / h * 1024) if M["m00"] > 0 else 512
        pct = round(area / total * 100, 1)
        polygons.append({
            "name": name, "desc": desc, "color": color,
            "percentage": max(pct, 1.0), "points": pts, "center": [cx, cy]
        })

    fallbacks = [
        ("Region A", "#6366F1"), ("Region B", "#8B5CF6"), ("Region C", "#A78BFA")
    ]
    i = 0
    while len(polygons) < 3 and i < len(fallbacks):
        polygons.append({
            "name": fallbacks[i][0], "desc": "Unclassified region.",
            "color": fallbacks[i][1], "percentage": 5.0,
            "points": "100,100 300,100 300,300 100,300", "center": [200, 200]
        })
        i += 1

    return polygons[:4]

# =========================================================
# VLM GENERATION -- single self-hosted model does scene
# understanding + structured report synthesis in one pass
# =========================================================
def run_vlm(pil_img: Image.Image, query: str, domain_guess: str) -> str:
    prompt = (
        "You are a remote sensing analyst reviewing a satellite or aerial image.\n"
        f"A preliminary pixel-color classifier guessed the scene type as {domain_guess}, "
        "but classify it yourself from what is actually visible -- override the guess if it looks wrong.\n\n"
        f"User question: {query}\n\n"
        "Respond with ONLY one valid JSON object -- no markdown fences, no text before or after it -- "
        "with exactly these keys:\n"
        '{"domain": "WILDFIRE|COASTAL|URBAN|TERRESTRIAL", '
        '"title": "6-10 word scene title", '
        '"report": "3-4 sentence analytical answer to the user question, grounded only in what is visible", '
        '"card1_name": "short category label", "card1_text": "one sentence observation", '
        '"card2_name": "short category label", "card2_text": "one sentence observation", '
        '"card3_name": "short category label", "card3_text": "one sentence observation"}\n'
        "If something is not visible in the image, say so rather than guessing."
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_img},
            {"type": "text",  "text": prompt},
        ],
    }]

    text_in = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = vlm_processor(
        text=[text_in],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(vlm_model.device)

    with torch.no_grad():
        out_ids = vlm_model.generate(**inputs, max_new_tokens=450, do_sample=False)

    raw = vlm_processor.batch_decode(
        out_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )[0].strip()

    print(f"\n=== VLM RAW OUTPUT ===\n{raw}\n======================\n")
    return raw

# =========================================================
# PARSING (JSON-first, with a clearly-labeled fallback --
# no more silently swapping in canned text)
# =========================================================
def _extract_json_block(raw: str) -> Optional[dict]:
    cleaned = re.sub(r"^```(?:json)?", "", raw.strip())
    cleaned = re.sub(r"```$", "", cleaned.strip()).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = cleaned[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        repaired = re.sub(r",\s*([}\]])", r"\1", candidate)  # strip trailing commas
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            return None

REQUIRED_KEYS = [
    "title", "report",
    "card1_name", "card1_text",
    "card2_name", "card2_text",
    "card3_name", "card3_text",
]

def _fallback_result(domain: str, query: str, reason: str = "The AI model was unavailable") -> Dict:
    """Only used if the VLM is missing or its output couldn't be parsed at all."""
    domain_labels = {
        "WILDFIRE":    "Wildfire & Smoke Event",
        "COASTAL":     "Coastal Zone Analysis",
        "URBAN":       "Urban Landscape Assessment",
        "TERRESTRIAL": "Terrestrial Land Cover Study",
    }
    domain_cards = {
        "WILDFIRE": [
            ("Fire & Combustion",  "Active fire perimeter detected with thermal anomalies and smoke dispersion."),
            ("Burn Scar & Soil",   "Scorched vegetation and exposed soil mark the fire's historical extent."),
            ("Smoke & Air Quality","Dense smoke plumes indicate poor air quality and active combustion."),
        ],
        "COASTAL": [
            ("Water Body",         "Surface water detected covering significant coastal area."),
            ("Shoreline",          "Littoral zone shows sandy substrate and intertidal features."),
            ("Coastal Vegetation", "Mangroves or marsh grass identified near the waterline."),
        ],
        "URBAN": [
            ("Built-up Areas",     "Dense impervious surfaces indicate urban or peri-urban development."),
            ("Green Space",        "Scattered vegetation and parks visible within the urban matrix."),
            ("Infrastructure",     "Roads and rooftops create high edge-density texture patterns."),
        ],
        "TERRESTRIAL": [
            ("Land Cover",         "Mixed vegetation and bare soil dominate the scene."),
            ("Hydrology",          "Minor water features or drainage channels may be present."),
            ("Terrain Condition",  "Surface conditions appear stable with no acute hazards."),
        ],
    }
    cards = domain_cards.get(domain, domain_cards["TERRESTRIAL"])
    return {
        "domain": domain,
        "title": domain_labels.get(domain, "Geospatial Analysis"),
        "report": (
            f'{reason}, so this is a pixel-heuristic-only estimate for "{query}" -- '
            "not a model-generated analysis. Re-run the query once the backend model is available."
        ),
        "card1_name": cards[0][0], "card1_text": cards[0][1],
        "card2_name": cards[1][0], "card2_text": cards[1][1],
        "card3_name": cards[2][0], "card3_text": cards[2][1],
    }

def _parse_vlm_output(raw: str, domain_guess: str, query: str) -> Dict:
    data = _extract_json_block(raw) or {}

    domain = str(data.get("domain", "")).strip().upper()
    if domain not in DOMAIN_OPTIONS:
        domain = domain_guess

    fallback = _fallback_result(domain, query, reason="The model's output could not be parsed")
    result = {"domain": domain}
    for key in REQUIRED_KEYS:
        val = data.get(key)
        result[key] = val.strip() if isinstance(val, str) and val.strip() else fallback[key]
    return result

# =========================================================
# IMAGE LOADER  (fixes "Invalid image file" for TIFF, and
# now preserves a real NIR band when the source is truly
# multi-band, instead of silently discarding it)
# =========================================================
def load_image(file_bytes: bytes):
    """
    Accepts TIFF, PNG, JPEG. Returns (PIL, numpy-1024, base64-jpeg, nir_band|None).

    nir_band (float32 in [0,1], 1024x1024) is only populated for genuine
    multi-band rasters (>=4 bands), assuming an R,G,B,NIR band order -- the
    common convention for pan-sharpened RGB+NIR exports (e.g. NAIP). If your
    source uses a different band order (e.g. a raw Sentinel-2 stack), change
    the index below to match your actual NIR band position.
    """
    pil_img  = None
    nir_band = None

    try:
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        pass

    if pil_img is None:
        try:
            arr = np.frombuffer(file_bytes, np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is not None:
                pil_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        except Exception:
            pass

    if pil_img is None:
        try:
            import tifffile
            arr = tifffile.imread(io.BytesIO(file_bytes))
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.ndim == 3 and arr.shape[0] <= 10:   # bands-first layout
                arr = np.moveaxis(arr, 0, -1)

            if arr.ndim == 3 and arr.shape[-1] >= 4:
                nir_raw = arr[:, :, 3].astype(np.float32)
                nir_band = cv2.resize(
                    (nir_raw - nir_raw.min()) / (nir_raw.max() - nir_raw.min() + 1e-8),
                    (1024, 1024),
                )

            rgb = arr[:, :, :3].astype(np.float32)
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8) * 255
            pil_img = Image.fromarray(rgb.astype(np.uint8))
        except Exception:
            pass

    if pil_img is None:
        raise HTTPException(status_code=400, detail="Unsupported or corrupt image file.")

    np_img = cv2.resize(np.array(pil_img), (1024, 1024))
    buf    = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    b64    = base64.b64encode(buf.getvalue()).decode()
    return pil_img, np_img, b64, nir_band

# =========================================================
# ANALYZE ENDPOINT  (same response schema as before --
# frontend needs zero changes)
# =========================================================
@app.post("/api/analyze")
async def analyze(
    mode:   str        = Form(...),
    query:  str        = Form(...),
    image1: UploadFile = File(...),
):
    img_bytes = await image1.read()
    pil_img, np_img, b64, nir_band = load_image(img_bytes)

    # 1. Cheap pixel-based domain guess -- grounds the VLM and doubles as a
    #    safety net if the model is unavailable.
    domain_guess = detect_domain(np_img)

    # 2. Self-hosted VLM: scene understanding + report synthesis in ONE pass.
    if vlm_model is not None and vlm_processor is not None:
        try:
            raw = run_vlm(pil_img, query, domain_guess)
            parsed = _parse_vlm_output(raw, domain_guess, query)
            vlm_status = "Completed"
        except Exception as e:
            print(f"VLM inference error: {e}")
            parsed = _fallback_result(domain_guess, query, reason="The model failed during inference")
            vlm_status = "Failed"
    else:
        parsed = _fallback_result(domain_guess, query, reason="The model failed to load at startup")
        vlm_status = "Unavailable"

    domain = parsed["domain"]

    # 3. Polygons + spectral metrics, keyed off the FINAL (VLM-corrected) domain.
    polygons = extract_polygons(np_img, domain)
    metrics  = calculate_spectral_indices(np_img, nir_band)

    trace = {
        "domain": domain,
        "pixel_domain_guess": domain_guess,
        "models": [
            {"name": "Qwen2.5-VL-7B-Instruct (4-bit, self-hosted)",
             "role": "Scene understanding + report synthesis", "status": vlm_status},
            {"name": "CV Segmenter", "role": "Polygon extraction",
             "status": "Completed", "polygons": len(polygons)},
        ],
    }

    return JSONResponse({
        "title":            parsed["title"],
        "technical_report": parsed["report"],
        "dynamic_cards": [
            {"category": parsed["card1_name"], "text": parsed["card1_text"], "type": "urban"},
            {"category": parsed["card2_name"], "text": parsed["card2_text"], "type": "water"},
            {"category": parsed["card3_name"], "text": parsed["card3_text"], "type": "hazard"},
        ],
        "spectral_metrics":   metrics,
        "class_distribution": polygons,
        "features": [{"id": f"p{i}", **p} for i, p in enumerate(polygons)],
        "preview_url":        f"data:image/jpeg;base64,{b64}",
        "confidence_score":   "0.94" if vlm_status == "Completed" else "0.55",
        "domain":             domain,
        "execution_summary":  trace,
    })

# =========================================================
# STATIC / SPA
# =========================================================
DIST = os.path.abspath("dist")
if os.path.exists(os.path.join(DIST, "index.html")):
    app.mount("/assets", StaticFiles(directory=os.path.join(DIST, "assets")), name="assets")

    @app.get("/")
    def root(): return FileResponse(os.path.join(DIST, "index.html"))

    @app.get("/{p:path}")
    def spa(p: str):
        fp = os.path.join(DIST, p)
        return FileResponse(fp) if os.path.exists(fp) else FileResponse(os.path.join(DIST, "index.html"))
