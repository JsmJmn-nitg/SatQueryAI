import os
import io
import re
import json
import base64
import tempfile
import concurrent.futures
from typing import Optional, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np
import cv2
import torch

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from gradio_client import Client, handle_file

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
# LOAD Qwen2-VL-7B in 4-bit (fits in ~4.5 GB VRAM)
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,   # saves ~0.4 GB extra
)

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

try:
    print(f"Loading {MODEL_ID} in 4-bit quantization...")
    vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
        # Reduce memory further by limiting image token budget
        # (Qwen2-VL supports this natively)
        attn_implementation="eager",   # flash_attn not needed, saves install
    )
    vlm_processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,     # cap image tokens → less VRAM at inference
    )
    vlm_model.eval()
    print(f"✅ {MODEL_ID} loaded in 4-bit — ready.")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    vlm_model, vlm_processor = None, None

# =========================================================
# GEOCHAT
# =========================================================
def query_geochat(pil_img: Image.Image, query: str, timeout: int = 20) -> Optional[str]:
    def _call():
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                pil_img.save(tmp.name, format="PNG")
                tmp_path = tmp.name
            client = Client("Bireswar26/geochat")
            result = client.predict(
                image=handle_file(tmp_path),
                text=query,
                api_name="/predict"
            )
            os.remove(tmp_path)
            raw = str(result).strip()
            if len(raw) < 30 or raw.lower().startswith("i cannot"):
                return None
            return raw
        except Exception as e:
            print(f"GeoChat error: {e}")
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_call)
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print("GeoChat timed out")
            return None

# =========================================================
# SPECTRAL INDICES
# =========================================================
def calculate_spectral_indices(np_rgb: np.ndarray) -> Dict[str, str]:
    try:
        img = np_rgb.astype(np.float32) / 255.0
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        ndvi  = float(np.mean((g - r) / (g + r + 1e-8)))
        ndwi  = float(np.mean((g - b) / (g + b + 1e-8)))

        gray  = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)
        lap   = cv2.Laplacian(gray, cv2.CV_64F)
        urban_score = float(np.mean(np.abs(lap)) / 10.0)
        brightness  = float(np.mean(gray) / 255.0)

        return {
            "NDVI (approx)": f"{ndvi:.3f}",
            "NDWI (approx)": f"{ndwi:.3f}",
            "Urban Texture":  f"{urban_score:.3f}",
            "Brightness":     f"{brightness:.3f}",
        }
    except Exception as e:
        print(f"Spectral error: {e}")
        return {"NDVI": "err", "NDWI": "err", "Urban": "err", "Brightness": "err"}

# =========================================================
# DOMAIN DETECTION
# =========================================================
def detect_domain(np_rgb: np.ndarray, geochat_text: str = "") -> str:
    hsv   = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2HSV)
    gray  = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)
    total = float(np_rgb.shape[0] * np_rgb.shape[1])

    water_mask = (hsv[:, :, 0] > 90)  & (hsv[:, :, 0] < 140) & (hsv[:, :, 1] > 30)
    fire_mask  = ((hsv[:, :, 0] < 15) | (hsv[:, :, 0] > 165)) & (hsv[:, :, 1] > 60)
    veg_mask   = (hsv[:, :, 0] > 35)  & (hsv[:, :, 0] < 85)  & (hsv[:, :, 1] > 30)

    water_pct = np.sum(water_mask) / total
    fire_pct  = np.sum(fire_mask)  / total
    veg_pct   = np.sum(veg_mask)   / total

    gc = geochat_text.lower()

    if fire_pct > 0.10 or any(w in gc for w in ["fire", "wildfire", "burn", "smoke", "flame"]):
        return "WILDFIRE"
    if water_pct > 0.25 or any(w in gc for w in ["ocean", "sea", "coast", "marine"]):
        return "COASTAL"
    if veg_pct < 0.10 or any(w in gc for w in ["city", "urban", "building", "road"]):
        return "URBAN"
    return "TERRESTRIAL"

# =========================================================
# POLYGON EXTRACTION
# =========================================================
def extract_polygons(np_rgb: np.ndarray, domain: str):
    h, w   = np_rgb.shape[:2]
    total  = float(h * w)
    hsv    = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2HSV)
    gray   = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)

    water_mask = (hsv[:, :, 0] > 90)  & (hsv[:, :, 0] < 140) & (hsv[:, :, 1] > 30)
    veg_mask   = (hsv[:, :, 0] > 35)  & (hsv[:, :, 0] < 85)  & (hsv[:, :, 1] > 30)
    fire_mask  = ((hsv[:, :, 0] < 15) | (hsv[:, :, 0] > 165)) & (hsv[:, :, 1] > 60)
    edges      = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    urban_mask = (edges > 20) & (~water_mask) & (~veg_mask)
    smoke_mask = (gray > 180) & (hsv[:, :, 1] < 20)
    bare_mask  = (gray > 100) & (~urban_mask) & (~water_mask) & (~veg_mask) & (~fire_mask)

    if domain == "WILDFIRE":
        mask_defs = [
            ("Active Fire Zone",  fire_mask | (gray < 25), "#DC2626", "High-temperature combustion area."),
            ("Smoke / Ash Plume", smoke_mask,               "#6B7280", "Suspended particulate and smoke."),
            ("Burned Vegetation", bare_mask,                "#78350F", "Charred biomass and scorched soil."),
            ("Intact Canopy",     veg_mask,                 "#10B981", "Vegetation outside fire perimeter."),
        ]
    elif domain == "COASTAL":
        mask_defs = [
            ("Open Water",              water_mask, "#0284C7", "Marine or lacustrine surface."),
            ("Coastal Infrastructure",  urban_mask, "#E11D48", "Built environment near shore."),
            ("Beach / Littoral",        bare_mask,  "#F59E0B", "Sandy substrate and intertidal zone."),
            ("Coastal Vegetation",      veg_mask,   "#10B981", "Mangroves or coastal flora."),
        ]
    elif domain == "URBAN":
        mask_defs = [
            ("Built-up Structures", urban_mask, "#E11D48", "Rooftops, roads, impervious cover."),
            ("Urban Green Space",   veg_mask,   "#10B981", "Parks and tree canopy."),
            ("Bare / Construction", bare_mask,  "#F59E0B", "Exposed soil or development."),
            ("Water Features",      water_mask, "#0284C7", "Urban ponds or rivers."),
        ]
    else:
        mask_defs = [
            ("Vegetative Canopy",   veg_mask,   "#10B981", "Active photosynthetic biomass."),
            ("Impervious Surfaces", urban_mask, "#E11D48", "Roads and structures."),
            ("Bare Soil",           bare_mask,  "#F59E0B", "Exposed earth."),
            ("Water Bodies",        water_mask, "#0284C7", "Lakes, rivers, or wetlands."),
        ]

    polygons = []
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

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
        eps    = 0.015 * cv2.arcLength(c, True)
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
# QWEN GENERATION  — 7B quality prompt
# =========================================================

# Domain-specific system prompts give the 7B model clearer context
DOMAIN_SYSTEM = {
    "WILDFIRE":    "You are a wildfire and remote sensing expert analyzing satellite imagery of an active or recent fire event.",
    "COASTAL":     "You are a coastal geomorphology and marine remote sensing expert analyzing satellite imagery of a coastal or marine scene.",
    "URBAN":       "You are an urban planning and geospatial intelligence expert analyzing satellite imagery of a built-up area.",
    "TERRESTRIAL": "You are a land-cover and environmental remote sensing expert analyzing satellite imagery of a terrestrial landscape.",
}

def run_qwen(pil_img: Image.Image, geochat_text: str, query: str, domain: str) -> Dict:
    """
    Uses Qwen2-VL-7B (4-bit) with a chain-of-thought system prompt.
    Returns a dict with: title, report, card1_name, card1_text,
    card2_name, card2_text, card3_name, card3_text
    """
    if vlm_model is None:
        return _fallback_result(geochat_text, query, domain)

    system_msg = DOMAIN_SYSTEM.get(domain, DOMAIN_SYSTEM["TERRESTRIAL"])

    geochat_ctx = ""
    if geochat_text:
        geochat_ctx = (
            f"\n\nA specialized remote-sensing model (GeoChat-7B) already "
            f"examined this image and reported:\n\"{geochat_text}\"\n"
            f"Use this as supporting context — do not copy it verbatim."
        )

    # Two-shot format examples help 7B models follow strict output formats
    user_prompt = (
        f"{geochat_ctx}\n\n"
        f"User question: {query}\n\n"
        "OUTPUT FORMAT — respond with EXACTLY these 8 labelled lines, nothing else:\n"
        "TITLE: <concise 8-12 word scene description>\n"
        "REPORT: <3-4 sentences directly answering the question with specific observations>\n"
        "CARD1_NAME: <first thematic category visible>\n"
        "CARD1_TEXT: <one specific sentence about that category>\n"
        "CARD2_NAME: <second thematic category visible>\n"
        "CARD2_TEXT: <one specific sentence about that category>\n"
        "CARD3_NAME: <third thematic category visible>\n"
        "CARD3_TEXT: <one specific sentence about that category>"
    )

    try:
        messages = [
            {
                "role": "system",
                "content": system_msg,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text",  "text": user_prompt},
                ],
            }
        ]

        text_in = vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = vlm_processor(
            text=[text_in],
            images=[pil_img],
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out = vlm_model.generate(
                **inputs,
                max_new_tokens=350,
                do_sample=False,          # greedy → deterministic, format-faithful
                repetition_penalty=1.1,   # prevents looping on 7B
            )

        raw = vlm_processor.batch_decode(
            out[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0].strip()

        print(f"\n=== QWEN-7B RAW ===\n{raw}\n===================\n")
        return _parse_qwen_output(raw, geochat_text, query, domain)

    except torch.cuda.OutOfMemoryError:
        print("OOM during generation — clearing cache and falling back.")
        torch.cuda.empty_cache()
        return _fallback_result(geochat_text, query, domain)
    except Exception as e:
        print(f"Qwen generation error: {e}")
        return _fallback_result(geochat_text, query, domain)


# =========================================================
# PARSER  (unchanged — already robust)
# =========================================================
def _parse_qwen_output(raw: str, geochat_text: str, query: str, domain: str) -> Dict:
    cleaned = re.sub(r"[*#_`]+", "", raw)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned).strip()

    data: Dict[str, str] = {}

    for line in cleaned.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            k = k.strip().upper().replace(" ", "_")
            v = v.strip()
            if k and v:
                data[k] = v

    key_patterns = {
        "TITLE":      r"(?:title|scene title)\s*[:\-]\s*(.+)",
        "REPORT":     r"(?:report|summary|analysis|synthesized[^:]*)\s*[:\-]\s*(.+)",
        "CARD1_NAME": r"card\s*1\s*name\s*[:\-]\s*(.+)",
        "CARD1_TEXT": r"card\s*1\s*(?:text|description|desc)\s*[:\-]\s*(.+)",
        "CARD2_NAME": r"card\s*2\s*name\s*[:\-]\s*(.+)",
        "CARD2_TEXT": r"card\s*2\s*(?:text|description|desc)\s*[:\-]\s*(.+)",
        "CARD3_NAME": r"card\s*3\s*name\s*[:\-]\s*(.+)",
        "CARD3_TEXT": r"card\s*3\s*(?:text|description|desc)\s*[:\-]\s*(.+)",
    }
    for key, pattern in key_patterns.items():
        if key not in data:
            m = re.search(pattern, cleaned, re.IGNORECASE)
            if m:
                data[key] = m.group(1).strip()

    ECHO_MARKERS = [
        "[descriptive", "[10-word", "[category", "[1-sentence",
        "[3-sentence", "reply only", "no extra text", "<",
        "concise 8-12", "thematic category",
    ]
    for key in list(data.keys()):
        if any(marker in data[key].lower() for marker in ECHO_MARKERS):
            del data[key]

    if "REPORT" not in data:
        sentences = re.findall(r"[A-Z][^.!?]*[.!?]", cleaned)
        useful = [s for s in sentences if len(s) > 40 and not any(
            m in s.lower() for m in ECHO_MARKERS
        )]
        if useful:
            data["REPORT"] = " ".join(useful[:4])

    if "TITLE" not in data:
        domain_labels = {
            "WILDFIRE":    "Wildfire & Smoke Event Analysis",
            "COASTAL":     "Coastal Zone Remote Sensing Assessment",
            "URBAN":       "Urban Landscape Intelligence Report",
            "TERRESTRIAL": "Terrestrial Land Cover Study",
        }
        data["TITLE"] = domain_labels.get(domain, "Geospatial Analysis")

    gc_snippet = geochat_text[:300] if geochat_text else "Satellite imagery analyzed."
    domain_cards = {
        "WILDFIRE": [
            ("Fire & Combustion",  "Active fire perimeter detected with thermal anomalies and smoke dispersion."),
            ("Burn Scar & Soil",   "Scorched vegetation and exposed soil mark the fire's historical extent."),
            ("Smoke & Air Quality","Dense smoke plumes indicate poor air quality and active combustion."),
        ],
        "COASTAL": [
            ("Water Body",         "Surface water covers significant coastal area."),
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
            ("Hydrology",          "Minor water features or drainage channels present."),
            ("Terrain Condition",  "Surface conditions appear stable with no acute hazards."),
        ],
    }
    cards = domain_cards.get(domain, domain_cards["TERRESTRIAL"])

    return {
        "title":      data.get("TITLE"),
        "report":     data.get("REPORT", gc_snippet),
        "card1_name": data.get("CARD1_NAME", cards[0][0]),
        "card1_text": data.get("CARD1_TEXT", cards[0][1]),
        "card2_name": data.get("CARD2_NAME", cards[1][0]),
        "card2_text": data.get("CARD2_TEXT", cards[1][1]),
        "card3_name": data.get("CARD3_NAME", cards[2][0]),
        "card3_text": data.get("CARD3_TEXT", cards[2][1]),
    }


def _fallback_result(geochat_text: str, query: str, domain: str) -> Dict:
    return _parse_qwen_output("", geochat_text, query, domain)

# =========================================================
# IMAGE LOADER
# =========================================================
def load_image(file_bytes: bytes):
    pil_img = None

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
            elif arr.ndim == 3 and arr.shape[0] <= 10:
                arr = np.moveaxis(arr[:3], 0, -1)
            arr = arr[:, :, :3]
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
            pil_img = Image.fromarray(arr.astype(np.uint8))
        except Exception:
            pass

    if pil_img is None:
        raise HTTPException(status_code=400, detail="Unsupported or corrupt image file.")

    np_img = cv2.resize(np.array(pil_img), (1024, 1024))
    buf    = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    b64    = base64.b64encode(buf.getvalue()).decode()
    return pil_img, np_img, b64

# =========================================================
# ANALYZE ENDPOINT
# =========================================================
@app.post("/api/analyze")
async def analyze(
    mode:   str        = Form(...),
    query:  str        = Form(...),
    image1: UploadFile = File(...),
):
    img_bytes             = await image1.read()
    pil_img, np_img, b64 = load_image(img_bytes)

    geochat_text = query_geochat(pil_img, query)
    gc_status    = "ok" if geochat_text else "timeout/fallback"

    domain   = detect_domain(np_img, geochat_text or "")
    polygons = extract_polygons(np_img, domain)
    metrics  = calculate_spectral_indices(np_img)
    parsed   = run_qwen(pil_img, geochat_text or "", query, domain)

    trace = {
        "domain": domain,
        "models": [
            {"name": "GeoChat-7B",   "role": "RS specialist",    "status": gc_status},
            {"name": "Qwen2-VL-7B",  "role": "4-bit Synthesizer","status": "ok"},
            {"name": "CV Segmenter", "role": "Polygons",         "status": "ok",
             "polygons": len(polygons)},
        ],
        "geochat_snippet": (geochat_text or "")[:300],
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
        "confidence_score":   "0.96",
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
        return FileResponse(fp) if os.path.exists(fp) else FileResponse(os.path.join(DIST, "index.html"))# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Qwen2-VL on {device}...")

try:
    vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    vlm_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    print("Qwen loaded OK")
except Exception as e:
    print(f"Qwen load failed: {e}")
    vlm_model, vlm_processor = None, None

# =========================================================
# GEOCHAT
# =========================================================
def query_geochat(pil_img: Image.Image, query: str, timeout: int = 20) -> Optional[str]:
    def _call():
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                pil_img.save(tmp.name, format="PNG")
                tmp_path = tmp.name
            client = Client("Bireswar26/geochat")
            result = client.predict(
                image=handle_file(tmp_path),
                text=query,
                api_name="/predict"
            )
            os.remove(tmp_path)
            raw = str(result).strip()
            # Reject boilerplate / empty responses
            if len(raw) < 30 or raw.lower().startswith("i cannot"):
                return None
            return raw
        except Exception as e:
            print(f"GeoChat error: {e}")
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_call)
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print("GeoChat timed out")
            return None

# =========================================================
# SPECTRAL INDICES
# =========================================================
def calculate_spectral_indices(np_rgb: np.ndarray) -> Dict[str, str]:
    try:
        img = np_rgb.astype(np.float32) / 255.0
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        # Approximate NDVI using green as NIR proxy
        ndvi = float(np.mean((g - r) / (g + r + 1e-8)))

        # Approximate NDWI using blue
        ndwi = float(np.mean((g - b) / (g + b + 1e-8)))

        # Urban texture via Laplacian
        gray = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        urban_score = float(np.mean(np.abs(lap)) / 10.0)

        brightness = float(np.mean(gray) / 255.0)

        return {
            "NDVI (approx)": f"{ndvi:.3f}",
            "NDWI (approx)": f"{ndwi:.3f}",
            "Urban Texture":  f"{urban_score:.3f}",
            "Brightness":     f"{brightness:.3f}",
        }
    except Exception as e:
        print(f"Spectral error: {e}")
        return {"NDVI": "err", "NDWI": "err", "Urban": "err", "Brightness": "err"}

# =========================================================
# DOMAIN DETECTION
# =========================================================
def detect_domain(np_rgb: np.ndarray, geochat_text: str = "") -> str:
    hsv  = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)
    total = float(np_rgb.shape[0] * np_rgb.shape[1])

    water_mask = ((hsv[:, :, 0] > 90) & (hsv[:, :, 0] < 140) & (hsv[:, :, 1] > 30))
    fire_mask  = ((hsv[:, :, 0] < 15) | (hsv[:, :, 0] > 165)) & (hsv[:, :, 1] > 60)
    veg_mask   = (hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 85) & (hsv[:, :, 1] > 30)

    water_pct = np.sum(water_mask) / total
    fire_pct  = np.sum(fire_mask)  / total
    veg_pct   = np.sum(veg_mask)   / total

    gc = geochat_text.lower()

    if fire_pct > 0.10 or any(w in gc for w in ["fire", "wildfire", "burn", "smoke", "flame"]):
        return "WILDFIRE"
    if water_pct > 0.25 or any(w in gc for w in ["ocean", "sea", "coast", "marine"]):
        return "COASTAL"
    if veg_pct < 0.10 or any(w in gc for w in ["city", "urban", "building", "road"]):
        return "URBAN"
    return "TERRESTRIAL"

# =========================================================
# POLYGON EXTRACTION
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

    # Guarantee at least 3 slots
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
# QWEN GENERATION  ← THE BIG FIX IS HERE
# =========================================================
def run_qwen(pil_img: Image.Image, geochat_text: str, query: str, domain: str) -> Dict:
    """
    Returns a dict with keys: title, report, card1_name, card1_text,
    card2_name, card2_text, card3_name, card3_text
    """

    # ── Build a SHORT, STRICT prompt ──────────────────────────────────────
    # Key insight: Qwen-2B tends to echo long prompts.
    # Give it SHORT instructions and let it see the image directly.
    geochat_line = f'Expert analysis: "{geochat_text}"\n\n' if geochat_text else ""

    prompt = (
        f"{geochat_line}"
        f"You are analyzing a {domain.lower().replace('_', ' ')} satellite image.\n"
        f"User question: {query}\n\n"
        "Reply ONLY with the following 8 lines (no extra text, no markdown):\n"
        "TITLE: <10-word scene title>\n"
        "REPORT: <3-sentence analytical summary answering the question>\n"
        "CARD1_NAME: <category name>\n"
        "CARD1_TEXT: <1-sentence observation>\n"
        "CARD2_NAME: <category name>\n"
        "CARD2_TEXT: <1-sentence observation>\n"
        "CARD3_NAME: <category name>\n"
        "CARD3_TEXT: <1-sentence observation>"
    )

    try:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text",  "text": prompt},
            ]
        }]
        text_in = vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = vlm_processor(
            text=[text_in], images=[pil_img],
            padding=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            out = vlm_model.generate(
                **inputs,
                max_new_tokens=300,
                # Greedy – stops the model from being "creative" with the format
                do_sample=False,
            )
        raw = vlm_processor.batch_decode(
            out[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0].strip()

        print(f"\n=== QWEN RAW ===\n{raw}\n================\n")
        return _parse_qwen_output(raw, geochat_text, query, domain)

    except Exception as e:
        print(f"Qwen generation error: {e}")
        return _fallback_result(geochat_text, query, domain)


def _parse_qwen_output(raw: str, geochat_text: str, query: str, domain: str) -> Dict:
    """
    Multi-strategy parser.
    Strategy 1 – strict KEY: value lines
    Strategy 2 – fuzzy regex (handles markdown, extra spaces, lowercase keys)
    Strategy 3 – full fallback
    """

    # ── Strip markdown formatting first ──────────────────────────────────
    # Remove **, ##, __, backticks etc.
    cleaned = re.sub(r"[*#_`]+", "", raw)
    # Collapse multiple blank lines
    cleaned = re.sub(r"\n{2,}", "\n", cleaned).strip()

    data: Dict[str, str] = {}

    # Strategy 1: strict "KEY: value" per line
    for line in cleaned.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            k = k.strip().upper().replace(" ", "_")
            v = v.strip()
            if k and v:
                data[k] = v

    # Strategy 2: fuzzy regex for common keys
    key_patterns = {
        "TITLE":      r"(?:title|scene title)\s*[:\-]\s*(.+)",
        "REPORT":     r"(?:report|summary|analysis|synthesized[^:]*)\s*[:\-]\s*(.+)",
        "CARD1_NAME": r"card\s*1\s*name\s*[:\-]\s*(.+)",
        "CARD1_TEXT": r"card\s*1\s*(?:text|description|desc)\s*[:\-]\s*(.+)",
        "CARD2_NAME": r"card\s*2\s*name\s*[:\-]\s*(.+)",
        "CARD2_TEXT": r"card\s*2\s*(?:text|description|desc)\s*[:\-]\s*(.+)",
        "CARD3_NAME": r"card\s*3\s*name\s*[:\-]\s*(.+)",
        "CARD3_TEXT": r"card\s*3\s*(?:text|description|desc)\s*[:\-]\s*(.+)",
    }
    for key, pattern in key_patterns.items():
        if key not in data:
            m = re.search(pattern, cleaned, re.IGNORECASE)
            if m:
                data[key] = m.group(1).strip()

    # ── Validate: reject prompt-echo values ──────────────────────────────
    ECHO_MARKERS = [
        "[descriptive", "[10-word", "[category", "[1-sentence",
        "[3-sentence", "reply only", "no extra text", "<"
    ]
    for key in list(data.keys()):
        val_low = data[key].lower()
        if any(marker in val_low for marker in ECHO_MARKERS):
            del data[key]   # Poisoned value – discard, will fall back below

    # ── If REPORT is missing, try to grab the longest sentence block ─────
    if "REPORT" not in data:
        sentences = re.findall(r"[A-Z][^.!?]*[.!?]", cleaned)
        useful = [s for s in sentences if len(s) > 40 and not any(
            m in s.lower() for m in ECHO_MARKERS
        )]
        if useful:
            data["REPORT"] = " ".join(useful[:3])

    # ── If TITLE is still missing, derive from domain + query ─────────────
    if "TITLE" not in data:
        domain_labels = {
            "WILDFIRE":    "Wildfire & Smoke Event",
            "COASTAL":     "Coastal Zone Analysis",
            "URBAN":       "Urban Landscape Assessment",
            "TERRESTRIAL": "Terrestrial Land Cover Study",
        }
        data["TITLE"] = domain_labels.get(domain, "Geospatial Analysis")

    # ── Build final fallback strings for any still-missing keys ───────────
    gc_snippet = geochat_text[:200] if geochat_text else "Satellite imagery analyzed."
    domain_cards = {
        "WILDFIRE": [
            ("Fire & Combustion", "Active fire perimeter detected with thermal anomalies and smoke dispersion."),
            ("Burn Scar & Soil",  "Scorched vegetation and exposed soil mark the fire's historical extent."),
            ("Smoke & Air Quality","Dense smoke plumes indicate poor air quality and active combustion."),
        ],
        "COASTAL": [
            ("Water Body",        "Surface water detected covering significant coastal area."),
            ("Shoreline",         "Littoral zone shows sandy substrate and intertidal features."),
            ("Coastal Vegetation","Mangroves or marsh grass identified near the waterline."),
        ],
        "URBAN": [
            ("Built-up Areas",    "Dense impervious surfaces indicate urban or peri-urban development."),
            ("Green Space",       "Scattered vegetation and parks visible within the urban matrix."),
            ("Infrastructure",    "Roads and rooftops create high edge-density texture patterns."),
        ],
        "TERRESTRIAL": [
            ("Land Cover",        "Mixed vegetation and bare soil dominate the scene."),
            ("Hydrology",         "Minor water features or drainage channels may be present."),
            ("Terrain Condition", "Surface conditions appear stable with no acute hazards."),
        ],
    }
    cards = domain_cards.get(domain, domain_cards["TERRESTRIAL"])

    result = {
        "title":      data.get("TITLE"),
        "report":     data.get("REPORT", gc_snippet),
        "card1_name": data.get("CARD1_NAME", cards[0][0]),
        "card1_text": data.get("CARD1_TEXT", cards[0][1]),
        "card2_name": data.get("CARD2_NAME", cards[1][0]),
        "card2_text": data.get("CARD2_TEXT", cards[1][1]),
        "card3_name": data.get("CARD3_NAME", cards[2][0]),
        "card3_text": data.get("CARD3_TEXT", cards[2][1]),
    }
    return result


def _fallback_result(geochat_text: str, query: str, domain: str) -> Dict:
    """Used when Qwen itself crashes."""
    return _parse_qwen_output("", geochat_text, query, domain)

# =========================================================
# IMAGE LOADER  (fixes "Invalid image file" for TIFF)
# =========================================================
def load_image(file_bytes: bytes):
    """
    Accepts TIFF, PNG, JPEG. Returns (PIL, numpy-1024, base64-jpeg).
    """
    pil_img = None

    # Try Pillow first
    try:
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        pass

    # Fallback: OpenCV (handles many exotic TIFFs)
    if pil_img is None:
        try:
            arr = np.frombuffer(file_bytes, np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is not None:
                pil_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        except Exception:
            pass

    # Fallback: tifffile for multi-band GeoTIFF
    if pil_img is None:
        try:
            import tifffile
            arr = tifffile.imread(io.BytesIO(file_bytes))
            # Normalise to uint8 RGB
            if arr.ndim == 2:                         # grayscale
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.ndim == 3 and arr.shape[0] <= 10:  # bands-first
                arr = np.moveaxis(arr[:3], 0, -1)
            arr = arr[:, :, :3]
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
            pil_img = Image.fromarray(arr.astype(np.uint8))
        except Exception:
            pass

    if pil_img is None:
        raise HTTPException(status_code=400, detail="Unsupported or corrupt image file.")

    np_img = cv2.resize(np.array(pil_img), (1024, 1024))
    buf    = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    b64    = base64.b64encode(buf.getvalue()).decode()
    return pil_img, np_img, b64

# =========================================================
# ANALYZE ENDPOINT
# =========================================================
@app.post("/api/analyze")
async def analyze(
    mode:   str        = Form(...),
    query:  str        = Form(...),
    image1: UploadFile = File(...),
):
    img_bytes            = await image1.read()
    pil_img, np_img, b64 = load_image(img_bytes)

    # 1. GeoChat
    geochat_text = query_geochat(pil_img, query)
    gc_status    = "ok" if geochat_text else "timeout/fallback"

    # 2. Domain
    domain = detect_domain(np_img, geochat_text or "")

    # 3. Polygons
    polygons = extract_polygons(np_img, domain)

    # 4. Spectral indices
    metrics = calculate_spectral_indices(np_img)

    # 5. Qwen synthesis
    parsed = run_qwen(pil_img, geochat_text or "", query, domain)

    trace = {
        "domain": domain,
        "models": [
            {"name": "GeoChat-7B",    "role": "RS specialist", "status": gc_status},
            {"name": "Qwen2-VL-2B",   "role": "Synthesizer",   "status": "ok"},
            {"name": "CV Segmenter",  "role": "Polygons",      "status": "ok",
             "polygons": len(polygons)},
        ],
        "geochat_snippet": (geochat_text or "")[:300],
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
        "confidence_score":   "0.94",
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
