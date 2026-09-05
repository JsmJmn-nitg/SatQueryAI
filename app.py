import gradio as gr
import os

from ui_styles import THEME, CSS, HERO_HTML
from mock_backend import run_place_workflow
from places_tool import autocomplete_places, geocode_place


def google_maps_embed(lat: float, lon: float, zoom: int = 12) -> str:
    # Does not require an API key
    src = f"https://www.google.com/maps?q={lat},{lon}&z={zoom}&output=embed"
    return f"""
    <div class="glass soft cardPad">
      <div class="sectionTitle">Map</div>
      <div style="margin-top:10px; border-radius:16px; overflow:hidden; border:1px solid rgba(148,163,184,.18);">
        <iframe width="100%" height="320" frameborder="0" style="border:0" referrerpolicy="no-referrer-when-downgrade" src="{src}"></iframe>
      </div>
      <div class="smallHint" style="margin-top:10px;">
        Autocomplete: {"Google Places API" if os.getenv("GOOGLE_MAPS_API_KEY") else "OSM Nominatim fallback"}.
        Sentinel/Google Earth imagery fetch can be wired later.
      </div>
    </div>
    """


def on_place_input(text):
    # Called on keystrokes; keep it light and only after 3 chars (handled in tool)
    try:
        suggestions = autocomplete_places(text, limit=5)
        return gr.update(choices=suggestions, value=None, visible=bool(suggestions))
    except Exception:
        return gr.update(choices=[], value=None, visible=False)


def on_suggestion_select(choice):
    if not choice:
        # don't overwrite user's typing if they didn't pick
        return "", "", ""

    lat, lon, formatted = geocode_place(choice)
    if lat is None or lon is None:
        return formatted, "", ""
    return formatted, str(lat), str(lon)


def on_run(place_text, lat_text, lon_text, start_date, end_date, goal, query):
    # If user didn't resolve lat/lon, geocode from place_text (best effort)
    place_text = (place_text or "").strip()
    query = (query or "").strip() or "Summarize what is happening in this area."

    # Normalize dates to strings (gr.DateTime may pass ISO-like strings)
    start_date = str(start_date) if start_date else "2024-01-01"
    end_date = str(end_date) if end_date else "2024-12-31"

    lat = None
    lon = None

    try:
        if lat_text and lon_text:
            lat = float(lat_text)
            lon = float(lon_text)
    except Exception:
        lat = lon = None

    if (lat is None or lon is None) and place_text:
        glat, glon, formatted = geocode_place(place_text)
        if glat is not None and glon is not None:
            lat, lon = glat, glon
            place_text = formatted

    # demo default if still missing
    if lat is None or lon is None:
        lat, lon = 28.6139, 77.2090
        if not place_text:
            place_text = "New Delhi (demo default)"

    answer_md, evidence, optical, sar, _map_html_unused, exec_summary, report_path = run_place_workflow(
        place=place_text,
        lat=lat,
        lon=lon,
        start_date=start_date,
        end_date=end_date,
        goal=goal,
        query=query,
    )
    map_html = google_maps_embed(lat, lon, zoom=12)
    return answer_md, evidence, optical, sar, map_html, exec_summary, report_path


with gr.Blocks(theme=THEME, css=CSS, title="SatQuery AI", elem_id="app") as demo:
    gr.HTML(HERO_HTML)

    with gr.Row():
        # Left: minimal inputs
        with gr.Column(scale=5, min_width=360, elem_classes=["glass", "cardPad"]):
            gr.Markdown('<div class="sectionTitle">Describe your question</div>')

            place_text = gr.Textbox(
                label="Location",
                placeholder="Start typing a place (e.g., 'Chilika Lake', 'Dehradun', 'Guwahati')",
            )
            place_suggestions = gr.Dropdown(
                label="Suggestions",
                choices=[],
                visible=False,
                interactive=True,
                allow_custom_value=False,
            )

            with gr.Row():
                start_date = gr.DateTime(label="From", value="2024-01-01")
                end_date = gr.DateTime(label="To", value="2024-12-31")

            goal = gr.Dropdown(
                label="Goal",
                choices=["Understand scene", "Detect change", "Water / flooding", "Wildfire", "Urban growth", "Custom"],
                value="Understand scene",
            )

            query = gr.Textbox(
                label="Question (plain language)",
                lines=3,
                value="What is happening in this area? Summarize land cover and any notable activity.",
            )

            with gr.Accordion("Advanced (optional)", open=False):
                gr.Markdown("<div class='smallHint'>These are auto-filled when you pick a suggestion.</div>")
                with gr.Row():
                    lat_text = gr.Textbox(label="Latitude", placeholder="auto", interactive=True)
                    lon_text = gr.Textbox(label="Longitude", placeholder="auto", interactive=True)

            with gr.Row():
                run_btn = gr.Button("Run", variant="primary")
                reset_btn = gr.Button("Reset", variant="secondary")

            gr.Markdown(
                "<div class='smallHint'>This is a UI-first demo for judging. Backends (SentinelHub/Google Earth, specialist models) can be plugged in later.</div>"
            )

        # Right: outputs
        with gr.Column(scale=7, min_width=460):
            with gr.Group(elem_classes=["glass", "cardPad"]):
                gr.Markdown('<div class="sectionTitle">Answer</div>')
                answer = gr.Markdown(elem_classes=["tightMd"])

            with gr.Row():
                with gr.Column(elem_classes=["glass", "cardPad", "imageFrame"]):
                    gr.Markdown('<div class="sectionTitle">Evidence</div>')
                    evidence = gr.Image(type="numpy", height=280, show_label=False)

                with gr.Column(elem_classes=["glass", "cardPad", "imageFrame"]):
                    gr.Markdown('<div class="sectionTitle">Optical</div>')
                    optical = gr.Image(type="numpy", height=280, show_label=False)

            with gr.Row():
                with gr.Column(elem_classes=["glass", "cardPad", "imageFrame"]):
                    gr.Markdown('<div class="sectionTitle">SAR</div>')
                    sar = gr.Image(type="numpy", height=260, show_label=False)
                with gr.Column():
                    map_html = gr.HTML()

            with gr.Accordion("Execution Trace + Report", open=False, elem_classes=["glass", "cardPad"]):
                trace = gr.JSON(show_label=False)
                report = gr.File(label="Download report", interactive=False)

    # Events
    place_text.input(fn=on_place_input, inputs=[place_text], outputs=[place_suggestions])
    place_suggestions.change(fn=on_suggestion_select, inputs=[place_suggestions], outputs=[place_text, lat_text, lon_text])

    run_btn.click(
        fn=on_run,
        inputs=[place_text, lat_text, lon_text, start_date, end_date, goal, query],
        outputs=[answer, evidence, optical, sar, map_html, trace, report],
    )

    def _reset():
        return (
            "", gr.update(choices=[], visible=False, value=None),
            "2024-01-01", "2024-12-31",
            "Understand scene",
            "What is happening in this area? Summarize land cover and any notable activity.",
            "", "",
            "", None, None, None, "", {}, None
        )

    reset_btn.click(
        fn=_reset,
        inputs=[],
        outputs=[
            place_text, place_suggestions,
            start_date, end_date,
            goal,
            query,
            lat_text, lon_text,
            answer, evidence, optical, sar, map_html, trace, report
        ],
    )

if __name__ == "__main__":
    demo.launch(share=True)
