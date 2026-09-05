# ===== ./app.py =====
from flask import Flask, render_template, request, jsonify
import numpy as np
from mock_backend import run_place_workflow, run_upload_workflow
import base64
from io import BytesIO
from PIL import Image
from pyngrok import ngrok

app = Flask(__name__, template_folder='templates', static_folder='static')

def array_to_base64(arr):
    if arr is None:
        return None
    pil_img = Image.fromarray(arr.astype(np.uint8))
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def handle_query():
    data = request.json
    mode = data.get('mode', 'Autofetch')
    query = data.get('query', 'Describe this image')
    place_text = data.get('place', '')
    
    img1, img2 = None, None
    if data.get('image1'):
        img1 = np.array(Image.open(BytesIO(base64.b64decode(data['image1'].split(',')[1]))))
    if data.get('image2'):
        img2 = np.array(Image.open(BytesIO(base64.b64decode(data['image2'].split(',')[1]))))
    
    if mode == 'Autofetch':
        ans_html, evidence, _, _, _, exec_s, _ = run_place_workflow(place=place_text or "Coastal Region", lat=0, lon=0, start_date="", end_date="", goal="Understand scene", query=query)
    else:
        ans_html, evidence, _, _, exec_s, _ = run_upload_workflow(mode=mode, query=query, preview1=img1, preview2=img2)
    
    return jsonify({'success': True, 'answer': ans_html, 'image': array_to_base64(evidence), 'trace': exec_s, 'confidence': 0.88})

if __name__ == '__main__':
    public_url = ngrok.connect(5000)
    print(f" SatQuery AI is live at: {public_url}")
    app.run(port=5000)
