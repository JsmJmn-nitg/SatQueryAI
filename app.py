# ===== ./app.py =====
from flask import Flask, render_template, request, jsonify
import numpy as np
from mock_backend import run_place_workflow, run_upload_workflow
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

def array_to_base64(arr):
    """Convert numpy array to base64 for JSON response"""
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
    
    # Handle image uploads (base64 decoded)
    img1 = None
    img2 = None
    
    if data.get('image1'):
        img_data = data['image1'].split(',')[1]
        img1 = np.array(Image.open(BytesIO(base64.b64decode(img_data))))
    
    if data.get('image2'):
        img_data = data['image2'].split(',')[1]
        img2 = np.array(Image.open(BytesIO(base64.b64decode(img_data))))
    
    # Run the appropriate workflow
    if mode == 'Autofetch':
        ans_html, evidence, opt, sar, map_h, exec_s, rep = run_place_workflow(
            place=place_text or "Coastal Region", 
            lat=0, lon=0, 
            start_date="", end_date="", 
            goal="Understand scene", 
            query=query
        )
        result_img = evidence
    else:
        ans_html, evidence, _, _, exec_s, rep = run_upload_workflow(
            mode=mode, 
            query=query, 
            preview1=img1, 
            preview2=img2
        )
        result_img = evidence
    
    return jsonify({
        'success': True,
        'answer': ans_html,
        'image': array_to_base64(result_img),
        'trace': exec_s,
        'confidence': 0.88
    })

@app.route('/api/health')
def health():
    return jsonify({'status': 'operational'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
