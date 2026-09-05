// ===== ./static/js/app.js =====
document.addEventListener('DOMContentLoaded', () => {
    // Mode switching
    const modeBtns = document.querySelectorAll('.mode-btn');
    const autofetchArea = document.getElementById('autofetchArea');
    const uploadArea = document.getElementById('uploadArea');
    const uploadBox2 = document.getElementById('uploadBox2');
    
    let currentMode = 'Autofetch';
    let image1Data = null;
    let image2Data = null;

    modeBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            modeBtns.forEach(b => b.classList.remove('selected'));
            btn.classList.add('selected');
            
            currentMode = btn.dataset.mode;
            
            if (currentMode === 'Autofetch') {
                autofetchArea.classList.add('active');
                uploadArea.classList.remove('active');
            } else {
                autofetchArea.classList.remove('active');
                uploadArea.classList.add('active');
                
                if (currentMode === 'Single Image') {
                    uploadBox2.style.display = 'none';
                } else {
                    uploadBox2.style.display = 'block';
                }
            }
        });
    });

    // File upload handling
    const fileInput1 = document.getElementById('fileInput1');
    const fileInput2 = document.getElementById('fileInput2');
    const uploadBox1 = document.getElementById('uploadBox1');
    const uploadBox2 = document.getElementById('uploadBox2');
    const fileInfo1 = document.getElementById('fileInfo1');
    const fileInfo2 = document.getElementById('fileInfo2');

    uploadBox1.addEventListener('click', () => fileInput1.click());
    uploadBox2.addEventListener('click', () => fileInput2.click());

    fileInput1.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (file) {
            image1Data = await fileToBase64(file);
            fileInfo1.textContent = `✓ ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
        }
    });

    fileInput2.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (file) {
            image2Data = await fileToBase64(file);
            fileInfo2.textContent = `✓ ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
        }
    });

    function fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => resolve(reader.result);
            reader.onerror = error => reject(error);
        });
    }

    // Submit query
    const submitBtn = document.getElementById('submitBtn');
    const queryInput = document.getElementById('queryInput');
    const placeInput = document.getElementById('placeInput');
    const resultsSection = document.getElementById('resultsSection');
    const answerContent = document.getElementById('answerContent');
    const resultImage = document.getElementById('resultImage');
    const traceContent = document.getElementById('traceContent');

    submitBtn.addEventListener('click', async () => {
        const query = queryInput.value.trim();
        const place = placeInput.value.trim();
        
        if (!query && currentMode === 'Autofetch') {
            queryInput.focus();
            return;
        }

        submitBtn.classList.add('loading');
        
        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    mode: currentMode,
                    query: query || 'Describe this image',
                    place: place,
                    image1: image1Data,
                    image2: image2Data
                })
            });

            const data = await response.json();
            
            if (data.success) {
                answerContent.innerHTML = data.answer;
                resultImage.src = data.image;
                traceContent.textContent = JSON.stringify(data.trace, null, 2);
                document.querySelector('.score').textContent = data.confidence.toFixed(2);
                
                resultsSection.classList.add('visible');
                resultsSection.scrollIntoView({ behavior: 'smooth' });
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to process query. Please try again.');
        } finally {
            submitBtn.classList.remove('loading');
        }
    });

    // Enter key to submit
    queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            submitBtn.click();
        }
    });

    placeInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            submitBtn.click();
        }
    });

    // Theme toggle (basic implementation)
    const themeToggle = document.getElementById('themeToggle');
    themeToggle.addEventListener('click', () => {
        document.body.classList.toggle('light-mode');
        themeToggle.textContent = document.body.classList.contains('light-mode') ? '🌙' : '☀️';
    });
});
