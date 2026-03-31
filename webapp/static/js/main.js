/**
 * NeuroScan AI — Frontend JavaScript
 * Handles drag-and-drop upload, AJAX prediction, and dynamic result rendering.
 */

document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    const previewContainer = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const clearBtn = document.getElementById('clearBtn');
    const newScanBtn = document.getElementById('newScanBtn');
    const uploadSection = document.getElementById('uploadSection');
    const resultsSection = document.getElementById('resultsSection');

    let selectedFile = null;

    // ─── Drag & Drop ───
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type.startsWith('image/')) {
            handleFile(files[0]);
        }
    });

    // ─── Click to Browse ───
    uploadArea.addEventListener('click', (e) => {
        if (e.target === browseBtn || browseBtn.contains(e.target)) return;
        fileInput.click();
    });

    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // ─── File Handling ───
    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload a valid image file.');
            return;
        }
        if (file.size > 10 * 1024 * 1024) {
            alert('File size must be less than 10MB.');
            return;
        }

        selectedFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadArea.style.display = 'none';
            previewContainer.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }

    // ─── Clear ───
    clearBtn.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        previewContainer.style.display = 'none';
        uploadArea.style.display = 'block';
    });

    // ─── New Scan ───
    newScanBtn.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        previewContainer.style.display = 'none';
        uploadArea.style.display = 'block';
        resultsSection.style.display = 'none';
        uploadSection.style.display = 'block';
    });

    // ─── Analyze ───
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        const btnText = analyzeBtn.querySelector('.btn-text');
        const btnLoader = analyzeBtn.querySelector('.btn-loader');
        btnText.style.display = 'none';
        btnLoader.style.display = 'inline-flex';
        analyzeBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || 'Server error');
            }

            const data = await response.json();
            displayResults(data);
        } catch (error) {
            alert(`Analysis failed: ${error.message}`);
            console.error('Prediction error:', error);
        } finally {
            btnText.style.display = 'inline';
            btnLoader.style.display = 'none';
            analyzeBtn.disabled = false;
        }
    });

    // ─── Display Results ───
    function displayResults(data) {
        uploadSection.style.display = 'none';
        resultsSection.style.display = 'block';

        // Original image
        const resultOriginal = document.getElementById('resultOriginal');
        resultOriginal.src = `data:image/png;base64,${data.original}`;

        // Classification
        if (data.classification) {
            const cls = data.classification;

            // Tumor type
            document.getElementById('tumorType').textContent = cls.class;
            document.getElementById('tumorDescription').textContent = cls.description;

            // Severity badge
            const badge = document.getElementById('severityBadge');
            badge.textContent = `Severity: ${cls.severity}`;
            badge.className = `result-badge severity-${cls.severity}`;

            // Confidence bar
            const confidence = cls.confidence;
            const confidenceFill = document.getElementById('confidenceFill');
            confidenceFill.textContent = `${confidence}%`;
            // Reset to 0 then animate
            confidenceFill.style.width = '0%';
            setTimeout(() => {
                confidenceFill.style.width = `${confidence}%`;
            }, 100);

            // Probability bars
            const probBars = document.getElementById('probBars');
            probBars.innerHTML = '';

            Object.entries(cls.probabilities).forEach(([name, prob]) => {
                const item = document.createElement('div');
                item.className = 'prob-item';
                item.innerHTML = `
                    <div class="prob-header">
                        <span>${name}</span>
                        <span>${prob}%</span>
                    </div>
                    <div class="prob-track">
                        <div class="prob-fill" style="width: 0%"></div>
                    </div>
                `;
                probBars.appendChild(item);

                // Animate fill
                setTimeout(() => {
                    item.querySelector('.prob-fill').style.width = `${prob}%`;
                }, 100);
            });
        }

        // GradCAM
        const gradcamCard = document.getElementById('gradcamCard');
        if (data.gradcam) {
            gradcamCard.style.display = 'block';
            document.getElementById('resultGradcam').src =
                `data:image/png;base64,${data.gradcam.overlay}`;
        } else {
            gradcamCard.style.display = 'none';
        }

        // Segmentation
        const segCard = document.getElementById('segmentationCard');
        const segOverlayCard = document.getElementById('segOverlayCard');

        if (data.segmentation) {
            segCard.style.display = 'block';
            segOverlayCard.style.display = 'block';

            document.getElementById('resultSegmentation').src =
                `data:image/png;base64,${data.segmentation.mask}`;
            document.getElementById('resultSegOverlay').src =
                `data:image/png;base64,${data.segmentation.overlay}`;
            document.getElementById('tumorArea').textContent =
                `${data.segmentation.tumor_area_percent}%`;
        } else {
            segCard.style.display = 'none';
            segOverlayCard.style.display = 'none';
        }

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
});
