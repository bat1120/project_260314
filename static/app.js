document.addEventListener('DOMContentLoaded', () => {
    // ----------------------------------------------------------------
    // 1. Tab Navigation
    // ----------------------------------------------------------------
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            const targetId = btn.getAttribute('data-tab');
            document.getElementById(targetId).classList.add('active');
            
            // Turn off camera when switching tabs
            stopAllStreams();
        });
    });

    const loadingOverlay = document.getElementById('loading');
    function showLoading() { loadingOverlay.classList.remove('hidden'); }
    function hideLoading() { loadingOverlay.classList.add('hidden'); }

    // ----------------------------------------------------------------
    // Webcam Helpers
    // ----------------------------------------------------------------
    let currentStream = null;

    async function startCamera(videoElement, captureBtn) {
        stopAllStreams();
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            videoElement.srcObject = stream;
            currentStream = stream;
            if(captureBtn) captureBtn.disabled = false;
        } catch (err) {
            alert("카메라에 접근할 수 없습니다: " + err.message);
        }
    }

    function stopAllStreams() {
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
            currentStream = null;
        }
        // Disable capture buttons
        document.getElementById('pose-capture').disabled = true;
        document.getElementById('face-capture').disabled = true;
        document.getElementById('ocr-capture').disabled = true;
    }

    function captureImage(videoElement, canvasElement) {
        const context = canvasElement.getContext('2d');
        canvasElement.width = videoElement.videoWidth;
        canvasElement.height = videoElement.videoHeight;
        context.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
        
        return new Promise(resolve => {
            canvasElement.toBlob(blob => resolve(blob), 'image/jpeg', 0.95);
        });
    }

    function fileToBlob(file) {
        return new Promise((resolve) => {
            resolve(file); // File inherits from Blob
        });
    }

    function displayResultImage(boxId, imageUrl) {
        const box = document.getElementById(boxId);
        box.innerHTML = `<img src="${imageUrl}?t=${new Date().getTime()}" alt="Result">`;
    }

    // ----------------------------------------------------------------
    // 2. POSE Tab Logic
    // ----------------------------------------------------------------
    const poseVideo = document.getElementById('pose-video');
    const poseCanvas = document.getElementById('pose-canvas');
    const btnPoseCam = document.getElementById('pose-start-cam');
    const btnPoseCap = document.getElementById('pose-capture');
    const inputPoseUp = document.getElementById('pose-upload');

    btnPoseCam.addEventListener('click', () => startCamera(poseVideo, btnPoseCap));

    btnPoseCap.addEventListener('click', async () => {
        const blob = await captureImage(poseVideo, poseCanvas);
        await sendPoseApi(blob);
    });

    inputPoseUp.addEventListener('change', async (e) => {
        if (e.target.files.length > 0) {
            stopAllStreams();
            const blob = e.target.files[0];
            await sendPoseApi(blob);
        }
    });

    async function sendPoseApi(imageBlob) {
        showLoading();
        const formData = new FormData();
        formData.append('file', imageBlob, 'pose.jpg');

        try {
            const res = await fetch('/api/pose', { method: 'POST', body: formData });
            const data = await res.json();
            if (data.status === 'success' || data.status === 'no_detection') {
                const box = document.getElementById('pose-result-box');
                box.innerHTML = `
                    <div style="margin-bottom: 1rem; padding: 1rem; background: rgba(0,0,0,0.4); border-radius: 8px; border-left: 4px solid var(--primary); width: 100%; text-align: left;">
                        <strong style="color: #818cf8; display: block; margin-bottom: 0.5rem; font-size: 1.1rem;">💡 자세 교정 피드백:</strong>
                        <span style="line-height: 1.5; color: #f8fafc;">${data.feedback || '분석 결과를 기다려 주세요.'}</span>
                    </div>
                    <img src="${data.result_url}?t=${new Date().getTime()}" alt="Result" style="border-radius: 8px; max-width: 100%;">
                `;
            } else {
                alert("분석 실패: " + data.error);
            }
        } catch(err) {
            alert("서버 연결 오류");
        }
        hideLoading();
    }

    // ----------------------------------------------------------------
    // 3. FACE Tab Logic
    // ----------------------------------------------------------------
    const faceRefUpload = document.getElementById('face-ref-upload');
    const faceRefPreview = document.getElementById('face-ref-preview');
    const faceVideo = document.getElementById('face-video');
    const faceCanvas = document.getElementById('face-canvas');
    const faceCamPreview = document.getElementById('face-cam-preview');
    
    const btnFaceCam = document.getElementById('face-start-cam');
    const btnFaceCap = document.getElementById('face-capture');
    const btnFaceVerify = document.getElementById('face-verify-btn');
    const faceResultBox = document.getElementById('face-result');

    let faceRefBlob = null;
    let faceCamBlob = null;

    faceRefUpload.addEventListener('change', (e) => {
        if(e.target.files.length > 0) {
            faceRefBlob = e.target.files[0];
            faceRefPreview.innerHTML = `<img src="${URL.createObjectURL(faceRefBlob)}">`;
            checkFaceReady();
        }
    });

    btnFaceCam.addEventListener('click', () => startCamera(faceVideo, btnFaceCap));

    btnFaceCap.addEventListener('click', async () => {
        faceCamBlob = await captureImage(faceVideo, faceCanvas);
        faceCamPreview.innerHTML = `<img src="${URL.createObjectURL(faceCamBlob)}">`;
        checkFaceReady();
    });

    function checkFaceReady() {
        if(faceRefBlob && faceCamBlob) {
            btnFaceVerify.disabled = false;
        }
    }

    btnFaceVerify.addEventListener('click', async () => {
        if(!faceRefBlob || !faceCamBlob) return;
        
        showLoading();
        const formData = new FormData();
        formData.append('ref_file', faceRefBlob, 'ref.jpg');
        formData.append('cam_file', faceCamBlob, 'cam.jpg');

        try {
            const res = await fetch('/api/face', { method: 'POST', body: formData });
            const data = await res.json();
            
            faceResultBox.classList.remove('hidden', 'success', 'error');
            
            if (data.status === 'success') {
                if(data.is_same) {
                    faceResultBox.classList.add('success');
                    faceResultBox.innerHTML = `✅ 본인 인증 성공! (유사도: ${(data.similarity*100).toFixed(1)}%)`;
                } else {
                    faceResultBox.classList.add('error');
                    faceResultBox.innerHTML = `❌ 본인 인증 실패. 타인입니다. (유사도: ${(data.similarity*100).toFixed(1)}%)`;
                }
            } else {
                faceResultBox.classList.add('error');
                faceResultBox.innerHTML = `⚠️ 오류: ${data.message || data.error}`;
            }
        } catch(err) {
            alert("서버 연결 오류");
        }
        hideLoading();
    });

    // ----------------------------------------------------------------
    // 4. OCR Tab Logic
    // ----------------------------------------------------------------
    const ocrVideo = document.getElementById('ocr-video');
    const ocrCanvas = document.getElementById('ocr-canvas');
    const btnOcrCam = document.getElementById('ocr-start-cam');
    const btnOcrCap = document.getElementById('ocr-capture');
    const inputOcrUp = document.getElementById('ocr-upload');

    btnOcrCam.addEventListener('click', () => startCamera(ocrVideo, btnOcrCap));

    btnOcrCap.addEventListener('click', async () => {
        const blob = await captureImage(ocrVideo, ocrCanvas);
        await sendOcrApi(blob);
    });

    inputOcrUp.addEventListener('change', async (e) => {
        if (e.target.files.length > 0) {
            stopAllStreams();
            const blob = e.target.files[0];
            await sendOcrApi(blob);
        }
    });

    async function sendOcrApi(imageBlob) {
        showLoading();
        const formData = new FormData();
        formData.append('file', imageBlob, 'ocr.jpg');

        try {
            const res = await fetch('/api/ocr', { method: 'POST', body: formData });
            const data = await res.json();
            if (data.status === 'success') {
                renderOcrTable('ocr-result-box', data.data);
            } else {
                alert("번역 실패: " + (data.message || data.error));
            }
        } catch(err) {
            alert("서버 연결 오류");
        }
        hideLoading();
    }

    function renderOcrTable(boxId, data) {
        const box = document.getElementById(boxId);
        if (!data || data.length === 0) {
            box.innerHTML = '<span class="placeholder">검출된 텍스트가 없습니다.</span>';
            return;
        }

        let html = `
            <div class="table-container">
                <table class="ocr-table">
                    <thead>
                        <tr>
                            <th>원문 (English)</th>
                            <th>번역 (Korean)</th>
                            <th>신뢰도</th>
                        </tr>
                    </thead>
                    <tbody>
        `;

        data.forEach(item => {
            html += `
                <tr>
                    <td>${item.original}</td>
                    <td>${item.translated}</td>
                    <td>${(item.confidence * 100).toFixed(1)}%</td>
                </tr>
            `;
        });

        html += `
                    </tbody>
                </table>
            </div>
        `;
        box.innerHTML = html;
    }
});
