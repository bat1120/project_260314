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
    // 1.5. Posture WebSockets Tab Logic
    // ----------------------------------------------------------------
    const wsVideo = document.getElementById('ws-video');
    const wsCanvas = document.getElementById('ws-canvas');
    const wsStartBtn = document.getElementById('ws-start-cam');
    const wsStopBtn = document.getElementById('ws-stop-cam');
    const wsCalibrateBtn = document.getElementById('ws-calibrate');
    const wsStatusText = document.getElementById('ws-status-text');
    const wsWarningMsg = document.getElementById('ws-warning-msg');
    const wsStatusCircle = document.getElementById('ws-status-circle');

    let postureWs = null;
    let wsInterval = null;
    let wsWarningStartTime = null;
    let audioContext = null;

    function playBeep() {
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        oscillator.type = 'sine';
        oscillator.frequency.value = 1000; // 1000Hz
        gainNode.gain.setValueAtTime(0.5, audioContext.currentTime);
        oscillator.start();
        oscillator.stop(audioContext.currentTime + 0.5); // Play for 0.5s
    }

    wsStartBtn.addEventListener('click', async () => {
        try {
            await startCamera(wsVideo, wsStopBtn);
            wsStartBtn.disabled = true;
            wsCalibrateBtn.disabled = false;
            wsStatusText.textContent = "연결 중...";
            
            // Connect to WebSocket
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            postureWs = new WebSocket(`${protocol}//${window.location.host}/ws/posture`);
            
            postureWs.onopen = () => {
                wsStatusText.textContent = "자세 모니터링 시작 (Not Calibrated)";
                // Send frames every 200ms (5 FPS)
                wsInterval = setInterval(() => {
                    const ctx = wsCanvas.getContext('2d');
                    wsCanvas.width = 640;
                    wsCanvas.height = 480;
                    ctx.drawImage(wsVideo, 0, 0, 640, 480);
                    // Send to WS
                    const base64Data = wsCanvas.toDataURL('image/jpeg', 0.5);
                    if(postureWs.readyState === WebSocket.OPEN) {
                        postureWs.send(JSON.stringify({ image: base64Data }));
                    }
                }, 200);
            };

            postureWs.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === 'result') {
                    wsStatusText.textContent = data.status_text;
                    wsStatusText.style.color = data.color;
                    wsStatusCircle.style.backgroundColor = data.color;
                    wsWarningMsg.textContent = data.warning_msg || "";
                    
                    if (data.is_bad_posture) {
                        if (!wsWarningStartTime) {
                            wsWarningStartTime = Date.now();
                        } else if (Date.now() - wsWarningStartTime > 3000) { // 3 seconds warning delay
                            playBeep();
                            wsWarningStartTime = Date.now(); // reset timer after playing sound to play every 3 sec
                        }
                    } else {
                        wsWarningStartTime = null;
                    }

                    // Save loose calibration data if user clicks Calibrate
                    if (data.calib_data) {
                        window.tempCalibData = data.calib_data;
                    }
                } else if (data.type === 'info') {
                    console.log(data.message);
                }
            };

            postureWs.onclose = () => {
                stopWsMonitor();
            };

        } catch (err) {
            alert("카메라 시작 실패");
        }
    });

    wsStopBtn.addEventListener('click', stopWsMonitor);

    function stopWsMonitor() {
        if(wsInterval) clearInterval(wsInterval);
        if(postureWs) {
            postureWs.close();
            postureWs = null;
        }
        stopAllStreams();
        wsStartBtn.disabled = false;
        wsStopBtn.disabled = true;
        wsCalibrateBtn.disabled = true;
        wsStatusText.textContent = "대기 중";
        wsStatusText.style.color = "#f8fafc";
        wsStatusCircle.style.backgroundColor = "#94a3b8";
        wsWarningMsg.textContent = "";
        wsWarningStartTime = null;
    }

    wsCalibrateBtn.addEventListener('click', () => {
        if(postureWs && postureWs.readyState === WebSocket.OPEN && window.tempCalibData) {
            postureWs.send(JSON.stringify({
                action: 'calibrate',
                data: window.tempCalibData
            }));
            alert("현재 자세가 올바른 자세 기준으로 등록되었습니다.");
            
            // Audio context must be resumed/created after a user gesture
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }
        } else {
            alert("아직 자세 정보가 로드되지 않았습니다. 잠시 후 캘리브레이션 버튼을 눌러주세요.");
        }
    });

    // Handle Tab switching (graceful closing of ws)
    const existingTabClickLogic = [];
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            if(btn.getAttribute('data-tab') !== 'posture_ws') {
                stopWsMonitor();
            }
        });
    });

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

    // ----------------------------------------------------------------
    // 5. SENTIMENT Tab Logic
    // ----------------------------------------------------------------
    const sentimentText = document.getElementById('sentiment-text');
    const sentimentBtn = document.getElementById('sentiment-analyze-btn');
    const sentimentResultBox = document.getElementById('sentiment-result-box');
    const sentimentBatchText = document.getElementById('sentiment-batch-text');
    const sentimentBatchBtn = document.getElementById('sentiment-batch-btn');
    const sentimentBatchResult = document.getElementById('sentiment-batch-result');

    function getSentimentEmoji(sentiment, stars) {
        if (stars >= 5) return '😄';
        if (stars >= 4) return '🙂';
        if (stars === 3) return '😐';
        if (stars === 2) return '😟';
        return '😢';
    }

    function getSentimentColor(sentiment) {
        if (sentiment === '긍정') return '#34d399';
        if (sentiment === '중립') return '#fbbf24';
        return '#f87171';
    }

    function renderStars(count) {
        return '★'.repeat(count) + '☆'.repeat(5 - count);
    }

    sentimentBtn.addEventListener('click', async () => {
        const text = sentimentText.value.trim();
        if (!text) {
            alert('텍스트를 입력해주세요.');
            return;
        }

        showLoading();
        try {
            const res = await fetch('/api/sentiment', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });
            const data = await res.json();

            if (data.status === 'success') {
                const r = data.result;
                const emoji = getSentimentEmoji(r.sentiment, r.stars);
                const color = getSentimentColor(r.sentiment);

                sentimentResultBox.innerHTML = `
                    <div class="sentiment-card">
                        <div class="sentiment-emoji">${emoji}</div>
                        <div class="sentiment-label" style="color: ${color};">${r.sentiment}</div>
                        <div class="sentiment-stars" style="color: ${color};">${renderStars(r.stars)}</div>
                        <div class="sentiment-conf">신뢰도: ${(r.confidence * 100).toFixed(1)}%</div>
                    </div>
                `;
            } else {
                sentimentResultBox.innerHTML = `<span class="placeholder">⚠️ ${data.error}</span>`;
            }
        } catch (err) {
            alert('서버 연결 오류');
        }
        hideLoading();
    });

    sentimentBatchBtn.addEventListener('click', async () => {
        const raw = sentimentBatchText.value.trim();
        if (!raw) {
            alert('텍스트를 입력해주세요.');
            return;
        }

        const texts = raw.split('\n').map(t => t.trim()).filter(t => t.length > 0);
        if (texts.length === 0) {
            alert('분석할 텍스트가 없습니다.');
            return;
        }

        showLoading();
        try {
            const res = await fetch('/api/sentiment/batch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(texts)
            });
            const data = await res.json();

            if (data.status === 'success') {
                let html = `<div class="table-container"><table class="ocr-table">
                    <thead><tr>
                        <th>텍스트</th>
                        <th>감정</th>
                        <th>별점</th>
                        <th>신뢰도</th>
                    </tr></thead><tbody>`;

                data.results.forEach(item => {
                    const color = getSentimentColor(item.sentiment);
                    const emoji = getSentimentEmoji(item.sentiment, item.stars);
                    html += `<tr>
                        <td>${item.text}</td>
                        <td style="color:${color}; font-weight:700;">${emoji} ${item.sentiment}</td>
                        <td style="color:${color};">${renderStars(item.stars)}</td>
                        <td>${(item.confidence * 100).toFixed(1)}%</td>
                    </tr>`;
                });

                html += '</tbody></table></div>';
                sentimentBatchResult.innerHTML = html;
            } else {
                sentimentBatchResult.innerHTML = `<span class="placeholder">⚠️ ${data.error}</span>`;
            }
        } catch (err) {
            alert('서버 연결 오류');
        }
        hideLoading();
    });
});
