// ============================================================
// PCB Detection & Inspection App JS
// ============================================================

let currentInspectResult = null;
let currentDetectResult = null;
let currentInspectOriginalB64 = null;
let currentDetectOriginalB64 = null;
let webcamStream = null;
let isAutoScanning = false;
let autoScanInterval = null;

// Khởi chạy khi tài liệu tải xong
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initSystemInfo();
    initSliders();
    initUploadDropzones();
    initViewerToggles();
    initReferenceTab();
    initCameraControls();
});

// ------------------------------------------------------------
// 1. TABS MANAGEMENT
// ------------------------------------------------------------
function initTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            const targetId = btn.getAttribute('data-tab');
            const targetContent = document.getElementById(targetId);
            if (targetContent) targetContent.classList.add('active');

            // Quản lý webcam khi chuyển tab
            if (targetId !== 'tab-camera' && webcamStream) {
                stopWebcam();
            }
        });
    });
}

// ------------------------------------------------------------
// 2. LOAD SYSTEM INFO & STATUS
// ------------------------------------------------------------
async function initSystemInfo() {
    try {
        const res = await fetch('/api/info');
        const data = await res.json();
        if (data.status === 'success') {
            document.getElementById('headerModelName').textContent = data.model_name;
            document.getElementById('headerDeviceBadge').textContent = data.device_name || data.device.toUpperCase();
            
            const refBadge = document.getElementById('headerRefBadge');
            if (data.reference.loaded) {
                refBadge.innerHTML = `<span class="badge-dot active"></span> Reference: ${data.reference.components_count} linh kiện`;
            } else {
                refBadge.innerHTML = `<span class="badge-dot" style="background:#ef4444"></span> Reference: Chưa nạp`;
            }

            // Ghi nhận tổng số class
            if (document.getElementById('totalClassesCount')) {
                document.getElementById('totalClassesCount').textContent = data.total_classes;
            }
        }
    } catch (err) {
        console.error('Lỗi khi tải thông tin hệ thống:', err);
    }
}

// ------------------------------------------------------------
// 3. SLIDERS & PARAMETERS
// ------------------------------------------------------------
function initSliders() {
    const setupSlider = (sliderId, valId, format = val => val) => {
        const slider = document.getElementById(sliderId);
        const valSpan = document.getElementById(valId);
        if (slider && valSpan) {
            slider.addEventListener('input', () => {
                valSpan.textContent = format(slider.value);
            });
        }
    };

    setupSlider('inspectConf', 'inspectConfVal', v => `${Math.round(v * 100)}%`);
    setupSlider('inspectTol', 'inspectTolVal', v => `${(v * 100).toFixed(1)}%`);
    setupSlider('detectConf', 'detectConfVal', v => `${Math.round(v * 100)}%`);
}

// ------------------------------------------------------------
// 4. UPLOAD DROPZONES
// ------------------------------------------------------------
function initUploadDropzones() {
    setupDropzone('inspectDropzone', 'inspectFileInput', file => {
        handleInspectFileUpload(file);
    });

    setupDropzone('detectDropzone', 'detectFileInput', file => {
        handleDetectFileUpload(file);
    });

    setupDropzone('refUploadDropzone', 'refFileInput', file => {
        handleRefFileUpload(file);
    });

    // Nút dùng ảnh mẫu có sẵn 2.jpg
    const btnSampleInspect = document.getElementById('btnSampleInspect');
    if (btnSampleInspect) {
        btnSampleInspect.addEventListener('click', loadSampleForInspect);
    }

    const btnSampleDetect = document.getElementById('btnSampleDetect');
    if (btnSampleDetect) {
        btnSampleDetect.addEventListener('click', loadSampleForDetect);
    }
}

function setupDropzone(dropzoneId, inputId, onFileSelected) {
    const dropzone = document.getElementById(dropzoneId);
    const input = document.getElementById(inputId);
    if (!dropzone || !input) return;

    dropzone.addEventListener('click', () => input.click());

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            onFileSelected(e.dataTransfer.files[0]);
        }
    });

    input.addEventListener('change', () => {
        if (input.files && input.files.length > 0) {
            onFileSelected(input.files[0]);
        }
    });
}

// ------------------------------------------------------------
// 5. INSPECT WORKFLOW
// ------------------------------------------------------------
async function loadSampleForInspect() {
    try {
        setInspectLoading(true);
        const res = await fetch('/api/sample');
        const data = await res.json();
        if (data.status === 'success') {
            currentInspectOriginalB64 = data.image_base64;
            document.getElementById('inspectFileName').textContent = '2.jpg (Ảnh PCB tham chiếu có sẵn)';
            await runInspectRequest({ image_base64: data.image_base64 });
        } else {
            alert('Không tải được ảnh mẫu: ' + data.message);
        }
    } catch (err) {
        alert('Lỗi: ' + err.message);
    } finally {
        setInspectLoading(false);
    }
}

function handleInspectFileUpload(file) {
    const reader = new FileReader();
    reader.onload = async (e) => {
        currentInspectOriginalB64 = e.target.result;
        document.getElementById('inspectFileName').textContent = file.name;
        await runInspectRequest({ file: file });
    };
    reader.readAsDataURL(file);
}

async function runInspectRequest(payload) {
    setInspectLoading(true);
    try {
        const conf = document.getElementById('inspectConf').value;
        const tolerance = document.getElementById('inspectTol').value;
        const imgsz = document.getElementById('inspectImgsz').value;
        const autoCrop = document.getElementById('inspectAutoCrop').checked;

        let res;
        if (payload.file) {
            const formData = new FormData();
            formData.append('image', payload.file);
            formData.append('conf', conf);
            formData.append('tolerance_ratio', tolerance);
            formData.append('imgsz', imgsz);
            formData.append('auto_crop', autoCrop);
            res = await fetch('/api/inspect', { method: 'POST', body: formData });
        } else {
            res = await fetch('/api/inspect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_base64: payload.image_base64,
                    conf: parseFloat(conf),
                    tolerance_ratio: parseFloat(tolerance),
                    imgsz: parseInt(imgsz),
                    auto_crop: autoCrop
                })
            });
        }

        const data = await res.json();
        if (data.status === 'success') {
            currentInspectResult = data;
            renderInspectResults(data);
        } else {
            alert('Lỗi kiểm tra mạch: ' + (data.message || 'Không xác định'));
        }
    } catch (err) {
        alert('Lỗi gọi API: ' + err.message);
    } finally {
        setInspectLoading(false);
    }
}

function renderInspectResults(data) {
    // 1. Hero Verdict
    const heroCard = document.getElementById('heroVerdictCard');
    const heroTitle = document.getElementById('verdictTitle');
    const heroSubtitle = document.getElementById('verdictSubtitle');
    const heroIcon = document.getElementById('verdictIcon');

    heroCard.classList.remove('ok', 'faulty');
    if (data.is_ok) {
        heroCard.classList.add('ok');
        heroTitle.textContent = 'MẠCH ĐẠT CHUẨN (OK)';
        heroSubtitle.textContent = 'Tất cả linh kiện đều khớp chính xác với mạch mẫu tham chiếu.';
        heroIcon.innerHTML = `<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
    } else {
        heroCard.classList.add('faulty');
        heroTitle.textContent = 'MẠCH BỊ LỖI (FAULTY)';
        const parts = [];
        if (data.stats.missing > 0) parts.push(`Thiếu ${data.stats.missing} linh kiện`);
        if (data.stats.extra > 0) parts.push(`${data.stats.extra} linh kiện sai khác`);
        heroSubtitle.textContent = `Phát hiện: ${parts.join(', ') || 'Sai khác linh kiện so với mạch mẫu tham chiếu.'}`;
        heroIcon.innerHTML = `<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>`;
    }

    // 2. Metrics
    document.getElementById('metricMatched').textContent = data.stats.matched;
    document.getElementById('metricMissing').textContent = data.stats.missing;
    document.getElementById('metricExtra').textContent = data.stats.extra;
    document.getElementById('metricTotalDet').textContent = data.stats.detected_count;
    document.getElementById('metricAligned').textContent = data.aligned ? (data.alignment_method || 'Có') : 'Không';
    document.getElementById('metricLatency').textContent = `${data.latency_ms} ms`;

    // 3. Defect Alert Box
    const alertBox = document.getElementById('defectAlertBox');
    const defectList = document.getElementById('defectList');
    defectList.innerHTML = '';
    if (!data.is_ok) {
        alertBox.style.display = 'block';
        if (data.stats.missing > 0) {
            const missingNames = data.missing_components_details.map(m => `Class: "${m.class}" (Vị trí ref: x=${m.ref_x}, y=${m.ref_y})`).join(', ');
            defectList.innerHTML += `<li><strong>Thiếu ${data.stats.missing} linh kiện:</strong> ${missingNames}</li>`;
        }
        if (data.stats.extra > 0) {
            const extraNames = data.stats.extra_components.join(', ');
            defectList.innerHTML += `<li><strong>Sai khác ${data.stats.extra} linh kiện:</strong> ${extraNames}</li>`;
        }
        if (!data.aligned) {
            defectList.innerHTML += `<li><strong>Ghi chú:</strong> Không thể căn chỉnh góc nhìn tương đồng với mạch tham chiếu.</li>`;
        }
    } else {
        alertBox.style.display = 'none';
    }

    // 4. Viewer Image
    switchInspectView('annotated');

    // 5. Populate Table
    populateInspectTable(data.detections, data.missing_components_details);

    // Hiển thị phần kết quả
    document.getElementById('inspectPlaceholder').style.display = 'none';
    document.getElementById('inspectResultWrapper').style.display = 'block';
}

function populateInspectTable(detections, missingDetails) {
    const tbody = document.getElementById('inspectTableBody');
    tbody.innerHTML = '';

    // Linh kiện thiếu (Đưa lên trước nếu có)
    if (missingDetails && missingDetails.length > 0) {
        missingDetails.forEach((m, idx) => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><span class="tag-status missing">THIẾU #${m.ref_id}</span></td>
                <td><strong style="color:var(--accent-purple)">${m.class}</strong></td>
                <td>--</td>
                <td>x: ${m.ref_x}, y: ${m.ref_y}</td>
                <td><span class="tag-status missing">Không phát hiện</span></td>
            `;
            tbody.appendChild(tr);
        });
    }

    // Linh kiện nhận diện được
    if (detections && detections.length > 0) {
        detections.forEach((d) => {
            const tr = document.createElement('tr');
            let statusTag = '';
            if (d.status === 'OK') {
                statusTag = `<span class="tag-status ok">KHỚP REF #${d.matched_ref_id}</span>`;
            } else if (d.status === 'EXTRA') {
                statusTag = `<span class="tag-status extra">SAI KHÁC</span>`;
            } else {
                statusTag = `<span class="tag-status low">ĐỘ TIN CẬY THẤP</span>`;
            }

            tr.innerHTML = `
                <td>#${d.id}</td>
                <td><strong>${d.class}</strong></td>
                <td>${(d.confidence * 100).toFixed(1)}%</td>
                <td>cx: ${d.cx}, cy: ${d.cy}</td>
                <td>${statusTag}</td>
            `;
            tbody.appendChild(tr);
        });
    }
}

function setInspectLoading(loading) {
    const btn = document.getElementById('btnRunInspect');
    const sp = document.getElementById('inspectSpinner');
    if (btn) btn.disabled = loading;
    if (sp) sp.style.display = loading ? 'inline-block' : 'none';
}

// ------------------------------------------------------------
// 6. DETECT WORKFLOW (YOLO ONLY)
// ------------------------------------------------------------
async function loadSampleForDetect() {
    try {
        setDetectLoading(true);
        const res = await fetch('/api/sample');
        const data = await res.json();
        if (data.status === 'success') {
            currentDetectOriginalB64 = data.image_base64;
            document.getElementById('detectFileName').textContent = '2.jpg (Ảnh PCB có sẵn)';
            await runDetectRequest({ image_base64: data.image_base64 });
        }
    } catch (err) {
        alert('Lỗi: ' + err.message);
    } finally {
        setDetectLoading(false);
    }
}

function handleDetectFileUpload(file) {
    const reader = new FileReader();
    reader.onload = async (e) => {
        currentDetectOriginalB64 = e.target.result;
        document.getElementById('detectFileName').textContent = file.name;
        await runDetectRequest({ file: file });
    };
    reader.readAsDataURL(file);
}

async function runDetectRequest(payload) {
    setDetectLoading(true);
    try {
        const conf = document.getElementById('detectConf').value;
        const imgsz = document.getElementById('detectImgsz').value;

        let res;
        if (payload.file) {
            const formData = new FormData();
            formData.append('image', payload.file);
            formData.append('conf', conf);
            formData.append('imgsz', imgsz);
            res = await fetch('/api/detect', { method: 'POST', body: formData });
        } else {
            res = await fetch('/api/detect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_base64: payload.image_base64,
                    conf: parseFloat(conf),
                    imgsz: parseInt(imgsz)
                })
            });
        }

        const data = await res.json();
        if (data.status === 'success') {
            currentDetectResult = data;
            renderDetectResults(data);
        } else {
            alert('Lỗi nhận diện: ' + (data.message || 'Không xác định'));
        }
    } catch (err) {
        alert('Lỗi gọi API: ' + err.message);
    } finally {
        setDetectLoading(false);
    }
}

function renderDetectResults(data) {
    document.getElementById('detectTotalCount').textContent = data.total_count;
    document.getElementById('detectLatency').textContent = `${data.latency_ms} ms`;
    document.getElementById('detectImageSize').textContent = `${data.image_size.width} x ${data.image_size.height} px`;

    // Chip đếm số lượng class
    const chipContainer = document.getElementById('detectClassChips');
    chipContainer.innerHTML = '';
    for (const [cls, count] of Object.entries(data.class_counts)) {
        const chip = document.createElement('div');
        chip.className = 'badge';
        chip.style.padding = '0.4rem 0.85rem';
        chip.innerHTML = `<strong>${cls}</strong>: <span style="color:var(--accent-cyan)">${count}</span>`;
        chipContainer.appendChild(chip);
    }

    // Ảnh hiển thị
    document.getElementById('detectPreviewImg').src = data.annotated_image;

    // Bảng chi tiết
    const tbody = document.getElementById('detectTableBody');
    tbody.innerHTML = '';
    data.detections.forEach(d => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>#${d.id}</td>
            <td><strong>${d.class}</strong></td>
            <td>${(d.confidence * 100).toFixed(1)}%</td>
            <td>[${d.x1}, ${d.y1}, ${d.x2}, ${d.y2}]</td>
            <td>${d.width} x ${d.height}</td>
        `;
        tbody.appendChild(tr);
    });

    document.getElementById('detectPlaceholder').style.display = 'none';
    document.getElementById('detectResultWrapper').style.display = 'block';
}

function setDetectLoading(loading) {
    const btn = document.getElementById('btnRunDetect');
    const sp = document.getElementById('detectSpinner');
    if (btn) btn.disabled = loading;
    if (sp) sp.style.display = loading ? 'inline-block' : 'none';
}

// ------------------------------------------------------------
// 7. VIEWER TOGGLE (ORIGINAL / ANNOTATED / CROP)
// ------------------------------------------------------------
function initViewerToggles() {
    document.querySelectorAll('.btn-view-toggle').forEach(btn => {
        btn.addEventListener('click', () => {
            const targetMode = btn.getAttribute('data-mode');
            document.querySelectorAll('.btn-view-toggle').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            switchInspectView(targetMode);
        });
    });
}

function switchInspectView(mode) {
    const imgEl = document.getElementById('inspectPreviewImg');
    if (!currentInspectResult) return;

    if (mode === 'annotated') {
        imgEl.src = currentInspectResult.annotated_image;
    } else if (mode === 'original') {
        imgEl.src = currentInspectOriginalB64 || currentInspectResult.annotated_image;
    } else if (mode === 'crop') {
        imgEl.src = currentInspectResult.board_crop_image || currentInspectResult.annotated_image;
    }
}

// ------------------------------------------------------------
// 8. LIVE CAMERA & IP STREAM WORKFLOW
// ------------------------------------------------------------
let currentCameraSource = 'ip'; // 'ip' hoặc 'webcam'
let cameraStatusPollTimer = null;

// Khởi tạo các sự kiện Camera
function initCameraControls() {
    const btnSourceIpCam = document.getElementById('btnSourceIpCam');
    const btnSourceWebcam = document.getElementById('btnSourceWebcam');
    const btnMainScan = document.getElementById('btnMainScan');
    const btnAutoScan = document.getElementById('btnAutoScan');
    const btnConnectIpCam = document.getElementById('btnConnectIpCam');
    const btnDisconnectIpCam = document.getElementById('btnDisconnectIpCam');

    // Chuyển đổi nguồn camera
    if (btnSourceIpCam && btnSourceWebcam) {
        btnSourceIpCam.addEventListener('click', () => switchCameraSource('ip'));
        btnSourceWebcam.addEventListener('click', () => switchCameraSource('webcam'));
    }

    // Kết nối / ngắt kết nối IP Cam
    if (btnConnectIpCam) {
        btnConnectIpCam.addEventListener('click', connectBackendCamera);
    }
    if (btnDisconnectIpCam) {
        btnDisconnectIpCam.addEventListener('click', disconnectBackendCamera);
    }

    // Nút Quét Ngay
    if (btnMainScan) {
        btnMainScan.addEventListener('click', () => triggerCameraScan());
    }

    // Nút Quét Tự Động
    if (btnAutoScan) {
        btnAutoScan.addEventListener('click', toggleCameraAutoScan);
    }

    // Lắng nghe phím tắt 'S' trên toàn bộ trang
    document.addEventListener('keydown', (e) => {
        if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.isContentEditable)) {
            return;
        }
        if (e.key === 's' || e.key === 'S') {
            e.preventDefault();
            // Nếu tab camera đang mở, quét camera
            const cameraTab = document.getElementById('tab-camera');
            if (cameraTab && cameraTab.classList.contains('active')) {
                triggerCameraScan();
            }
        }
    });

    // Bắt đầu thăm dò trạng thái camera backend
    startCameraStatusPolling();
}

function switchCameraSource(src) {
    currentCameraSource = src;
    const btnIp = document.getElementById('btnSourceIpCam');
    const btnCam = document.getElementById('btnSourceWebcam');
    const ipStreamImg = document.getElementById('backendCameraStream');
    const webcamVideo = document.getElementById('webcamVideo');
    const panelIpConfig = document.getElementById('panelIpConfig');

    if (src === 'ip') {
        btnIp.classList.add('active');
        btnCam.classList.remove('active');
        ipStreamImg.style.display = 'block';
        webcamVideo.style.display = 'none';
        if (panelIpConfig) panelIpConfig.style.display = 'block';
        stopBrowserWebcam();
        document.getElementById('cameraOverlayStatus').textContent = 'Camera IP / Backend';
        addCameraLog('Đã chuyển sang nguồn Camera IP / Backend OpenCV.', 'info');
    } else {
        btnCam.classList.add('active');
        btnIp.classList.remove('active');
        ipStreamImg.style.display = 'none';
        webcamVideo.style.display = 'block';
        if (panelIpConfig) panelIpConfig.style.display = 'none';
        startBrowserWebcam();
        document.getElementById('cameraOverlayStatus').textContent = 'Webcam Trình Duyệt';
        addCameraLog('Đã chuyển sang nguồn Webcam máy tính / trình duyệt.', 'info');
    }
}

async function connectBackendCamera() {
    const urlInput = document.getElementById('ipCameraUrl');
    const source = urlInput ? urlInput.value.trim() : 'http://192.168.0.190:8080/video';
    const btn = document.getElementById('btnConnectIpCam');
    if (btn) {
        btn.disabled = true;
        btn.textContent = 'Đang kết nối...';
    }

    addCameraLog(`Đang kết nối camera tới nguồn: "${source}"...`, 'info');

    try {
        const res = await fetch('/api/camera/connect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source: source })
        });
        const data = await res.json();
        if (data.status === 'success') {
            addCameraLog(`Kết nối thành công tới ${source}. Luồng video đang truyền trực tiếp.`, 'ok');
            document.getElementById('cameraOverlayStatus').textContent = `Kết nối: ${source}`;
            // Làm mới nguồn ảnh stream
            const streamImg = document.getElementById('backendCameraStream');
            if (streamImg) {
                streamImg.src = `/api/camera/stream?t=${Date.now()}`;
            }
        } else {
            addCameraLog(`Kết nối thất bại: ${data.message}`, 'faulty');
            alert(`Lỗi kết nối camera:\n${data.message}`);
        }
    } catch (err) {
        addCameraLog(`Lỗi mạng khi kết nối camera: ${err.message}`, 'faulty');
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.textContent = 'Kết Nối';
        }
    }
}

async function disconnectBackendCamera() {
    try {
        const res = await fetch('/api/camera/disconnect', { method: 'POST' });
        const data = await res.json();
        addCameraLog('Đã ngắt kết nối camera backend.', 'info');
        document.getElementById('cameraOverlayStatus').textContent = 'Đã ngắt kết nối';
        const streamImg = document.getElementById('backendCameraStream');
        if (streamImg) {
            streamImg.src = `/api/camera/stream?t=${Date.now()}`;
        }
    } catch (err) {
        console.error(err);
    }
}

async function startBrowserWebcam() {
    try {
        const video = document.getElementById('webcamVideo');
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert('Trình duyệt không hỗ trợ getUserMedia.');
            return;
        }
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 } }
        });
        webcamStream = stream;
        video.srcObject = stream;
        addCameraLog('Webcam trình duyệt đã khởi động thành công.', 'ok');
    } catch (err) {
        addCameraLog(`Lỗi mở webcam: ${err.message}`, 'faulty');
        alert(`Không mở được webcam: ${err.message}\nHãy kiểm tra quyền camera.`);
    }
}

function stopBrowserWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(t => t.stop());
        webcamStream = null;
    }
    const video = document.getElementById('webcamVideo');
    if (video) video.srcObject = null;
}

// Kích hoạt quét PCB (qua nút bấm hoặc phím S)
async function triggerCameraScan() {
    const feedBox = document.getElementById('cameraFeedBox');
    if (feedBox) {
        feedBox.classList.add('flash-scan');
        setTimeout(() => feedBox.classList.remove('flash-scan'), 350);
    }

    const scanBtn = document.getElementById('btnMainScan');
    if (scanBtn) scanBtn.disabled = true;

    try {
        if (currentCameraSource === 'ip') {
            // Quét từ Camera Backend
            const res = await fetch('/api/camera/scan', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ conf: 0.25, imgsz: 1536, tolerance_ratio: 0.045 })
            });
            const data = await res.json();
            if (data.status === 'success') {
                renderCameraScanResult(data);
            } else {
                addCameraLog(`Lỗi quét camera: ${data.message}`, 'faulty');
            }
        } else {
            // Quét từ Webcam trình duyệt
            const video = document.getElementById('webcamVideo');
            if (!video || !webcamStream) {
                addCameraLog('Webcam trình duyệt chưa được bật. Vui lòng bật webcam trước.', 'faulty');
                return;
            }
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth || 1280;
            canvas.height = video.videoHeight || 720;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const b64 = canvas.toDataURL('image/jpeg', 0.9);

            const res = await fetch('/api/inspect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_base64: b64, conf: 0.25, tolerance_ratio: 0.045, auto_crop: true })
            });
            const data = await res.json();
            if (data.status === 'success') {
                renderCameraScanResult(data);
            } else {
                addCameraLog(`Lỗi quét: ${data.message}`, 'faulty');
            }
        }
    } catch (err) {
        addCameraLog(`Lỗi gọi API: ${err.message}`, 'faulty');
    } finally {
        if (scanBtn) scanBtn.disabled = false;
    }
}

function toggleCameraAutoScan() {
    isAutoScanning = !isAutoScanning;
    const btn = document.getElementById('btnAutoScan');
    const intervalSelect = document.getElementById('autoScanInterval');
    const intervalMs = parseInt(intervalSelect ? intervalSelect.value : 3000, 10);

    if (isAutoScanning) {
        btn.textContent = '⏹ Dừng Quét Tự Động';
        btn.style.background = '#ef4444';
        btn.style.color = '#fff';
        addCameraLog(`Đã BẬT chế độ quét tự động (chu kỳ ${intervalMs / 1000}s).`, 'info');
        triggerCameraScan();
        autoScanInterval = setInterval(() => {
            triggerCameraScan();
        }, intervalMs);
    } else {
        btn.textContent = '🔄 Quét Tự Động: Tắt';
        btn.style.background = '';
        btn.style.color = '';
        addCameraLog('Đã TẮT chế độ quét tự động.', 'info');
        if (autoScanInterval) clearInterval(autoScanInterval);
    }
}

function renderCameraScanResult(data) {
    const card = document.getElementById('camVerdictCard');
    const title = document.getElementById('camVerdictTitle');
    const subtitle = document.getElementById('camVerdictSubtitle');
    const icon = document.getElementById('camVerdictIcon');

    card.classList.remove('ok', 'faulty');
    if (data.is_ok) {
        card.classList.add('ok');
        title.textContent = 'M\u1ea0CH \u0110\u1ea0T CHU\u1ea8N (OK)';
        subtitle.textContent = `Kh\u1edbp ${data.stats.matched}/${data.stats.reference_count} linh ki\u1ec7n v\u1edbi m\u1ea1ch tham chi\u1ebfu.`;
        icon.innerHTML = `<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
    } else {
        card.classList.add('faulty');
        title.textContent = 'M\u1ea0CH B\u1eca L\u1ed6I (FAULTY)';
        const parts = [];
        if (data.stats.missing > 0) parts.push(`Thi\u1ebfu ${data.stats.missing} linh ki\u1ec7n`);
        if (data.stats.extra > 0) parts.push(`${data.stats.extra} sai kh\u00e1c`);
        subtitle.textContent = `Ph\u00e1t hi\u1ec7n l\u1ed7i: ${parts.join(', ') || 'Kh\u00f4ng kh\u1edbp v\u1ecb tr\u00ed linh ki\u1ec7n'}.`;
        icon.innerHTML = `<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>`;
    }

    document.getElementById('camMatched').textContent = data.stats.matched;
    document.getElementById('camMissing').textContent = data.stats.missing;
    document.getElementById('camExtra').textContent = data.stats.extra;
    document.getElementById('camLatency').textContent = `${data.latency_ms} ms`;
    document.getElementById('camAnnotatedImg').src = data.annotated_image;

    // Hi\u1ec3n th\u1ecb ph\u01b0\u01a1ng ph\u00e1p crop
    const camCropStatus = document.getElementById('camCropStatus');
    if (camCropStatus && data.crop_method) {
        camCropStatus.textContent = data.crop_method === 'perspective' ? 'Perspective Crop' :
                                    data.crop_method === 'roi' ? 'ROI Crop' : 'Full Frame';
    }

    const timeStr = new Date().toLocaleTimeString('vi-VN');
    document.getElementById('camScanTimeBadge').textContent = `L\u00fac ${timeStr}`;

    // Defect Alert Box
    const camDefectBox = document.getElementById('camDefectAlertBox');
    const camDefectList = document.getElementById('camDefectList');
    if (camDefectList) camDefectList.innerHTML = '';
    if (!data.is_ok && camDefectBox) {
        camDefectBox.style.display = 'block';
        if (data.stats.missing > 0 && data.missing_components_details && data.missing_components_details.length > 0) {
            const missingNames = data.missing_components_details
                .map(m => `<strong>${m.class}</strong> (Ref#${m.ref_id})`)
                .join(', ');
            camDefectList.innerHTML += `<li>Thi\u1ebfu ${data.stats.missing} linh ki\u1ec7n: ${missingNames}</li>`;
        }
        if (data.stats.extra > 0 && data.stats.extra_components) {
            const extraNames = data.stats.extra_components.join(', ');
            camDefectList.innerHTML += `<li>Sai kh\u00e1c ${data.stats.extra} linh ki\u1ec7n: ${extraNames}</li>`;
        }
        if (!data.aligned) {
            camDefectList.innerHTML += `<li><strong>Ghi ch\u00fa:</strong> Kh\u00f4ng th\u1ec3 c\u0103n ch\u1ec9nh g\u00f3c nh\u00ecn t\u01b0\u01a1ng \u0111\u1ed3ng v\u1edbi m\u1ea1ch tham chi\u1ebfu.</li>`;
        }
    } else if (camDefectBox) {
        camDefectBox.style.display = 'none';
    }

    // B\u1ea3ng chi ti\u1ebft linh ki\u1ec7n camera
    const camTablePanel = document.getElementById('camDetailTablePanel');
    const camTableBody = document.getElementById('camDetailTableBody');
    if (camTableBody && data.detections && data.detections.length > 0) {
        camTableBody.innerHTML = '';

        // Linh ki\u1ec7n thi\u1ebfu l\u00ean tr\u01b0\u1edbc
        if (data.missing_components_details && data.missing_components_details.length > 0) {
            data.missing_components_details.forEach(m => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><span class="tag-status missing">THI\u1ebeU #${m.ref_id}</span></td>
                    <td><strong style="color:var(--accent-purple)">${m.class}</strong></td>
                    <td>--</td>
                    <td>x: ${Math.round(m.ref_x)}, y: ${Math.round(m.ref_y)}</td>
                    <td><span class="tag-status missing">Kh\u00f4ng ph\u00e1t hi\u1ec7n</span></td>
                `;
                camTableBody.appendChild(tr);
            });
        }

        // Linh ki\u1ec7n ph\u00e1t hi\u1ec7n \u0111\u01b0\u1ee3c
        data.detections.forEach(d => {
            const tr = document.createElement('tr');
            let statusTag = '';
            if (d.status === 'OK') {
                statusTag = `<span class="tag-status ok">KH\u1ee4P REF #${d.matched_ref_id}</span>`;
            } else if (d.status === 'EXTRA') {
                statusTag = `<span class="tag-status extra">SAI KH\u00c1C</span>`;
            } else {
                statusTag = `<span class="tag-status low">\u0110\u1ed8 TIN C\u1eacY TH\u1ea4P</span>`;
            }
            tr.innerHTML = `
                <td>#${d.id}</td>
                <td><strong>${d.class}</strong></td>
                <td>${(d.confidence * 100).toFixed(1)}%</td>
                <td>cx: ${Math.round(d.cx)}, cy: ${Math.round(d.cy)}</td>
                <td>${statusTag}</td>
            `;
            camTableBody.appendChild(tr);
        });

        if (camTablePanel) camTablePanel.style.display = 'block';
    } else if (camTablePanel) {
        camTablePanel.style.display = 'none';
    }

    // Ghi nh\u1eadt k\u00fd Live Log thay th\u1ebf terminal
    if (data.is_ok) {
        addCameraLog(`M\u1ea0CH \u0110\u1ea0T CHU\u1ea8N (OK) \u2014 Kh\u1edbp: ${data.stats.matched} | Thi\u1ebfu: 0 | Sai kh\u00e1c: 0 (${data.latency_ms}ms)`, 'ok');
    } else {
        const errDetails = [];
        if (data.stats.missing > 0) errDetails.push(`Thi\u1ebfu: ${data.stats.missing}`);
        if (data.stats.extra > 0) errDetails.push(`Sai kh\u00e1c: ${data.stats.extra}`);
        addCameraLog(`M\u1ea0CH B\u1eca L\u1ed6I (FAULTY) \u2014 ${errDetails.join(' | ')} (${data.latency_ms}ms)`, 'faulty');
    }
}
function addCameraLog(msg, type = 'info') {
    const logBox = document.getElementById('cameraLogBox');
    if (!logBox) return;
    const entry = document.createElement('div');
    entry.className = 'camera-log-entry';
    const now = new Date().toLocaleTimeString('vi-VN');

    let classType = 'camera-log-info';
    if (type === 'ok') classType = 'camera-log-ok';
    if (type === 'faulty') classType = 'camera-log-faulty';

    entry.innerHTML = `
        <span class="camera-log-time">[${now}]</span>
        <span class="${classType}">${msg}</span>
    `;
    logBox.prepend(entry);
    // Giữ tối đa 50 dòng log
    while (logBox.children.length > 50) {
        logBox.removeChild(logBox.lastChild);
    }
}

function startCameraStatusPolling() {
    if (cameraStatusPollTimer) clearInterval(cameraStatusPollTimer);
    cameraStatusPollTimer = setInterval(async () => {
        const camTab = document.getElementById('tab-camera');
        if (!camTab || !camTab.classList.contains('active')) return;
        if (currentCameraSource !== 'ip') return;

        try {
            const res = await fetch('/api/camera/status');
            const data = await res.json();
            if (data.status === 'success' && data.camera) {
                const fpsBadge = document.getElementById('cameraFpsBadge');
                if (fpsBadge) {
                    if (data.camera.connected) {
                        fpsBadge.textContent = `${data.camera.fps} FPS`;
                        fpsBadge.style.color = '#10b981';
                        document.getElementById('camLiveDot').style.background = '#10b981';
                    } else {
                        fpsBadge.textContent = 'Chưa kết nối';
                        fpsBadge.style.color = '#ef4444';
                        document.getElementById('camLiveDot').style.background = '#ef4444';
                    }
                }
            }
        } catch (e) {
            // bỏ qua lỗi polling
        }
    }, 2000);
}

// ------------------------------------------------------------
// 9. REFERENCE TAB
// ------------------------------------------------------------
async function initReferenceTab() {
    try {
        const res = await fetch('/api/reference');
        const data = await res.json();
        if (data.status === 'success') {
            document.getElementById('refCompCount').textContent = `${data.components_count} linh kiện`;
            if (data.detected_image_base64) {
                document.getElementById('refDetectedImg').src = data.detected_image_base64;
            }

            const tbody = document.getElementById('refTableBody');
            tbody.innerHTML = '';
            data.components.forEach(c => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>#${c.id}</td>
                    <td><strong>${c.class}</strong></td>
                    <td>${(c.confidence_reference * 100).toFixed(1)}%</td>
                    <td>x: ${c.x}, y: ${c.y}</td>
                `;
                tbody.appendChild(tr);
            });
        }
    } catch (err) {
        console.error('Lỗi khi tải reference:', err);
    }
}

function handleRefFileUpload(file) {
    const reader = new FileReader();
    reader.onload = async (e) => {
        document.getElementById('refFileName').textContent = file.name;
        if (confirm(`Bạn có chắc muốn tạo lại Reference PCB từ ảnh "${file.name}" không?`)) {
            await uploadNewReference(file);
        }
    };
    reader.readAsDataURL(file);
}

async function uploadNewReference(file) {
    const formData = new FormData();
    formData.append('image', file);
    try {
        const res = await fetch('/api/reference/update', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        if (data.status === 'success') {
            alert(data.message);
            initReferenceTab();
            initSystemInfo();
        } else {
            alert('Lỗi tạo reference: ' + data.message);
        }
    } catch (err) {
        alert('Lỗi gọi API: ' + err.message);
    }
}

// ------------------------------------------------------------
// 10. COPY CODE UTILITY
// ------------------------------------------------------------
function copyCode(btn, codeId) {
    const codeEl = document.getElementById(codeId);
    if (!codeEl) return;
    navigator.clipboard.writeText(codeEl.textContent).then(() => {
        const oldText = btn.textContent;
        btn.textContent = 'Đã sao chép!';
        setTimeout(() => btn.textContent = oldText, 2000);
    });
}
window.copyCode = copyCode;
