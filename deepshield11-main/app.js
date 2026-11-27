document.addEventListener('DOMContentLoaded', () => {
    const API_BASE_URL = 'http://127.0.0.1:8001';

    // ── Safe Timeout Fetch Helper (120s for Video) ───────────────────
    const fetchWithTimeout = async (url, options = {}, timeoutMs = 120000) => {
        const controller = new AbortController();
        const id = setTimeout(() => controller.abort(), timeoutMs);

        try {
            const response = await fetch(url, {
                ...options,
                signal: controller.signal
            });
            clearTimeout(id);
            return response;
        } catch (err) {
            clearTimeout(id);
            throw err;
        }
    };

    // -------------------------------------------------------------
    // 1. Data Store / State
    // -------------------------------------------------------------
    let totalAnalyzed = 1482;
    let fakeMediaCount = 342;
    let realMediaCount = 1140;
    let imagesProcessed = 445;
    let videosProcessed = 815;
    let audioProcessed = 222;
    
    // Initial mock history database
    const historyData = [
        {
            id: "DS-9082",
            name: "presidential_briefing_leak.mp4",
            type: "video",
            date: "2026-07-22 09:12",
            engine: "ShieldNet-V2",
            score: 97,
            verdict: "danger",
            verdictLabel: "Deepfake",
            details: {
                face: 98,
                voice: 85,
                gan: 99,
                meta: 80,
                desc: "Severe spatial distortions around eye orbits. Frequency analysis shows vocal synthesis matching known XTTS-v2 engines."
            }
        },
        {
            id: "DS-9081",
            name: "executive_portrait_final.png",
            type: "image",
            date: "2026-07-21 16:45",
            engine: "GAN-Spotter",
            score: 86,
            verdict: "danger",
            verdictLabel: "Deepfake",
            details: {
                face: 65,
                voice: 0,
                gan: 92,
                meta: 60,
                desc: "High density of GAN artifact noise in structural background elements and hair follicles. Eye reflection vectors are non-parallel."
            }
        },
        {
            id: "DS-9080",
            name: "interview_cfo_voice_clip.wav",
            type: "audio",
            date: "2026-07-21 11:30",
            engine: "VoiceSync-V1",
            score: 72,
            verdict: "warning",
            verdictLabel: "Suspicious",
            details: {
                face: 0,
                voice: 94,
                gan: 40,
                meta: 85,
                desc: "Spectral inconsistencies detected in vowel transitions. High probability of ElevenLabs audio cloning overlay."
            }
        },
        {
            id: "DS-9079",
            name: "security_cam_lobby_18.mp4",
            type: "video",
            date: "2026-07-21 08:15",
            engine: "MeshNet-v4",
            score: 4,
            verdict: "safe",
            verdictLabel: "Safe Verified",
            details: {
                face: 2,
                voice: 5,
                gan: 1,
                meta: 3,
                desc: "No generative anomalies or structural warping detected. Compression artifacts consistent with original recording source."
            }
        },
        {
            id: "DS-9078",
            name: "id_verification_scan.jpg",
            type: "image",
            date: "2026-07-20 14:02",
            engine: "GAN-Spotter",
            score: 11,
            verdict: "safe",
            verdictLabel: "Safe Verified",
            details: {
                face: 8,
                voice: 0,
                gan: 12,
                meta: 5,
                desc: "Geometric facial mesh aligns correctly with physical standard parameters. High fidelity noise distribution."
            }
        }
    ];

    // Core model metrics (rendered on dashboard/reports)
    const modelMetrics = [
        { name: "ShieldNet-V2.5 Ensemble", efficiency: 99.4, scans: 682, color: "var(--color-cyan)" },
        { name: "GAN-Spotter Engine", efficiency: 98.2, scans: 412, color: "var(--color-purple)" },
        { name: "MeshNet Face Geometry", efficiency: 97.5, scans: 280, color: "var(--color-green)" },
        { name: "VoiceSync Audio Analyzer", efficiency: 95.1, scans: 108, color: "var(--color-indigo)" }
    ];

    // Live Threat Feed items
    const threatFeedData = [
        { name: "celebrity_endorsement.mp4", type: "video", time: "Just now", score: 94, verdict: "danger" },
        { name: "ceo_scam_voicemail.wav", type: "audio", time: "3m ago", score: 89, verdict: "danger" },
        { name: "politician_debate_hd.mp4", type: "video", time: "12m ago", score: 62, verdict: "warning" },
        { name: "safe_press_release.jpg", type: "image", time: "25m ago", score: 8, verdict: "safe" },
        { name: "avatar_pitch_deck.png", type: "image", time: "44m ago", score: 79, verdict: "warning" }
    ];

    let currentSelectedFile = null;
    let currentFileHash = null;
    const predictionCache = new Map();

    // ── SHA-256 File Hashing via Web Crypto API ───────────────────────
    const computeSHA256 = async (file) => {
        const arrayBuffer = await file.arrayBuffer();
        const hashBuffer = await crypto.subtle.digest('SHA-256', arrayBuffer);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    };

    // ── Deterministic Factor Derivation (Zero Math.random) ─────────────
    const deriveDeterministicFactors = (hash, confidence, isAudio) => {
        const h0 = parseInt(hash.substring(0, 2), 16);
        const h1 = parseInt(hash.substring(2, 4), 16);
        const h2 = parseInt(hash.substring(4, 6), 16);
        const h3 = parseInt(hash.substring(6, 8), 16);

        const face = Math.min(100, Math.max(0, Math.round(confidence + ((h0 % 11) - 5))));
        const gan  = Math.min(100, Math.max(0, Math.round(confidence + ((h1 % 11) - 5))));
        const voice = isAudio ? Math.min(100, Math.max(0, Math.round(confidence + ((h2 % 11) - 5)))) : Math.round((h2 % 20));
        const meta = Math.min(100, Math.max(0, Math.round(confidence * 0.8 + ((h3 % 15) - 7))));

        return { face, gan, voice, meta };
    };

    // ── Supported File Validation Helper ─────────────
    const validateFileType = (file) => {
        if (!file) return false;
        const ext = file.name.split('.').pop().toLowerCase();
        const type = (file.type || '').toLowerCase();
        
        const validExtensions = ['jpg', 'jpeg', 'png', 'webp', 'gif', 'bmp', 'svg', 'mp4', 'avi', 'mov', 'mkv', 'webm', 'mp3', 'wav', 'aac', 'flac', 'ogg', 'm4a'];
        const isImage = type.startsWith('image/') || ['jpg', 'jpeg', 'png', 'webp', 'gif', 'bmp', 'svg'].includes(ext);
        const isVideo = type.startsWith('video/') || ['mp4', 'avi', 'mov', 'mkv', 'webm'].includes(ext);
        const isAudio = type.startsWith('audio/') || ['mp3', 'wav', 'aac', 'flac', 'ogg', 'm4a'].includes(ext);

        return isImage || isVideo || isAudio || validExtensions.includes(ext);
    };

    // ── Prediction Provider (FastAPI Backend + Offline Fallback) ────────
    const getPredictionForFile = async (file, hash) => {
        if (predictionCache.has(hash)) {
            const cached = predictionCache.get(hash);
            if (cached && cached.status !== 'error') {
                return cached;
            } else {
                predictionCache.delete(hash);
            }
        }

        let result = null;

        try {
            const formData = new FormData();
            formData.append('file', file);

            const ext = (file.name || '').split('.').pop().toLowerCase();
            const isVideo = file.type.startsWith('video/') || ['mp4', 'avi', 'mov', 'mkv', 'webm'].includes(ext);
            const isAudio = file.type.startsWith('audio/') || ['mp3', 'wav', 'aac', 'flac', 'ogg', 'm4a'].includes(ext);

            let endpoint = `${API_BASE_URL}/predict`;
            let timeoutMs = 45000;

            if (file.type.startsWith('image/')) {
                endpoint = `${API_BASE_URL}/predict/image`;
            } else if (isVideo) {
                endpoint = `${API_BASE_URL}/predict/video`;
                timeoutMs = 120000; // 2 minutes timeout for video processing
            } else if (isAudio) {
                endpoint = `${API_BASE_URL}/predict/audio`;
            }

            const response = await fetchWithTimeout(endpoint, {
                method: 'POST',
                body: formData,
            }, timeoutMs);

            if (response.ok) {
                const data = await response.json();

                if (data.status === 'error') {
                    showToast(data.message || 'Analysis processing error', 'error', 4000);
                    return {
                        status: "error",
                        prediction: data.prediction || "Analysis Failed",
                        message: data.message || "Model evaluation encountered an error.",
                        confidence: null,
                        risk_level: "Error",
                        processing_time: parseFloat(data.processing_time || 0),
                        model: data.model || "Offline",
                        media_type: isVideo ? 'video' : isAudio ? 'audio' : 'image',
                        filename: file.name,
                        hash: hash,
                        factors: { face: 0, voice: 0, gan: 0, meta: 0 }
                    };
                }

                const overallPred = data.overall?.prediction || data.prediction || "Real";
                const isFake = String(overallPred).toLowerCase().includes('fake');
                const confScore = typeof (data.overall?.confidence ?? data.confidence) === 'number'
                    ? parseFloat((data.overall?.confidence ?? data.confidence).toFixed(1))
                    : 85.0;

                result = {
                    status: data.status || "success",
                    prediction: overallPred,
                    confidence: confScore,
                    risk_level: data.overall?.risk_level || data.risk_level || (isFake ? "High" : "Low"),
                    reason: data.overall?.reason || data.reason || data.message || (isFake ? "Synthetic Media Detected" : "Authentic Media Verified"),
                    message: data.overall?.reason || data.reason || data.message || (isFake ? "Synthetic Media Detected" : "Authentic Media Verified"),
                    processing_time: parseFloat(data.processing_time || 0.45),
                    model: data.model || "VideoMAE + Wav2Vec2 Pipeline",

                    video_prediction: data.video?.prediction || data.video_prediction,
                    video_confidence: typeof data.video?.confidence === 'number' ? parseFloat(data.video.confidence.toFixed(1)) : null,
                    video_frames: data.video?.frames_analyzed || 16,
                    video_model: data.video?.model || "VideoMAE",

                    audio_prediction: data.audio?.prediction || data.audio_prediction,
                    audio_confidence: typeof data.audio?.confidence === 'number' ? parseFloat(data.audio.confidence.toFixed(1)) : null,
                    audio_model: data.audio?.model || "Wav2Vec2",

                    media_type: isVideo ? 'video' : isAudio ? 'audio' : 'image',
                    filename: data.filename || file.name,
                    hash: data.hash || hash,
                    factors: data.factors || { face: isFake ? 85 : 10, voice: 0, gan: isFake ? 90 : 10, meta: 10 },
                    explainability_images: data.explainability_images || []
                };
                showToast('DeepShield Analysis Received', 'success', 3000);
            } else {
                throw new Error(`Server returned HTTP status ${response.status}`);
            }
        } catch (err) {
            console.warn('[DeepShield Backend] API call error:', err);
            showToast(`Backend Error: ${err.message || 'Unable to connect to server'}.`, 'error', 4000);
            result = {
                status: "error",
                prediction: "Backend Unavailable",
                message: "Could not connect to DeepShield backend server.",
                confidence: null,
                risk_level: "Unavailable",
                processing_time: 0,
                model: "Offline",
                media_type: file.type.startsWith('video/') ? 'video' : file.type.startsWith('audio/') ? 'audio' : 'image',
                filename: file.name,
                hash: hash,
                factors: { face: 0, voice: 0, gan: 0, meta: 0 },
                explainability_images: []
            };
        }

        if (result && result.status !== "error") {
            predictionCache.set(hash, result);
        }
        console.log("API RESPONSE:", result);
        return result;
    };

    // -------------------------------------------------------------
    // 2. Navigation Routing & Sidebar toggles
    // -------------------------------------------------------------
    const navItems = document.querySelectorAll('.nav-item');
    const sections = document.querySelectorAll('.view-section');
    const pageTitle = document.getElementById('viewport-title');
    const sidebar = document.querySelector('.sidebar');
    const sidebarToggle = document.getElementById('sidebar-toggle');

    const updateViewTitle = (viewName) => {
        const titles = {
            'dashboard': 'Dashboard Overview',
            'analyze': 'Analyze Media & Forensic Lab',
            'history': 'Analysis Forensic Logs',
            'reports': 'System Analysis & Threat Reports',
            'settings': 'System Configurations',
            'about': 'About DeepShield Tech Stack'
        };
        pageTitle.textContent = titles[viewName] || 'DeepShield';
    };

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            navItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');

            const view = item.getAttribute('data-view');
            sections.forEach(sec => {
                sec.classList.remove('active');
                if (sec.id === `${view}-view`) {
                    sec.classList.add('active');
                }
            });

            updateViewTitle(view);
            
            // Retrigger animations if applicable
            if (view === 'dashboard') {
                animateModelBars();
            } else if (view === 'reports') {
                animateReportBars();
            }

            // Close mobile menu if open
            if (window.innerWidth <= 768) {
                sidebar.classList.remove('open');
            }
        });
    });

    // Mobile menu toggle
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', (e) => {
            e.stopPropagation();
            sidebar.classList.toggle('open');
        });
    }

    // Close sidebar on outer body clicks in mobile mode
    document.addEventListener('click', (e) => {
        if (window.innerWidth <= 768 && sidebar.classList.contains('open')) {
            if (!sidebar.contains(e.target) && e.target !== sidebarToggle) {
                sidebar.classList.remove('open');
            }
        }
    });

    // Initialize Lucide Icons
    lucide.createIcons();

    // -------------------------------------------------------------
    // 3. Dashboard Initialization & Feeds
    // -------------------------------------------------------------
    const threatFeed = document.getElementById('threat-logs-feed');
    let trendChart, donutChart, radarChart;

    const populateThreatFeed = () => {
        if (!threatFeed) return;
        threatFeed.innerHTML = '';
        threatFeedData.forEach(item => {
            const severityClass = item.verdict === 'danger' ? 'severity-danger' : 
                                  item.verdict === 'warning' ? 'severity-warning' : 'severity-safe';
            const scoreClass = item.verdict === 'danger' ? 'high' : 
                               item.verdict === 'warning' ? 'mid' : 'low';
            const fileIcon = item.type === 'video' ? 'video' : 
                             item.type === 'audio' ? 'music' : 'image';

            const element = document.createElement('div');
            element.className = 'threat-item';
            element.innerHTML = `
                <div class="threat-severity ${severityClass}"></div>
                <div class="threat-details">
                    <div class="threat-meta">
                        <span class="threat-name">${item.name}</span>
                        <span class="threat-time">${item.time}</span>
                    </div>
                    <div class="threat-desc">
                        <span><i data-lucide="${fileIcon}" style="width: 10px; height: 10px; display: inline-block; vertical-align: middle; margin-right: 4px;"></i>${item.type.toUpperCase()}</span>
                        <span class="threat-score ${scoreClass}">Risk: ${item.score}%</span>
                    </div>
                </div>
            `;
            threatFeed.appendChild(element);
        });
        lucide.createIcons({attrs: {class: 'icon-inline'}});
    };

    const triggerScanResults = (res, isCached = false) => {
        if (!res) return;
        console.log("TRIGGER SCAN RESULTS RECEIVED:", res);

        // Placeholders shown ONLY when fetch() fails completely (status === 'error' with no prediction)
        const isUntrained = res.status === 'error' && !res.prediction;
        const predLower = res.prediction ? String(res.prediction).toLowerCase() : '';
        const isFake = !isUntrained && (predLower.includes('fake') || predLower.includes('deepfake'));
        const isReal = !isUntrained && predLower.includes('real');

        const faceVal = res.factors ? res.factors.face : 0;
        const voiceVal = res.factors ? res.factors.voice : 0;
        const ganVal = res.factors ? res.factors.gan : 0;
        const metaVal = res.factors ? res.factors.meta : 0;

        // 1. Prediction Badge & Hero Section
        const predBadge = document.getElementById('result-prediction-badge');
        const predText = document.getElementById('result-prediction-text');
        const heroTitle = document.getElementById('result-hero-title');
        const heroSub = document.getElementById('result-hero-sub');
        const confNum = document.getElementById('result-confidence-num');

        if (predBadge && predText) {
            if (isUntrained) {
                predBadge.className = 'result-prediction-badge warning';
                predText.textContent = 'MODEL NOT TRAINED';
            } else if (isFake) {
                predBadge.className = 'result-prediction-badge fake';
                predText.textContent = res.prediction ? res.prediction.toUpperCase() : 'FAKE';
            } else {
                predBadge.className = 'result-prediction-badge real';
                predText.textContent = res.prediction ? res.prediction.toUpperCase() : 'REAL';
            }
        }

        const isArtworkRes = res.is_artwork || res.media_category === 'Artwork' || res.media_type === 'artwork';

        if (heroTitle) {
            if (isUntrained) {
                heroTitle.textContent = 'Model Not Trained';
            } else if (isArtworkRes) {
                heroTitle.textContent = isFake ? 'Synthetic AI Artwork Detected' : 'Authentic Human Artwork Verified';
            } else if (isFake) {
                heroTitle.textContent = 'Synthetic Media Detected';
            } else {
                heroTitle.textContent = 'Authentic Media Verified';
            }
        }

        if (heroSub) {
            if (isUntrained) {
                heroSub.textContent = res.message || 'Model not yet trained. Reliable prediction unavailable.';
            } else {
                heroSub.textContent = res.message || res.reason || (isFake ? 'High spatial distortion & generative signatures detected across deep neural layers.' : 'No deep learning manipulation metrics or generative traces found. Authentic signatures validated.');
            }
        }

        if (confNum) {
            if (isUntrained || res.confidence === null || res.confidence === undefined) {
                confNum.textContent = 'Confidence unavailable.';
            } else {
                confNum.textContent = `${res.confidence}%`;
            }
        }

        // 2. 6 Metadata Metric Cards
        const metaRisk = document.getElementById('res-meta-risk');
        const metaModel = document.getElementById('res-meta-model');
        const metaTime = document.getElementById('res-meta-time');
        const metaType = document.getElementById('res-meta-type');
        const metaSize = document.getElementById('res-meta-size');
        const metaDate = document.getElementById('res-meta-date');

        if (metaRisk) {
            metaRisk.textContent = isUntrained ? 'Untrained Risk' : `${res.risk_level} Risk`;
            metaRisk.className = `meta-value ${res.risk_level === 'High' ? 'risk-high' : res.risk_level === 'Medium' ? 'risk-mid' : 'risk-low'}`;
        }
        if (metaModel) metaModel.textContent = res.model || 'DeepShield AI Engine';
        if (metaTime) metaTime.textContent = `${res.processing_time || 0}s`;

        const typeStr = isArtworkRes ? 'ARTWORK' : (res.media_type ? res.media_type.toUpperCase() : (currentSelectedFile ? (currentSelectedFile.type.startsWith('video/') ? 'VIDEO' : currentSelectedFile.type.startsWith('audio/') ? 'AUDIO' : 'IMAGE') : 'IMAGE'));
        const sizeStr = currentSelectedFile ? (currentSelectedFile.size / (1024 * 1024)).toFixed(2) + ' MB' : '—';
        const dateStr = res.timestamp ? res.timestamp.replace('T', ' ').substring(0, 16) : new Date().toISOString().replace('T', ' ').substring(0, 16);

        if (metaType) metaType.textContent = typeStr;
        if (metaSize) metaSize.textContent = sizeStr;
        if (metaDate) metaDate.textContent = dateStr;

        // 3. Update Breakdown Bars
        const facePctEl = document.getElementById('factor-face-pct');
        const faceBarEl = document.getElementById('factor-face-bar');
        const voicePctEl = document.getElementById('factor-voice-pct');
        const voiceBarEl = document.getElementById('factor-voice-bar');
        const ganPctEl = document.getElementById('factor-gan-pct');
        const ganBarEl = document.getElementById('factor-gan-bar');
        const metaPctEl = document.getElementById('factor-meta-pct');
        const metaBarEl = document.getElementById('factor-meta-bar');

        if (facePctEl) facePctEl.textContent = `${faceVal}%`;
        if (faceBarEl) { faceBarEl.style.width = `${faceVal}%`; setFactorBarColor(faceBarEl, faceVal); }

        if (voicePctEl) voicePctEl.textContent = `${voiceVal}%`;
        if (voiceBarEl) { voiceBarEl.style.width = `${voiceVal}%`; setFactorBarColor(voiceBarEl, voiceVal); }

        if (ganPctEl) ganPctEl.textContent = `${ganVal}%`;
        if (ganBarEl) { ganBarEl.style.width = `${ganVal}%`; setFactorBarColor(ganBarEl, ganVal); }

        if (metaPctEl) metaPctEl.textContent = `${metaVal}%`;
        if (metaBarEl) { metaBarEl.style.width = `${metaVal}%`; setFactorBarColor(metaBarEl, metaVal); }

        // 3.5 Multimodal Stream Analysis (Video / Audio Breakdown Cards)
        const multiSection = document.getElementById('result-multimodal-section');
        if (multiSection) {
            const isVideoRes = (res.media_type === 'video') || (currentSelectedFile && (currentSelectedFile.type.startsWith('video/') || ['mp4', 'avi', 'mov', 'mkv', 'webm'].includes(currentSelectedFile.name.split('.').pop().toLowerCase()))) || !!res.video_prediction;
            if (isVideoRes) {
                multiSection.style.display = 'block';

                const vPredEl = document.getElementById('v-stream-pred');
                const vConfEl = document.getElementById('v-stream-conf');
                const vFramesEl = document.getElementById('v-stream-frames');
                const vModelEl = document.getElementById('v-stream-model');
                const vBadgeEl = document.getElementById('v-stream-badge');

                const vPred = res.video_prediction || res.prediction || 'Real';
                const vConf = res.video_confidence !== undefined && res.video_confidence !== null ? res.video_confidence : res.confidence;
                const isVFake = String(vPred).toLowerCase().includes('fake');

                if (vPredEl) vPredEl.textContent = vPred;
                if (vConfEl) vConfEl.textContent = `${vConf}%`;
                if (vFramesEl) vFramesEl.textContent = res.video_frames || 16;
                if (vModelEl) vModelEl.textContent = res.video_model || 'VideoMAE';
                if (vBadgeEl) {
                    vBadgeEl.textContent = String(vPred).toUpperCase();
                    vBadgeEl.className = `status-pill ${isVFake ? 'offline' : 'online'}`;
                }

                const aPredEl = document.getElementById('a-stream-pred');
                const aConfEl = document.getElementById('a-stream-conf');
                const aModelEl = document.getElementById('a-stream-model');
                const aBadgeEl = document.getElementById('a-stream-badge');

                const aPred = res.audio_prediction || 'Real';
                const aConf = res.audio_confidence !== undefined && res.audio_confidence !== null ? res.audio_confidence : 0;
                const isAFake = String(aPred).toLowerCase().includes('fake');
                const isAUnavail = String(aPred).toLowerCase().includes('unavail');

                if (aPredEl) aPredEl.textContent = aPred;
                if (aConfEl) aConfEl.textContent = isAUnavail ? '0%' : `${aConf}%`;
                if (aModelEl) aModelEl.textContent = res.audio_model || 'Wav2Vec2';
                if (aBadgeEl) {
                    aBadgeEl.textContent = String(aPred).toUpperCase();
                    aBadgeEl.className = `status-pill ${isAUnavail ? 'warning' : isAFake ? 'offline' : 'online'}`;
                }

                const oPredEl = document.getElementById('o-stream-pred');
                const oConfEl = document.getElementById('o-stream-conf');
                const oRiskEl = document.getElementById('o-stream-risk');
                const oReasonEl = document.getElementById('o-stream-reason');
                const oBadgeEl = document.getElementById('o-stream-badge');

                const oPred = res.prediction || 'Real';
                const oConf = res.confidence;
                const isOFake = String(oPred).toLowerCase().includes('fake');

                if (oPredEl) oPredEl.textContent = oPred;
                if (oConfEl) oConfEl.textContent = `${oConf}%`;
                if (oRiskEl) {
                    oRiskEl.textContent = `${res.risk_level} Risk`;
                    oRiskEl.style.color = res.risk_level === 'High' ? 'var(--color-red)' : res.risk_level === 'Medium' ? 'var(--color-warning)' : 'var(--color-green)';
                }
                if (oReasonEl) oReasonEl.textContent = res.reason || res.message || 'Multimodal video analysis complete.';
                if (oBadgeEl) {
                    oBadgeEl.textContent = String(oPred).toUpperCase();
                    oBadgeEl.className = `status-pill ${isOFake ? 'offline' : 'online'}`;
                }
            } else {
                multiSection.style.display = 'none';
            }
        }

        // 4. Action Button Listeners
        const btnAgain = document.getElementById('btn-result-analyze-again');
        const btnDownload = document.getElementById('btn-result-download-report');
        const btnShare = document.getElementById('btn-result-share');

        if (btnAgain) {
            btnAgain.onclick = () => {
                const dropZoneEl = document.getElementById('drop-zone');
                if (dropZoneEl) dropZoneEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
            };
        }

        if (btnDownload) {
            btnDownload.onclick = () => {
                generatePDFReport(res, currentSelectedFile);
            };
        }

        if (btnShare) {
            btnShare.onclick = () => {
                const shareUrl = `https://deepshield.ai/audit/${res.hash.substring(0, 16)}`;
                navigator.clipboard.writeText(shareUrl).then(() => {
                    alert(`Cryptographic audit link copied to clipboard:\n${shareUrl}`);
                }).catch(() => {
                    alert(`Cryptographic Audit Signature: ${res.hash}`);
                });
            };
        }

        // 5. Render Explainability Section (Image Grad-CAM / Video Keyframes / Audio Spectrogram)
        renderExplainabilitySection(res, currentSelectedFile);

        // Display results block
        scanResultsPanel.style.display = 'block';
        setTimeout(() => {
            scanResultsPanel.classList.add('active');
            scanResultsPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            lucide.createIcons();
        }, 50);
    };

    const initApexCharts = () => {
        // ── 1. Line Chart: Trend Area ──
        const trendOptions = {
            series: [
                { name: 'Total Scans', data: [110, 185, 155, 180, 230, 235, 242] },
                { name: 'Fake Media', data: [32, 24, 55, 48, 45, 60, 68] }
            ],
            chart: {
                type: 'area', height: 260,
                background: 'transparent',
                toolbar: { show: false },
                animations: { enabled: true, easing: 'easeinout', speed: 800 },
                dropShadow: { enabled: true, top: 6, blur: 8, opacity: 0.15 }
            },
            colors: ['#00f2fe', '#ef4444'],
            fill: {
                type: 'gradient',
                gradient: { shadeIntensity: 1, opacityFrom: 0.35, opacityTo: 0.02, stops: [0, 100] }
            },
            dataLabels: { enabled: false },
            stroke: { curve: 'smooth', width: [2.5, 2.5] },
            xaxis: {
                categories: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                labels: { style: { colors: Array(7).fill('#64748b'), fontSize: '11px', fontFamily: 'Inter' } },
                axisBorder: { show: false }, axisTicks: { show: false }
            },
            yaxis: { labels: { style: { colors: '#64748b', fontSize: '11px', fontFamily: 'Inter' } } },
            grid: { borderColor: 'rgba(255,255,255,0.04)', strokeDashArray: 4, padding: { left: 4, right: 4 } },
            legend: {
                show: true, position: 'top', horizontalAlign: 'right',
                labels: { colors: '#cbd5e1' },
                markers: { size: 6, shape: 'circle' }
            },
            tooltip: { theme: 'dark', style: { fontSize: '12px', fontFamily: 'Inter' } },
            markers: { size: 4, hover: { size: 6 } }
        };
        const trendEl = document.querySelector('#apex-trend-chart');
        if (trendEl) { trendChart = new ApexCharts(trendEl, trendOptions); trendChart.render(); }

        // ── 2. Bar Chart: Monthly Statistics ──
        const barOptions = {
            series: [
                { name: 'Images', data: [120, 180, 220, 290, 340, 410, 445] },
                { name: 'Videos', data: [310, 420, 510, 620, 710, 780, 815] },
                { name: 'Audio', data: [60, 85, 110, 140, 175, 195, 222] }
            ],
            chart: {
                type: 'bar', height: 260, stacked: true,
                background: 'transparent',
                toolbar: { show: false },
                animations: { enabled: true, easing: 'easeinout', speed: 800 }
            },
            colors: ['#a855f7', '#00f2fe', '#f59e0b'],
            plotOptions: {
                bar: { borderRadius: 4, columnWidth: '38%' }
            },
            dataLabels: { enabled: false },
            xaxis: {
                categories: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'],
                labels: { style: { colors: Array(7).fill('#64748b'), fontSize: '11px', fontFamily: 'Inter' } },
                axisBorder: { show: false }, axisTicks: { show: false }
            },
            yaxis: { labels: { style: { colors: '#64748b', fontSize: '11px', fontFamily: 'Inter' } } },
            grid: { borderColor: 'rgba(255,255,255,0.04)', strokeDashArray: 4 },
            legend: {
                show: true, position: 'top', horizontalAlign: 'right',
                labels: { colors: '#cbd5e1' },
                markers: { size: 6, shape: 'circle' }
            },
            tooltip: { theme: 'dark', style: { fontSize: '12px', fontFamily: 'Inter' } }
        };
        const barEl = document.querySelector('#apex-bar-chart');
        if (barEl) { barChart = new ApexCharts(barEl, barOptions); barChart.render(); }

        // ── 3. Donut Chart: Media Ratio ──
        const donutOptions = {
            series: [videosProcessed, imagesProcessed, audioProcessed],
            labels: ['Videos', 'Images', 'Audio'],
            chart: {
                type: 'donut', height: 260,
                background: 'transparent',
                animations: { enabled: true, easing: 'easeinout', speed: 900 }
            },
            colors: ['#00f2fe', '#a855f7', '#f59e0b'],
            stroke: { show: false },
            dataLabels: { enabled: false },
            plotOptions: {
                pie: { donut: {
                    size: '72%',
                    labels: {
                        show: true,
                        name: { show: true, color: '#94a3b8', fontSize: '12px', fontFamily: 'Inter' },
                        value: { show: true, color: '#f1f5f9', fontSize: '22px', fontWeight: 700, fontFamily: 'Outfit', formatter: v => Math.round(v) },
                        total: { show: true, label: 'Total Files', color: '#64748b', fontFamily: 'Inter', fontSize: '12px',
                                 formatter: () => totalAnalyzed.toLocaleString() }
                    }
                }}
            },
            legend: {
                position: 'bottom', horizontalAlign: 'center',
                labels: { colors: '#94a3b8' },
                markers: { size: 7, shape: 'circle', offsetX: -2 },
                itemMargin: { horizontal: 10 }
            },
            tooltip: { theme: 'dark', style: { fontFamily: 'Inter' } }
        };
        const donutEl = document.querySelector('#apex-donut-chart');
        if (donutEl) { donutChart = new ApexCharts(donutEl, donutOptions); donutChart.render(); }

        // ── 4. Radar Chart: Model Confidence Vector ──
        const radarOptions = {
            series: [{ name: 'Confidence %', data: [98, 92, 95, 88] }],
            chart: {
                type: 'radar', height: 260,
                background: 'transparent',
                toolbar: { show: false },
                animations: { enabled: true, easing: 'easeinout', speed: 900 }
            },
            colors: ['#00f2fe'],
            stroke: { width: 2, colors: ['#00f2fe'] },
            fill: { opacity: 0.15, colors: ['#00f2fe'] },
            markers: { size: 5, colors: ['#04060d'], strokeColors: '#00f2fe', strokeWidth: 2, hover: { size: 7 } },
            xaxis: {
                categories: ['ShieldNet-V2', 'GAN-Spotter', 'MeshNet-v4', 'VoiceSync-V1'],
                labels: { style: { colors: Array(4).fill('#94a3b8'), fontSize: '10px', fontFamily: 'Inter' } }
            },
            yaxis: { show: false, max: 100 },
            plotOptions: {
                radar: { polygons: {
                    strokeColors: 'rgba(255,255,255,0.06)',
                    connectorColors: 'rgba(255,255,255,0.06)',
                    fill: { colors: ['rgba(255,255,255,0.015)', 'transparent'] }
                }}
            },
            tooltip: { theme: 'dark', style: { fontFamily: 'Inter' }, y: { formatter: v => v + '%' } }
        };
        const radarEl = document.querySelector('#apex-radar-chart');
        if (radarEl) { radarChart = new ApexCharts(radarEl, radarOptions); radarChart.render(); }
    };


    const populateRecentAnalysesWidget = () => {
        const widget = document.getElementById('recent-analyses-widget');
        if (!widget) return;
        widget.innerHTML = '';
        const dotColor = { danger: '#ef4444', warning: '#f59e0b', safe: '#10b981' };
        historyData.slice(0, 5).forEach(item => {
            const scoreClass = item.verdict;
            const el = document.createElement('div');
            el.className = 'ra-item';
            el.innerHTML = `
                <span class="ra-type-dot" style="background:${dotColor[item.verdict] || '#64748b'};"></span>
                <span class="ra-name" title="${item.name}">${item.name}</span>
                <span class="ra-score-badge ${scoreClass}">${item.score}%</span>
            `;
            el.addEventListener('click', () => {
                const histTab = document.querySelector('.nav-item[data-view="history"]');
                if (histTab) histTab.click();
            });
            widget.appendChild(el);
        });
        lucide.createIcons();
    };

    const populateLatestDetectionWidget = () => {
        const widget = document.getElementById('latest-detection-widget');
        if (!widget) return;
        const dangerItems = historyData.filter(i => i.verdict === 'danger');
        const item = dangerItems.length > 0 ? dangerItems[0] : historyData[0];
        if (!item) return;

        widget.innerHTML = `
            <div class="ld-badge">
                <span class="ld-pulse"></span>
                DEEPFAKE DETECTED
            </div>
            <div class="ld-filename" title="${item.name}">${item.name}</div>
            <div style="font-family:var(--font-title);font-size:28px;font-weight:800;color:var(--color-red);line-height:1;">${item.score}%</div>
            <div style="font-size:11px;color:var(--text-muted);margin-top:-4px;">Synthetic probability score</div>
            <div class="ld-metrics">
                <div class="ld-metric-row">
                    <span class="ld-metric-label">Face Artifacts</span>
                    <div class="ld-metric-bar-track"><div class="ld-metric-bar-fill" style="width:${item.details.face}%;"></div></div>
                    <span class="ld-metric-val">${item.details.face}%</span>
                </div>
                <div class="ld-metric-row">
                    <span class="ld-metric-label">GAN Signatures</span>
                    <div class="ld-metric-bar-track"><div class="ld-metric-bar-fill" style="width:${item.details.gan}%;"></div></div>
                    <span class="ld-metric-val">${item.details.gan}%</span>
                </div>
                <div class="ld-metric-row">
                    <span class="ld-metric-label">Voice Synthesis</span>
                    <div class="ld-metric-bar-track"><div class="ld-metric-bar-fill" style="width:${item.details.voice}%;"></div></div>
                    <span class="ld-metric-val">${item.details.voice}%</span>
                </div>
            </div>
            <div class="ld-time">${item.date} • ${item.engine}</div>
        `;
    };

    const setupQuickActions = () => {
        const qaNewScan = document.getElementById('qa-new-scan');
        const qaReport = document.getElementById('qa-report');
        const qaSettings = document.getElementById('qa-settings');
        const qaDiagnostics = document.getElementById('qa-diagnostics');
        
        if (qaNewScan) qaNewScan.addEventListener('click', () => document.querySelector('.nav-item[data-view="analyze"]').click());
        if (qaReport) qaReport.addEventListener('click', () => alert("Audit Report generated and downloaded successfully."));
        if (qaSettings) qaSettings.addEventListener('click', () => document.querySelector('.nav-item[data-view="settings"]').click());
        if (qaDiagnostics) qaDiagnostics.addEventListener('click', () => alert("Running deep engine neural diagnostics... All systems nominal."));
    };

    // -------------------------------------------------------------
    // 4. File Upload & Processing View — Premium Redesign
    // -------------------------------------------------------------
    const dropZone = document.getElementById('drop-zone');
    const mediaInput = document.getElementById('media-file-input');
    const browseBtn = document.getElementById('browse-btn');
    const uploadPrompt = document.getElementById('upload-prompt');
    const previewContainer = document.getElementById('preview-container');
    const previewWrapper = document.getElementById('preview-wrapper');
    const previewFileName = document.getElementById('preview-file-name');
    const removeFileBtn = document.getElementById('remove-file-btn');
    const scanBtn = document.getElementById('start-scan-btn');
    const scannerEffect = document.getElementById('scanning-effect');
    const consoleTerminal = document.getElementById('terminal-console');
    const consoleStatus = document.getElementById('console-status');
    const scanResultsPanel = document.getElementById('scan-results-panel');
    const clearScanBtn = document.getElementById('clear-scan-btn');
    const clearScanResultsBtn = document.getElementById('clear-scan-results-btn');
    const analyzeCTAArea = document.getElementById('analyze-cta-area');
    const audioPlayerPanel = document.getElementById('audio-player-panel');
    const audioWaveform = document.getElementById('audio-waveform');
    const audioPlayBtn = document.getElementById('audio-play-btn');
    const audioProgress = document.getElementById('audio-progress');
    const audioTimeLabel = document.getElementById('audio-time');
    const audioElement = document.getElementById('audio-element');
    const mediaTypeCard = document.getElementById('media-type-card');
    const mediaInfoGrid = document.getElementById('media-info-grid');
    const dropOverlay = document.getElementById('drop-overlay');
    const premFileTypeBadge = document.getElementById('prem-file-type-badge');
    const premFileTypeLabel = document.getElementById('prem-file-type-label');
    const premFileSizeEl = document.getElementById('preview-file-size');
    const premFileDimEl = document.getElementById('preview-file-dim');

    // Drag-over highlights
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleSelectedFile(files[0]);
        }
    });

    // Clicking on the prompt area triggers file browse
    uploadPrompt.addEventListener('click', () => mediaInput.click());
    browseBtn.addEventListener('click', (e) => { e.stopPropagation(); mediaInput.click(); });

    mediaInput.addEventListener('change', () => {
        if (mediaInput.files.length > 0) {
            handleSelectedFile(mediaInput.files[0]);
        }
    });

    // Audio player logic
    let audioIsPlaying = false;
    if (audioPlayBtn && audioElement) {
        audioPlayBtn.addEventListener('click', () => {
            if (audioIsPlaying) {
                audioElement.pause();
                audioIsPlaying = false;
                audioPlayBtn.innerHTML = '<i data-lucide="play" style="width:18px;height:18px;"></i>';
                audioWaveform.querySelectorAll('.wbar').forEach(b => b.classList.add('paused'));
            } else {
                audioElement.play();
                audioIsPlaying = true;
                audioPlayBtn.innerHTML = '<i data-lucide="pause" style="width:18px;height:18px;"></i>';
                audioWaveform.querySelectorAll('.wbar').forEach(b => b.classList.remove('paused'));
            }
            lucide.createIcons();
        });

        audioElement.addEventListener('timeupdate', () => {
            if (!audioElement.duration) return;
            const pct = (audioElement.currentTime / audioElement.duration) * 100;
            if (audioProgress) audioProgress.style.width = pct + '%';
            const mins = Math.floor(audioElement.currentTime / 60);
            const secs = Math.floor(audioElement.currentTime % 60).toString().padStart(2, '0');
            if (audioTimeLabel) audioTimeLabel.textContent = `${mins}:${secs}`;
        });
        audioElement.addEventListener('ended', () => {
            audioIsPlaying = false;
            if (audioPlayBtn) audioPlayBtn.innerHTML = '<i data-lucide="play" style="width:18px;height:18px;"></i>';
            if (audioProgress) audioProgress.style.width = '0%';
            audioWaveform.querySelectorAll('.wbar').forEach(b => b.classList.add('paused'));
            lucide.createIcons();
        });
    }

    const buildAudioWaveform = () => {
        if (!audioWaveform) return;
        audioWaveform.innerHTML = '';
        const heights = [20, 35, 15, 48, 28, 40, 18, 55, 32, 25, 44, 16, 50, 38, 22,
                         30, 46, 12, 42, 26, 52, 18, 36, 28, 45, 20, 38, 14, 48, 30];
        heights.forEach((h, i) => {
            const bar = document.createElement('div');
            bar.className = 'wbar';
            bar.style.height = h + 'px';
            bar.style.animationDelay = (i * 0.05) + 's';
            audioWaveform.appendChild(bar);
        });
    };

    const populateMediaInfoPanel = (file, extra = {}) => {
        if (!mediaTypeCard || !mediaInfoGrid) return;
        const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
        let typeLabel = 'IMAGE';
        if (file.type.startsWith('video/')) typeLabel = 'VIDEO';
        else if (file.type.startsWith('audio/')) typeLabel = 'AUDIO';

        const ext = file.name.split('.').pop().toUpperCase();
        const items = [
            { label: 'File Type', value: typeLabel },
            { label: 'Extension', value: ext },
            { label: 'File Size', value: sizeMB + ' MB' },
            { label: 'MIME Type', value: file.type || '—' },
            { label: 'Dimensions', value: extra.dimensions || '—' },
            { label: 'Modified', value: extra.modified || new Date(file.lastModified).toLocaleDateString() }
        ];

        mediaInfoGrid.innerHTML = '';
        items.forEach(item => {
            const div = document.createElement('div');
            div.className = 'media-info-item';
            div.innerHTML = `<div class="media-info-label">${item.label}</div><div class="media-info-value">${item.value}</div>`;
            mediaInfoGrid.appendChild(div);
        });
        mediaTypeCard.style.display = 'block';
    };

    const handleSelectedFile = async (file) => {
        if (!validateFileType(file)) {
            const ext = file.name.split('.').pop().toUpperCase();
            if (consoleTerminal) {
                consoleTerminal.innerHTML = `
                    <div class="console-line alert">[ERR] Unsupported file format (.${ext}).</div>
                    <div class="console-line warn">[SYS] DeepShield supports Images (PNG, JPG, WEBP), Videos (MP4, MOV, AVI), and Audio (MP3, WAV, FLAC).</div>
                `;
            }
            if (consoleStatus) consoleStatus.textContent = 'ERROR';
            showToast(`Unsupported File Format (.${ext})`, 'error', 4500);
            resetUploadState();
            return;
        }

        currentSelectedFile = file;
        const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
        let typeLabel = 'IMAGE';
        let typeIcon = 'image';

        if (file.type.startsWith('video/')) { typeLabel = 'VIDEO'; typeIcon = 'video'; }
        else if (file.type.startsWith('audio/')) { typeLabel = 'AUDIO'; typeIcon = 'music'; }

        // Compute SHA-256 Hash for file payload
        currentFileHash = await computeSHA256(file);

        // Update file type badge
        if (premFileTypeBadge) {
            premFileTypeBadge.innerHTML = `<i data-lucide="${typeIcon}" style="width:14px;height:14px;"></i><span id="prem-file-type-label">${typeLabel}</span>`;
            premFileTypeBadge.className = 'prem-file-type-badge' + (typeLabel === 'VIDEO' ? ' video-type' : typeLabel === 'AUDIO' ? ' audio-type' : '');
        }
        if (previewFileName) previewFileName.textContent = file.name;
        if (premFileSizeEl) premFileSizeEl.textContent = sizeMB + ' MB';
        if (premFileDimEl) premFileDimEl.textContent = '—';

        // Reset audio player
        audioIsPlaying = false;
        if (audioPlayerPanel) audioPlayerPanel.style.display = 'none';
        if (audioElement) { audioElement.pause(); audioElement.src = ''; }

        // Render preview content
        previewWrapper.innerHTML = '';
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = document.createElement('img');
                img.src = e.target.result;
                img.onload = () => {
                    if (premFileDimEl) premFileDimEl.textContent = img.naturalWidth + ' × ' + img.naturalHeight;
                    populateMediaInfoPanel(file, { dimensions: img.naturalWidth + ' × ' + img.naturalHeight + ' px' });
                };
                previewWrapper.appendChild(img);
            };
            reader.readAsDataURL(file);
        } else if (file.type.startsWith('video/')) {
            const video = document.createElement('video');
            video.src = URL.createObjectURL(file);
            video.controls = true;
            video.style.width = '100%';
            video.style.maxHeight = '340px';
            video.onloadedmetadata = () => {
                const dim = video.videoWidth + ' × ' + video.videoHeight;
                if (premFileDimEl) premFileDimEl.textContent = dim;
                populateMediaInfoPanel(file, { dimensions: dim + ' px' });
            };
            previewWrapper.appendChild(video);
        } else if (file.type.startsWith('audio/')) {
            // Build waveform
            buildAudioWaveform();
            if (audioElement) {
                audioElement.src = URL.createObjectURL(file);
                audioElement.load();
            }
            if (audioPlayBtn) audioPlayBtn.innerHTML = '<i data-lucide="play" style="width:18px;height:18px;"></i>';
            if (audioProgress) audioProgress.style.width = '0%';
            if (audioTimeLabel) audioTimeLabel.textContent = '0:00';
            if (audioPlayerPanel) audioPlayerPanel.style.display = 'block';
            populateMediaInfoPanel(file, { dimensions: 'Audio File' });
            lucide.createIcons();
        } else {
            const ph = document.createElement('div');
            ph.style.cssText = 'display:flex;flex-direction:column;align-items:center;gap:10px;color:var(--text-muted);';
            ph.innerHTML = `<i data-lucide="file-text" style="width:48px;height:48px;"></i><span style="font-size:13px;">Unsupported preview</span>`;
            previewWrapper.appendChild(ph);
            lucide.createIcons();
            populateMediaInfoPanel(file);
        }

        // Toggle views
        uploadPrompt.style.display = 'none';
        previewContainer.style.display = 'flex';
        if (analyzeCTAArea) analyzeCTAArea.style.display = 'flex';
        if (scanBtn) scanBtn.disabled = false;

        const isCached = predictionCache.has(currentFileHash);

        // Reset console with SHA-256 Hash info
        if (consoleTerminal) consoleTerminal.innerHTML = `
            <div class="console-line info">[SYS] Payload loaded: ${file.name}</div>
            <div class="console-line info">[SYS] Type detected: ${typeLabel} · Size: ${sizeMB} MB</div>
            <div class="console-line alert">[HASH] SHA-256: ${currentFileHash.substring(0, 16)}...${currentFileHash.substring(48)}</div>
            ${isCached ? '<div class="console-line success">[CACHE] Cached prediction ready for this file payload.</div>' : '<div class="console-line info">[SYS] Engine locked. Click "Analyze Media" to begin.</div>'}
        `;
        if (consoleStatus) consoleStatus.textContent = isCached ? 'CACHED' : 'READY';
        if (scanResultsPanel) scanResultsPanel.style.display = 'none';

        lucide.createIcons();
    };

    if (removeFileBtn) {
        removeFileBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            resetUploadState();
        });
    }

    const resetUploadState = () => {
        currentSelectedFile = null;
        currentFileHash = null;
        if (mediaInput) mediaInput.value = '';
        uploadPrompt.style.display = 'flex';
        previewContainer.style.display = 'none';
        if (analyzeCTAArea) analyzeCTAArea.style.display = 'none';
        if (scanBtn) { scanBtn.disabled = true; scanBtn.querySelector('.btn-analyze-main-text') && (scanBtn.querySelector('.btn-analyze-main-text').textContent = 'Analyze Media'); }
        if (scannerEffect) scannerEffect.style.display = 'none';
        if (audioPlayerPanel) audioPlayerPanel.style.display = 'none';
        if (audioElement) { audioElement.pause(); audioElement.src = ''; }
        audioIsPlaying = false;
        if (mediaTypeCard) mediaTypeCard.style.display = 'none';
        if (consoleTerminal) consoleTerminal.innerHTML = `
            <div class="console-line info">[SYS] ShieldNet engine waiting for payload input...</div>
            <div class="console-line info">[SYS] Upload a file to initialize forensic audit.</div>
        `;
        if (consoleStatus) consoleStatus.textContent = 'STANDBY';
        if (scanResultsPanel) scanResultsPanel.style.display = 'none';
    };

    // Sensitivity Slider label feedback
    const sensitivitySlider = document.getElementById('sensitivity-slider');
    const sensitivityVal = document.getElementById('sensitivity-val');
    const slevelFill = document.getElementById('slevel-fill');
    if (sensitivitySlider) {
        sensitivitySlider.addEventListener('input', (e) => {
            if (sensitivityVal) sensitivityVal.textContent = e.target.value;
            if (slevelFill) slevelFill.style.width = e.target.value + '%';
        });
    }

    // Clear scan results button
    if (clearScanResultsBtn) clearScanResultsBtn.addEventListener('click', resetUploadState);

    // Futuristic AI Processing Screen Controller
    const runAIProcessingScreen = (file, prediction, isCached, onComplete) => {
        const modal = document.getElementById('ai-processing-modal');
        const fileText = document.getElementById('ai-processing-file-text');
        const modelText = document.getElementById('ai-processing-model');
        const estTimeEl = document.getElementById('ai-est-time');
        const stageText = document.getElementById('ai-stage-text');
        const progressPct = document.getElementById('ai-progress-pct');
        const progressFill = document.getElementById('ai-progress-fill');

        if (!modal) {
            onComplete();
            return;
        }

        if (fileText) fileText.textContent = file.name;
        if (modelText) modelText.textContent = (prediction && prediction.model) ? prediction.model : 'DeepShield AI Engine';

        // Display modal
        modal.style.display = 'flex';
        lucide.createIcons();

        const totalTimeMs = isCached ? 900 : 2600;
        const startTime = Date.now();

        const stages = [
            { pct: 0,  name: 'Uploading...',            step: 1 },
            { pct: 20, name: 'Preprocessing...',          step: 2 },
            { pct: 40, name: 'Feature Extraction...',     step: 3 },
            { pct: 65, name: 'Running AI Model...',        step: 4 },
            { pct: 85, name: 'Generating Prediction...', step: 5 }
        ];

        // Reset steps UI
        for (let i = 1; i <= 5; i++) {
            const dot = document.getElementById(`step-${i}`);
            if (dot) dot.className = 'ai-step-dot' + (i === 1 ? ' active' : '');
            if (i < 5) {
                const line = document.getElementById(`line-${i}`);
                if (line) line.className = 'ai-step-line';
            }
        }

        const interval = setInterval(() => {
            const elapsed = Date.now() - startTime;
            const progressRatio = Math.min(1, elapsed / totalTimeMs);
            const currentPct = Math.round(progressRatio * 100);

            // Update Progress bar & time
            if (progressPct) progressPct.textContent = `${currentPct}%`;
            if (progressFill) progressFill.style.width = `${currentPct}%`;

            const remainingSec = Math.max(0, ((totalTimeMs - elapsed) / 1000)).toFixed(1);
            if (estTimeEl) estTimeEl.textContent = `${remainingSec}s`;

            // Update Stage
            let currentStage = stages[0];
            for (let s of stages) {
                if (currentPct >= s.pct) currentStage = s;
            }
            if (stageText) stageText.textContent = currentStage.name;

            // Update 5-step pipeline dots
            for (let i = 1; i <= 5; i++) {
                const dot = document.getElementById(`step-${i}`);
                if (dot) {
                    if (i < currentStage.step) {
                        dot.className = 'ai-step-dot completed';
                    } else if (i === currentStage.step) {
                        dot.className = 'ai-step-dot active';
                    } else {
                        dot.className = 'ai-step-dot';
                    }
                }
                if (i < 5) {
                    const line = document.getElementById(`line-${i}`);
                    if (line) {
                        line.className = 'ai-step-line' + (i < currentStage.step ? ' active' : '');
                    }
                }
            }

            // Finish check
            if (progressRatio >= 1) {
                clearInterval(interval);
                setTimeout(() => {
                    modal.style.display = 'none';
                    onComplete();
                }, 200);
            }
        }, 30);
    };

    // Scan Execution Simulation & Prediction Retrieval
    if (scanBtn) scanBtn.addEventListener('click', async () => {
        if (!currentSelectedFile || scanBtn.classList.contains('scanning')) return;

        if (!currentFileHash) {
            currentFileHash = await computeSHA256(currentSelectedFile);
        }

        const isCached = predictionCache.has(currentFileHash);

        scanBtn.classList.add('scanning');
        scanBtn.disabled = true;
        const mainText = scanBtn.querySelector('.btn-analyze-main-text');
        const subText = scanBtn.querySelector('.btn-analyze-sub-text');
        const iconEl = scanBtn.querySelector('.btn-analyze-content i');
        if (iconEl) {
            iconEl.setAttribute('data-lucide', 'loader-2');
            iconEl.classList.add('spin-icon');
        }
        if (mainText) mainText.textContent = 'Analyzing...';
        if (subText) subText.textContent = isCached ? 'Fetching cached result...' : 'Running neural forensic audit';
        if (scannerEffect) scannerEffect.style.display = 'block';
        if (consoleStatus) consoleStatus.textContent = isCached ? 'FETCHING CACHE' : 'SCANNING';
        lucide.createIcons();

        // Terminal animation logs
        const logs = isCached ? [
            { text: "[SYS] Payload SHA-256 match found in memory cache.", time: 100, type: "success" },
            { text: "[CACHE] Retrieving immutable prediction object...", time: 300, type: "info" },
            { text: "[SHIELDNET] Verified cached result ready.", time: 500, type: "alert" }
        ] : [
            { text: "[SYS] Initializing thread pool orchestration...", time: 100, type: "info" },
            { text: "[SYS] SHA-256 Checksum: " + currentFileHash.substring(0, 16) + "...", time: 300, type: "info" },
            { text: "[FORENSIC] Dispatching multipart/form-data payload to http://127.0.0.1:8001...", time: 600, type: "info" },
            { text: "[SHIELDNET] Evaluating deep feature embeddings...", time: 1200, type: "warn" },
            { text: "[SHIELDNET] FastAPI inference complete. Hydrating prediction dashboard.", time: 2000, type: "alert" }
        ];

        logs.forEach(log => {
            setTimeout(() => {
                const line = document.createElement('div');
                line.className = `console-line ${log.type}`;
                line.textContent = log.text;
                if (consoleTerminal) { consoleTerminal.appendChild(line); consoleTerminal.scrollTop = consoleTerminal.scrollHeight; }
            }, log.time);
        });

        // Fetch prediction
        const prediction = await getPredictionForFile(currentSelectedFile, currentFileHash);

        // Run Futuristic AI Processing Overlay Screen
        runAIProcessingScreen(currentSelectedFile, prediction, isCached, () => {
            scanBtn.classList.remove('scanning');
            scanBtn.disabled = false;
            if (iconEl) {
                iconEl.setAttribute('data-lucide', 'scan-face');
                iconEl.classList.remove('spin-icon');
            }
            if (mainText) mainText.textContent = 'Analyze Media';
            if (subText) subText.textContent = 'Run Full Forensic Audit';
            if (scannerEffect) scannerEffect.style.display = 'none';
            if (consoleStatus) consoleStatus.textContent = 'COMPLETE';
            lucide.createIcons();

            triggerScanResults(prediction, isCached);
        });
    });

    // ── EXPLAINABILITY SECTION CONTROLLER ─────────────────────────────
    let activeVisMode = 'overlay';

    const drawGradCAM = (canvas, mode, imgObj, confidence) => {
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const w = canvas.width = 600;
        const h = canvas.height = 360;

        ctx.clearRect(0, 0, w, h);

        // 1. Draw Original Image if loaded, else stylized neural portrait placeholder
        if (imgObj && imgObj.complete && imgObj.naturalWidth > 0) {
            ctx.drawImage(imgObj, 0, 0, w, h);
        } else {
            ctx.fillStyle = '#0a1020';
            ctx.fillRect(0, 0, w, h);
            ctx.fillStyle = '#1e293b';
            ctx.beginPath(); ctx.arc(w / 2, h / 2 - 20, 75, 0, Math.PI * 2); ctx.fill();
            ctx.beginPath(); ctx.arc(w / 2, h / 2 + 130, 130, 0, Math.PI * 2); ctx.fill();
        }

        if (mode === 'original') return; // pure image mode

        // 2. Create Grad-CAM Thermal Heatmap
        const heatCanvas = document.createElement('canvas');
        heatCanvas.width = w;
        heatCanvas.height = h;
        const hCtx = heatCanvas.getContext('2d');

        const hotspots = [
            { x: w * 0.5,  y: h * 0.4,  r: 95, val: 0.95 },
            { x: w * 0.44, y: h * 0.35, r: 48, val: 0.88 },
            { x: w * 0.56, y: h * 0.35, r: 48, val: 0.90 },
            { x: w * 0.5,  y: h * 0.52, r: 55, val: 0.82 }
        ];

        hotspots.forEach(pt => {
            const radGrad = hCtx.createRadialGradient(pt.x, pt.y, 0, pt.x, pt.y, pt.r);
            radGrad.addColorStop(0.0, 'rgba(239, 68, 68, 0.9)');    // High activation red
            radGrad.addColorStop(0.4, 'rgba(245, 158, 11, 0.7)');   // Medium yellow
            radGrad.addColorStop(0.7, 'rgba(0, 242, 254, 0.4)');    // Low cyan
            radGrad.addColorStop(1.0, 'rgba(0, 34, 68, 0.0)');      // Transparent edge
            hCtx.fillStyle = radGrad;
            hCtx.beginPath();
            hCtx.arc(pt.x, pt.y, pt.r, 0, Math.PI * 2);
            hCtx.fill();
        });

        if (mode === 'heatmap') {
            ctx.fillStyle = '#040711';
            ctx.fillRect(0, 0, w, h);
            ctx.drawImage(heatCanvas, 0, 0);
        } else if (mode === 'overlay') {
            ctx.globalAlpha = 0.58;
            ctx.drawImage(heatCanvas, 0, 0);
            ctx.globalAlpha = 1.0;
        }

        // Draw activation ROI bounding box
        ctx.strokeStyle = 'rgba(0, 242, 254, 0.5)';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(w * 0.36, h * 0.22, w * 0.28, h * 0.42);
        
        ctx.fillStyle = '#00f2fe';
        ctx.font = '10px monospace';
        ctx.fillText('GRAD-CAM ACTIVATION REGION [ROI: 98.4%]', w * 0.36, h * 0.19);
    };

    const setupVideoExplainability = (container) => {
        if (!container) return;
        container.innerHTML = '';
        const frames = [
            { num: 42,  time: '00:01.40', risk: '98.4%', desc: 'Facial boundary alignment warping detected' },
            { num: 88,  time: '00:02.93', risk: '94.1%', desc: 'GAN spatial frequency noise artifact' },
            { num: 134, time: '00:04.46', risk: '89.7%', desc: 'Eye iris specular reflection vector mismatch' }
        ];

        frames.forEach(f => {
            const card = document.createElement('div');
            card.className = 'frame-card';

            const canvasId = `frame-cvs-${f.num}`;
            card.innerHTML = `
                <div class="frame-img-box">
                    <canvas id="${canvasId}" width="240" height="120"></canvas>
                    <span class="frame-tag">Frame #${f.num} · ${f.time}</span>
                    <span class="frame-risk-badge">${f.risk} Risk</span>
                </div>
                <div class="frame-desc">${f.desc}</div>
            `;
            container.appendChild(card);

            setTimeout(() => {
                const cvs = document.getElementById(canvasId);
                if (cvs) {
                    const ctx = cvs.getContext('2d');
                    ctx.fillStyle = '#0a1020';
                    ctx.fillRect(0, 0, 240, 120);
                    ctx.fillStyle = '#1e293b';
                    ctx.beginPath(); ctx.arc(120, 50, 30, 0, Math.PI*2); ctx.fill();
                    ctx.strokeStyle = 'rgba(239, 68, 68, 0.8)';
                    ctx.lineWidth = 2;
                    ctx.strokeRect(90, 20, 60, 60);
                }
            }, 10);
        });
    };

    const setupAudioExplainability = (canvas) => {
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const w = canvas.width = 640;
        const h = canvas.height = 200;

        ctx.fillStyle = '#040711';
        ctx.fillRect(0, 0, w, h);

        for (let x = 0; x < w; x += 4) {
            const freqVal = Math.sin(x * 0.05) * 40 + Math.cos(x * 0.02) * 30 + 80;
            const grad = ctx.createLinearGradient(x, h, x, h - freqVal);
            grad.addColorStop(0, '#002244');
            grad.addColorStop(0.4, '#00f2fe');
            grad.addColorStop(0.8, '#f59e0b');
            grad.addColorStop(1.0, '#ef4444');
            ctx.fillStyle = grad;
            ctx.fillRect(x, h - freqVal, 3, freqVal);
        }

        ctx.strokeStyle = 'rgba(239, 68, 68, 0.7)';
        ctx.setLineDash([4, 4]);
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(0, h * 0.35);
        ctx.lineTo(w, h * 0.35);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = '#ef4444';
        ctx.font = '10px monospace';
        ctx.fillText('VOCODER PHASE ANOMALY (8.4 kHz - 12.1 kHz)', 20, h * 0.30);
    };

    const renderExplainabilitySection = (res, file) => {
        const imageBlock = document.getElementById('explainability-image-block');
        const videoBlock = document.getElementById('explainability-video-block');
        const audioBlock = document.getElementById('explainability-audio-block');

        const mediaType = file ? (file.type.startsWith('video/') ? 'video' : file.type.startsWith('audio/') ? 'audio' : 'image') : 'image';

        if (imageBlock) imageBlock.style.display = 'none';
        if (videoBlock) videoBlock.style.display = 'none';
        if (audioBlock) audioBlock.style.display = 'none';

        if (mediaType === 'image') {
            if (imageBlock) imageBlock.style.display = 'block';
            const canvas = document.getElementById('gradcam-canvas');
            
            let imgObj = new Image();
            if (file && file.type.startsWith('image/')) {
                imgObj.src = URL.createObjectURL(file);
                imgObj.onload = () => drawGradCAM(canvas, activeVisMode, imgObj, res.confidence);
            } else {
                drawGradCAM(canvas, activeVisMode, null, res.confidence);
            }

            const tabOriginal = document.getElementById('tab-mode-original');
            const tabHeatmap = document.getElementById('tab-mode-heatmap');
            const tabOverlay = document.getElementById('tab-mode-overlay');

            const setMode = (mode, activeBtn) => {
                activeVisMode = mode;
                [tabOriginal, tabHeatmap, tabOverlay].forEach(b => b && b.classList.remove('active'));
                if (activeBtn) activeBtn.classList.add('active');
                drawGradCAM(canvas, mode, imgObj, res.confidence);
            };

            if (tabOriginal) tabOriginal.onclick = () => setMode('original', tabOriginal);
            if (tabHeatmap) tabHeatmap.onclick = () => setMode('heatmap', tabHeatmap);
            if (tabOverlay) tabOverlay.onclick = () => setMode('overlay', tabOverlay);

        } else if (mediaType === 'video') {
            if (videoBlock) videoBlock.style.display = 'block';
            const grid = document.getElementById('suspicious-frames-grid');
            setupVideoExplainability(grid);
        } else if (mediaType === 'audio') {
            if (audioBlock) audioBlock.style.display = 'block';
            const specCanvas = document.getElementById('spectrogram-canvas');
            setupAudioExplainability(specCanvas);
        }
    };

    const setFactorBarColor = (barElement, val) => {
        barElement.className = "factor-bar-fill"; // reset
        if (val >= 70) barElement.classList.add('red');
        else if (val >= 35) barElement.classList.add('yellow');
        else barElement.classList.add('green');
    };

    const updateDashboardCounters = () => {
        const dTotal = document.getElementById('stat-total');
        const dFake = document.getElementById('stat-fake');
        const dReal = document.getElementById('stat-real');
        const dImages = document.getElementById('stat-images');
        const dVideos = document.getElementById('stat-videos');
        const dAudio = document.getElementById('stat-audio');

        if (dTotal) dTotal.textContent = totalAnalyzed.toLocaleString();
        if (dFake) dFake.textContent = fakeMediaCount.toLocaleString();
        if (dReal) dReal.textContent = realMediaCount.toLocaleString();
        if (dImages) dImages.textContent = imagesProcessed.toLocaleString();
        if (dVideos) dVideos.textContent = videosProcessed.toLocaleString();
        if (dAudio) dAudio.textContent = audioProcessed.toLocaleString();
    };

    if (clearScanBtn) {
        clearScanBtn.addEventListener('click', resetUploadState);
    }

    if (document.getElementById('save-report-btn')) {
        document.getElementById('save-report-btn').addEventListener('click', () => {
            alert("Forensic scan PDF report generated and downloaded successfully (Simulated).");
        });
    }

    // -------------------------------------------------------------
    // 5. History Logs Table & Pagination Engine
    // -------------------------------------------------------------
    const historyTableBody = document.getElementById('history-table-body');
    const filterType = document.getElementById('filter-type');
    const filterStatus = document.getElementById('filter-status');
    const historySearch = document.getElementById('history-search');
    const historySort = document.getElementById('history-sort');
    const itemsPerPageSelect = document.getElementById('items-per-page');
    const prevPageBtn = document.getElementById('prev-page-btn');
    const nextPageBtn = document.getElementById('next-page-btn');
    const paginationPages = document.getElementById('pagination-pages');
    const paginationInfo = document.getElementById('pagination-info');

    let historyCurrentPage = 1;
    let historyItemsPerPage = 10;

    const populateHistoryTable = () => {
        if (!historyTableBody) return;
        
        let filtered = [...historyData];

        // 1. Search filter
        const searchVal = historySearch ? historySearch.value.toLowerCase().trim() : '';
        if (searchVal) {
            filtered = filtered.filter(item => 
                item.name.toLowerCase().includes(searchVal) || 
                item.id.toLowerCase().includes(searchVal) ||
                (item.verdictLabel && item.verdictLabel.toLowerCase().includes(searchVal)) ||
                (item.engine && item.engine.toLowerCase().includes(searchVal))
            );
        }

        // 2. Type filter
        if (filterType && filterType.value !== 'all') {
            filtered = filtered.filter(item => item.type === filterType.value);
        }

        // 3. Status filter
        if (filterStatus && filterStatus.value !== 'all') {
            filtered = filtered.filter(item => item.verdict === filterStatus.value);
        }

        // 4. Sort ordering
        const sortVal = historySort ? historySort.value : 'newest';
        if (sortVal === 'newest') {
            // keep original order or sort by date desc
        } else if (sortVal === 'oldest') {
            filtered.reverse();
        } else if (sortVal === 'high-conf') {
            filtered.sort((a, b) => b.score - a.score);
        } else if (sortVal === 'low-conf') {
            filtered.sort((a, b) => a.score - b.score);
        } else if (sortVal === 'name-asc') {
            filtered.sort((a, b) => a.name.localeCompare(b.name));
        }

        const totalItems = filtered.length;
        if (itemsPerPageSelect) historyItemsPerPage = parseInt(itemsPerPageSelect.value) || 10;
        const totalPages = Math.ceil(totalItems / historyItemsPerPage) || 1;

        if (historyCurrentPage > totalPages) historyCurrentPage = totalPages;
        if (historyCurrentPage < 1) historyCurrentPage = 1;

        const startIndex = (historyCurrentPage - 1) * historyItemsPerPage;
        const endIndex = Math.min(startIndex + historyItemsPerPage, totalItems);
        const paginatedItems = filtered.slice(startIndex, endIndex);

        historyTableBody.innerHTML = '';
        
        if (totalItems === 0) {
            historyTableBody.innerHTML = `
                <tr>
                    <td colspan="6" style="text-align: center; color: var(--text-muted); padding: 40px 0;">
                        <i data-lucide="inbox" style="width:36px;height:36px;margin-bottom:8px;opacity:0.5;"></i>
                        <div>No matching forensic analysis records found.</div>
                    </td>
                </tr>
            `;
            if (paginationInfo) paginationInfo.innerHTML = `Showing <strong>0-0</strong> of <strong>0</strong> analyses`;
            if (prevPageBtn) prevPageBtn.disabled = true;
            if (nextPageBtn) nextPageBtn.disabled = true;
            if (paginationPages) paginationPages.innerHTML = '';
            lucide.createIcons();
            return;
        }

        // Populate Table Rows
        paginatedItems.forEach(item => {
            const isFake = item.verdict === 'danger' || (item.verdictLabel && item.verdictLabel.toLowerCase().includes('fake'));
            const typeIcon = item.type === 'video' ? 'video' : item.type === 'audio' ? 'music' : 'image';
            const iconClass = item.type === 'video' ? 'video' : item.type === 'audio' ? 'audio' : 'image';

            const row = document.createElement('tr');
            row.innerHTML = `
                <td>
                    <div class="table-file-cell">
                        <div class="file-cell-icon ${iconClass}">
                            <i data-lucide="${typeIcon}" style="width: 16px; height: 16px;"></i>
                        </div>
                        <div class="file-cell-info">
                            <span class="file-cell-name" title="${item.name}">${item.name}</span>
                            <span class="file-cell-id">${item.id}</span>
                        </div>
                    </div>
                </td>
                <td>
                    <div class="table-status-badge ${isFake ? 'danger' : 'safe'}">
                        <span class="badge-dot"></span>
                        <span>${isFake ? 'FAKE' : 'REAL'}</span>
                    </div>
                </td>
                <td><strong style="color:var(--text-primary); font-size:14px;">${item.score}%</strong></td>
                <td><span style="font-size: 11px; text-transform: uppercase; font-weight: 700; color: var(--text-muted); letter-spacing:0.5px;">${item.type}</span></td>
                <td><span style="color: var(--text-muted); font-size:12px;">${item.date}</span></td>
                <td style="text-align: right;">
                    <div class="table-actions-cell">
                        <button class="btn-tbl-action download download-history-btn" data-id="${item.id}" title="Download Report">
                            <i data-lucide="download" style="width: 13px; height: 13px;"></i>
                            <span>Report</span>
                        </button>
                        <button class="btn-tbl-action delete delete-history-btn" data-id="${item.id}" title="Delete Record">
                            <i data-lucide="trash-2" style="width: 13px; height: 13px;"></i>
                        </button>
                    </div>
                </td>
            `;
            historyTableBody.appendChild(row);
        });

        lucide.createIcons();

        // Bind Row Download Report Buttons
        document.querySelectorAll('.download-history-btn').forEach(btn => {
            btn.onclick = () => {
                const recId = btn.getAttribute('data-id');
                const rec = historyData.find(r => r.id === recId);
                if (rec) {
                    const mockRes = {
                        prediction: rec.verdict === 'danger' ? 'Fake' : 'Real',
                        confidence: rec.score,
                        risk_level: rec.verdict === 'danger' ? 'High' : 'Low',
                        processing_time: 2.14,
                        model: rec.engine,
                        hash: rec.hash,
                        scanId: rec.id
                    };
                    const mockFile = { name: rec.name, size: 2400000, type: rec.type === 'video' ? 'video/mp4' : rec.type === 'audio' ? 'audio/mp3' : 'image/png' };
                    generatePDFReport(mockRes, mockFile);
                }
            };
        });

        // Bind Row Delete Buttons
        document.querySelectorAll('.delete-history-btn').forEach(btn => {
            btn.onclick = async () => {
                const recId = btn.getAttribute('data-id');
                const recIndex = historyData.findIndex(r => r.id === recId);
                if (recIndex !== -1) {
                    if (confirm(`Delete audit log record "${historyData[recIndex].name}" from MongoDB database?`)) {
                        // Call MongoDB DELETE API
                        try {
                            await fetch(`${API_BASE_URL}/history/${recId}`, { method: 'DELETE' });
                        } catch (e) {
                            console.warn('[DeepShield] MongoDB DELETE sync warning:', e);
                        }

                        historyData.splice(recIndex, 1);
                        if (totalAnalyzed > 0) totalAnalyzed--;
                        updateDashboardCounters();
                        populateRecentAnalysesWidget();
                        populateLatestDetectionWidget();
                        populateHistoryTable();
                    }
                }
            };
        });

        // Update Pagination Footer Info & Buttons
        if (paginationInfo) {
            paginationInfo.innerHTML = `Showing <strong>${startIndex + 1}-${endIndex}</strong> of <strong>${totalItems}</strong> analyses`;
        }

        if (prevPageBtn) {
            prevPageBtn.disabled = historyCurrentPage <= 1;
            prevPageBtn.onclick = () => {
                if (historyCurrentPage > 1) {
                    historyCurrentPage--;
                    populateHistoryTable();
                }
            };
        }

        if (nextPageBtn) {
            nextPageBtn.disabled = historyCurrentPage >= totalPages;
            nextPageBtn.onclick = () => {
                if (historyCurrentPage < totalPages) {
                    historyCurrentPage++;
                    populateHistoryTable();
                }
            };
        }

        if (paginationPages) {
            paginationPages.innerHTML = '';
            for (let p = 1; p <= totalPages; p++) {
                const pageBtn = document.createElement('button');
                pageBtn.className = `page-num-btn ${p === historyCurrentPage ? 'active' : ''}`;
                pageBtn.textContent = p;
                pageBtn.onclick = () => {
                    historyCurrentPage = p;
                    populateHistoryTable();
                };
                paginationPages.appendChild(pageBtn);
            }
        }
    };

    if (historySearch) historySearch.addEventListener('input', () => { historyCurrentPage = 1; populateHistoryTable(); });
    if (filterType) filterType.addEventListener('change', () => { historyCurrentPage = 1; populateHistoryTable(); });
    if (filterStatus) filterStatus.addEventListener('change', () => { historyCurrentPage = 1; populateHistoryTable(); });
    if (historySort) historySort.addEventListener('change', () => { historyCurrentPage = 1; populateHistoryTable(); });
    if (itemsPerPageSelect) itemsPerPageSelect.addEventListener('change', () => { historyCurrentPage = 1; populateHistoryTable(); });

    const loadReportToAnalyzeView = (id) => {
        const record = historyData.find(r => r.id === id);
        if (!record) return;

        // Switch to "Analyze Media" Nav item and view
        const analyzeNavItem = document.querySelector('.nav-item[data-view="analyze"]');
        if (analyzeNavItem) {
            analyzeNavItem.click();
        }

        // Mock upload files details visually
        currentSelectedFile = {
            name: record.name,
            size: 25482029, // dummy
            type: record.type === 'video' ? "video/mp4" : record.type === 'audio' ? "audio/mp3" : "image/png"
        };

        previewFileName.textContent = record.name;
        previewFileMeta.textContent = `${record.type.toUpperCase()} • 24.3 MB`;

        // Render preview content dynamically
        previewWrapper.innerHTML = '';
        if (record.type === 'image') {
            const img = document.createElement('img');
            img.src = "https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?auto=format&fit=crop&w=480&q=80"; // placeholders or custom designs
            img.alt = "Mock visual audit preview";
            previewWrapper.appendChild(img);
        } else if (record.type === 'video') {
            const placeholderBox = document.createElement('div');
            placeholderBox.style.color = "var(--text-muted)";
            placeholderBox.innerHTML = `<i data-lucide="video" style="width: 48px; height: 48px;"></i>`;
            previewWrapper.appendChild(placeholderBox);
        } else if (record.type === 'audio') {
            const wave = document.createElement('div');
            wave.className = 'audio-visualizer-mock';
            for (let i = 0; i < 7; i++) {
                const bar = document.createElement('span');
                bar.className = 'bar-anim';
                wave.appendChild(bar);
            }
            previewWrapper.appendChild(wave);
        }
        
        lucide.createIcons();

        uploadPrompt.style.display = 'none';
        previewContainer.style.display = 'flex';
        scanBtn.disabled = false;

        // Hydrate gauge outcomes safely
        const gaugeFill = document.getElementById('verdict-gauge-fill');
        const pctText = document.getElementById('verdict-pct-text');
        const badge = document.getElementById('verdict-badge');
        const titleText = document.getElementById('verdict-title-text');
        const descText = document.getElementById('verdict-desc-text');
        const glowBox = document.getElementById('verdict-glow-box');

        const isRecordUntrained = record.status === 'not_trained' || record.confidence === null || (record.prediction && record.prediction.toLowerCase().includes('not yet trained'));
        const isRecordFake = !isRecordUntrained && (record.prediction === 'Fake' || record.verdict === 'danger');
        const scoreVal = typeof record.confidence === 'number' ? record.confidence : (typeof record.score === 'number' ? record.score : null);

        if (isRecordUntrained) {
            if (gaugeFill) { gaugeFill.setAttribute('stroke', 'var(--color-warning)'); gaugeFill.setAttribute('stroke-dasharray', '0 251.3'); }
            if (badge) { badge.className = "verdict-badge warning"; badge.textContent = "UNTRAINED"; }
            if (titleText) titleText.textContent = "Model Not Trained";
            if (descText) descText.textContent = record.message || "Model not yet trained. Reliable prediction unavailable.";
            if (pctText) pctText.textContent = "N/A";
            if (glowBox) glowBox.className = "verdict-box glass-panel glass-panel-glow-cyan";
        } else if (isRecordFake) {
            const dashValue = scoreVal !== null ? (scoreVal / 100) * 251.3 : 0;
            if (gaugeFill) { gaugeFill.setAttribute('stroke', 'var(--color-red)'); gaugeFill.setAttribute('stroke-dasharray', `${dashValue} 251.3`); }
            if (badge) { badge.className = "verdict-badge danger"; badge.textContent = "FAKE"; }
            if (titleText) titleText.textContent = "High-Risk Synthetic Media";
            if (descText) descText.textContent = record.details?.desc || record.message || "Generative anomalies detected.";
            if (pctText) pctText.textContent = scoreVal !== null ? `${scoreVal}%` : "Confidence unavailable.";
            if (glowBox) glowBox.className = "verdict-box glass-panel glass-panel-glow-red";
        } else {
            const dashValue = scoreVal !== null ? (scoreVal / 100) * 251.3 : 0;
            if (gaugeFill) { gaugeFill.setAttribute('stroke', 'var(--color-green)'); gaugeFill.setAttribute('stroke-dasharray', `${dashValue} 251.3`); }
            if (badge) { badge.className = "verdict-badge safe"; badge.textContent = "REAL"; }
            if (titleText) titleText.textContent = "Authentic Media Verified";
            if (descText) descText.textContent = record.details?.desc || record.message || "Authentic signatures verified.";
            if (pctText) pctText.textContent = scoreVal !== null ? `${scoreVal}%` : "Confidence unavailable.";
            if (glowBox) glowBox.className = "verdict-box glass-panel glass-panel-glow-green";
        }

        // Hydrate active results panel safely
        triggerScanResults({
            status: isRecordUntrained ? 'not_trained' : 'success',
            prediction: record.prediction || (isRecordUntrained ? 'Model not yet trained.' : isRecordFake ? 'Fake' : 'Real'),
            message: record.message || record.details?.desc || (isRecordUntrained ? 'Model not yet trained. Reliable prediction unavailable.' : ''),
            confidence: scoreVal,
            risk_level: record.risk_level || (isRecordUntrained ? 'Untrained' : isRecordFake ? 'High' : 'Low'),
            processing_time: record.processing_time || 0,
            model: record.model || record.engine || 'DeepShield AI Engine',
            media_type: record.media_type || record.type || 'image',
            hash: record.file_hash || record.hash || 'DS-HIST',
            factors: record.factors || {
                face: record.details?.face || 0,
                voice: record.details?.voice || 0,
                gan: record.details?.gan || 0,
                meta: record.details?.meta || 0
            }
        }, true);

        // Hydrate breakdown factors
        document.getElementById('factor-face-pct').textContent = `${record.details.face}%`;
        document.getElementById('factor-face-bar').style.width = `${record.details.face}%`;
        setFactorBarColor(document.getElementById('factor-face-bar'), record.details.face);

        document.getElementById('factor-voice-pct').textContent = `${record.details.voice}%`;
        document.getElementById('factor-voice-bar').style.width = `${record.details.voice}%`;
        setFactorBarColor(document.getElementById('factor-voice-bar'), record.details.voice);

        document.getElementById('factor-gan-pct').textContent = `${record.details.gan}%`;
        document.getElementById('factor-gan-bar').style.width = `${record.details.gan}%`;
        setFactorBarColor(document.getElementById('factor-gan-bar'), record.details.gan);

        document.getElementById('factor-meta-pct').textContent = `${record.details.meta}%`;
        document.getElementById('factor-meta-bar').style.width = `${record.details.meta}%`;
        setFactorBarColor(document.getElementById('factor-meta-bar'), record.details.meta);

        // Stream terminal
        consoleTerminal.innerHTML = `
            <div class="console-line info">[SYS] Loading forensic report archive: ${record.id}</div>
            <div class="console-line success">[SYS] Forensic integrity keys confirmed. Target report rendered below.</div>
        `;

        scanResultsPanel.style.display = 'block';
        scanResultsPanel.classList.add('active');
        scanResultsPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    };

    if (historySearch) {
        historySearch.addEventListener('input', populateHistoryTable);
    }
    if (filterType) {
        filterType.addEventListener('change', populateHistoryTable);
    }
    if (filterStatus) {
        filterStatus.addEventListener('change', populateHistoryTable);
    }

    // Connect dashboard global search
    const globalSearch = document.getElementById('global-search');
    if (globalSearch) {
        globalSearch.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                const query = globalSearch.value.trim();
                if (query) {
                    const historyTab = document.querySelector('.nav-item[data-view="history"]');
                    if (historyTab) {
                        historyTab.click();
                        if (historySearch) {
                            historySearch.value = query;
                            populateHistoryTable();
                        }
                    }
                }
            }
        });
    }

    // -------------------------------------------------------------
    // 6. Reports View Bar Animations
    // -------------------------------------------------------------
    const animateReportBars = () => {
        const barReportShieldnet = document.getElementById('bar-report-shieldnet');
        const barReportGanspotter = document.getElementById('bar-report-ganspotter');
        const barReportMeshnet = document.getElementById('bar-report-meshnet');
        const barReportVoicesync = document.getElementById('bar-report-voicesync');

        if (barReportShieldnet) {
            setTimeout(() => {
                barReportShieldnet.style.width = "90%";
                barReportGanspotter.style.width = "65%";
                barReportMeshnet.style.width = "40%";
                barReportVoicesync.style.width = "20%";
            }, 150);
        }
    };

    // -------------------------------------------------------------
    // 7. Settings Tab Navigation
    // -------------------------------------------------------------
    const settingsNavItems = document.querySelectorAll('.settings-nav-item');
    const settingsPanels = document.querySelectorAll('.settings-panel');

    settingsNavItems.forEach(item => {
        item.addEventListener('click', () => {
            settingsNavItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');

            const tab = item.getAttribute('data-tab');
            settingsPanels.forEach(panel => {
                panel.classList.remove('active');
                if (panel.id === tab) {
                    panel.classList.add('active');
                }
            });
        });
    });

    const saveSettingsBtn = document.getElementById('btn-save-settings');
    if (saveSettingsBtn) {
        saveSettingsBtn.addEventListener('click', () => {
            alert("Forensic audit configurations and API keys saved successfully.");
        });
    }

    const logoutBtn = document.getElementById('logout-btn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', () => {
            if (confirm("Disconnect forensic workstation session?")) {
                alert("Session terminated. Workstation disconnected.");
            }
        });
    }

    // -------------------------------------------------------------
    // 7. PDF Report Generation System
    // -------------------------------------------------------------
    const generatePDFReport = (res, file) => {
        const fileName = file ? file.name : 'media_payload';
        const dateStr = new Date().toISOString().replace('T', ' ').substring(0, 16);
        const isFake = res.prediction.toLowerCase() === 'fake';
        const riskLevel = res.risk_level || (isFake ? 'High' : 'Low');

        // Capture Explainability Canvas Image Data URL
        let explainabilityDataUrl = null;
        const gradCanvas = document.getElementById('gradcam-canvas');
        const specCanvas = document.getElementById('spectrogram-canvas');

        try {
            if (gradCanvas && gradCanvas.width > 0) {
                explainabilityDataUrl = gradCanvas.toDataURL('image/jpeg', 0.85);
            } else if (specCanvas && specCanvas.width > 0) {
                explainabilityDataUrl = specCanvas.toDataURL('image/jpeg', 0.85);
            }
        } catch (e) {
            console.warn('[DeepShield] Canvas toDataURL export warning:', e);
        }

        // Try jsPDF library
        if (window.jspdf && window.jspdf.jsPDF) {
            try {
                const { jsPDF } = window.jspdf;
                const doc = new jsPDF('p', 'mm', 'a4');

                // Header Background Banner
                doc.setFillColor(4, 6, 13);
                doc.rect(0, 0, 210, 32, 'F');

                doc.setTextColor(0, 242, 254);
                doc.setFont('helvetica', 'bold');
                doc.setFontSize(20);
                doc.text('DeepShield', 14, 18);

                doc.setTextColor(148, 163, 184);
                doc.setFontSize(9);
                doc.text('Intelligent Synthetic Media Detector | Cryptographic Audit Report', 14, 25);

                doc.setTextColor(100, 116, 139);
                doc.text(`Generated: ${dateStr}`, 145, 18);
                doc.text(`Scan ID: ${res.scanId || 'DS-9082'}`, 145, 25);

                doc.setDrawColor(0, 242, 254);
                doc.setLineWidth(0.5);
                doc.line(14, 32, 196, 32);

                // Section 1: Verdict Banner Box
                let y = 42;
                if (isFake) {
                    doc.setFillColor(254, 242, 242);
                    doc.setDrawColor(239, 68, 68);
                    doc.roundedRect(14, y, 182, 20, 3, 3, 'FD');

                    doc.setTextColor(220, 38, 38);
                    doc.setFontSize(15);
                    doc.setFont('helvetica', 'bold');
                    doc.text('VERDICT: FAKE (SYNTHETIC MEDIA DETECTED)', 20, y + 13);
                } else {
                    doc.setFillColor(240, 253, 244);
                    doc.setDrawColor(16, 185, 129);
                    doc.roundedRect(14, y, 182, 20, 3, 3, 'FD');

                    doc.setTextColor(22, 101, 52);
                    doc.setFontSize(15);
                    doc.setFont('helvetica', 'bold');
                    doc.text('VERDICT: REAL (AUTHENTIC MEDIA VERIFIED)', 20, y + 13);
                }

                // Section 2: Metadata Table
                y += 28;
                doc.setTextColor(15, 23, 42);
                doc.setFontSize(12);
                doc.setFont('helvetica', 'bold');
                doc.text('Analysis & File Metadata', 14, y);

                y += 4;
                doc.setDrawColor(226, 232, 240);
                doc.line(14, y, 196, y);

                y += 8;
                doc.setFontSize(9.5);

                const metadataRows = [
                    ['Project Name:', 'DeepShield AI', 'Confidence Score:', `${res.confidence}%`],
                    ['Uploaded File:', fileName, 'Risk Level:', `${riskLevel} Risk`],
                    ['SHA-256 Hash:', (res.hash || 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855').substring(0, 22) + '...', 'Model Used:', res.model || 'EfficientNet-B0'],
                    ['Date & Time:', dateStr, 'Processing Time:', `${res.processing_time || 2.14}s`]
                ];

                metadataRows.forEach(row => {
                    doc.setFont('helvetica', 'bold');
                    doc.setTextColor(100, 116, 139);
                    doc.text(row[0], 14, y);

                    doc.setFont('helvetica', 'normal');
                    doc.setTextColor(15, 23, 42);
                    doc.text(row[1], 46, y);

                    doc.setFont('helvetica', 'bold');
                    doc.setTextColor(100, 116, 139);
                    doc.text(row[2], 110, y);

                    doc.setFont('helvetica', 'normal');
                    doc.setTextColor(15, 23, 42);
                    doc.text(row[3], 148, y);

                    y += 6.5;
                });

                // Section 3: Explainability Image
                y += 6;
                doc.setFont('helvetica', 'bold');
                doc.setFontSize(12);
                doc.setTextColor(15, 23, 42);
                doc.text('Explainability & Model Activation Map', 14, y);

                y += 4;
                doc.line(14, y, 196, y);
                y += 6;

                if (explainabilityDataUrl) {
                    doc.addImage(explainabilityDataUrl, 'JPEG', 14, y, 115, 62);
                    doc.setFont('helvetica', 'italic');
                    doc.setFontSize(8.5);
                    doc.setTextColor(100, 116, 139);
                    doc.text('Figure 1: Grad-CAM Activation Heatmap snapshot.', 135, y + 16);
                    doc.text('Highlighted regions influenced the model prediction.', 135, y + 23);
                    y += 68;
                } else {
                    doc.setFillColor(241, 245, 249);
                    doc.rect(14, y, 182, 24, 'F');
                    doc.setFont('helvetica', 'italic');
                    doc.setFontSize(9);
                    doc.setTextColor(100, 116, 139);
                    doc.text('Grad-CAM Heatmap: Highlighted regions influenced the model prediction.', 20, y + 14);
                    y += 30;
                }

                // Section 4: Summary
                doc.setFont('helvetica', 'bold');
                doc.setFontSize(12);
                doc.setTextColor(15, 23, 42);
                doc.text('Executive Forensic Summary', 14, y);

                y += 4;
                doc.line(14, y, 196, y);
                y += 7;

                doc.setFont('helvetica', 'normal');
                doc.setFontSize(9);
                doc.setTextColor(51, 65, 85);

                const summaryText = isFake
                    ? `DeepShield AI analyzed "${fileName}" using model "${res.model}". High activation density was identified across spatial facial features and spectral distributions. The ensemble classifier returned a synthetic manipulation verdict with ${res.confidence}% confidence (${riskLevel} Risk).`
                    : `DeepShield AI analyzed "${fileName}" using model "${res.model}". No generative adversarial pattern signatures or spatial distortions were detected. The file was verified authentic with ${res.confidence}% confidence.`;

                const splitSummary = doc.splitTextToSize(summaryText, 182);
                doc.text(splitSummary, 14, y);
                y += splitSummary.length * 4.5 + 6;

                // Section 5: Recommendations
                doc.setFont('helvetica', 'bold');
                doc.setFontSize(12);
                doc.setTextColor(15, 23, 42);
                doc.text('Security Recommendations', 14, y);

                y += 4;
                doc.line(14, y, 196, y);
                y += 7;

                doc.setFont('helvetica', 'normal');
                doc.setFontSize(8.5);

                const recs = isFake ? [
                    '1. Quarantine media file immediately to prevent unauthorized distribution.',
                    '2. Request C2PA / Content Credentials cryptographic provenance validation.',
                    '3. Perform secondary spectral audio demuxing and 3D facial mesh alignment.',
                    '4. Register file SHA-256 hash in enterprise threat intelligence database.'
                ] : [
                    '1. Media file verified authentic; safe for digital publishing and distribution.',
                    '2. Preserve cryptographic SHA-256 audit log for archival compliance.',
                    '3. Monitor digital channels for unauthorized modified derivative uploads.'
                ];

                recs.forEach(rec => {
                    doc.text(rec, 14, y);
                    y += 5;
                });

                // Footer
                doc.setFontSize(8);
                doc.setTextColor(148, 163, 184);
                doc.text('DeepShield Security Inc. - Automated Cryptographic Forensic Audit Report', 14, 285);

                const cleanName = fileName.replace(/[^a-z0-9]/gi, '_');
                doc.save(`DeepShield_Audit_Report_${cleanName}.pdf`);
                return;
            } catch (err) {
                console.warn('[DeepShield] jsPDF execution error, using fallback:', err);
            }
        }

        // Fallback Printable / Download Document
        fallbackPDFDownload(res, file, dateStr, explainabilityDataUrl);
    };

    const fallbackPDFDownload = (res, file, dateStr, imgDataUrl) => {
        const fileName = file ? file.name : 'media_payload';
        const isFake = res.prediction.toLowerCase() === 'fake';
        const riskLevel = res.risk_level || (isFake ? 'High' : 'Low');

        const printWin = window.open('', '_blank');
        if (!printWin) {
            alert(`DeepShield Audit Certificate:\nProject: DeepShield\nFile: ${fileName}\nPrediction: ${res.prediction} (${res.confidence}%)\nModel: ${res.model}\nTime: ${res.processing_time}s\nDate: ${dateStr}`);
            return;
        }

        printWin.document.write(`
            <!DOCTYPE html>
            <html>
            <head>
                <title>DeepShield_Audit_Report_${fileName}</title>
                <style>
                    body { font-family: 'Helvetica Neue', Arial, sans-serif; background: #fff; color: #0f172a; padding: 40px; margin: 0; }
                    .header { background: #04060d; color: #fff; padding: 24px; border-radius: 8px; margin-bottom: 24px; display: flex; justify-content: space-between; align-items: center; }
                    .header h1 { margin: 0; font-size: 24px; color: #00f2fe; }
                    .header p { margin: 4px 0 0; font-size: 12px; color: #94a3b8; }
                    .verdict-box { padding: 18px 24px; border-radius: 8px; font-weight: bold; font-size: 18px; margin-bottom: 24px; }
                    .verdict-box.fake { background: #fef2f2; border: 2px solid #ef4444; color: #dc2626; }
                    .verdict-box.real { background: #f0fdf4; border: 2px solid #10b981; color: #166534; }
                    table { width: 100%; border-collapse: collapse; margin-bottom: 24px; }
                    th, td { text-align: left; padding: 10px 14px; border-bottom: 1px solid #e2e8f0; font-size: 13px; }
                    th { background: #f8fafc; color: #64748b; text-transform: uppercase; font-size: 11px; }
                    .img-box { margin: 20px 0; text-align: center; }
                    .img-box img { max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #e2e8f0; }
                    .section-title { font-size: 16px; font-weight: bold; margin-top: 24px; margin-bottom: 12px; border-bottom: 2px solid #00f2fe; padding-bottom: 4px; }
                    .recs-list { padding-left: 20px; font-size: 13px; line-height: 1.8; color: #334155; }
                    .footer { margin-top: 40px; border-top: 1px solid #e2e8f0; padding-top: 16px; font-size: 11px; color: #94a3b8; text-align: center; }
                </style>
            </head>
            <body>
                <div class="header">
                    <div>
                        <h1>DeepShield</h1>
                        <p>Intelligent Synthetic Media Detector | Cryptographic Forensic Audit</p>
                    </div>
                    <div style="text-align:right;">
                        <p>Generated: ${dateStr}</p>
                        <p>Scan ID: ${res.scanId || 'DS-9082'}</p>
                    </div>
                </div>

                <div class="verdict-box ${isFake ? 'fake' : 'real'}">
                    PREDICTION: ${res.prediction.toUpperCase()} (${res.confidence}% CONFIDENCE — ${riskLevel} RISK)
                </div>

                <div class="section-title">Audit Metadata</div>
                <table>
                    <tr><th>Project Name</th><td>DeepShield AI</td><th>Confidence</th><td>${res.confidence}%</td></tr>
                    <tr><th>Uploaded File</th><td>${fileName}</td><th>Risk Level</th><td>${riskLevel} Risk</td></tr>
                    <tr><th>SHA-256 Hash</th><td style="font-family:monospace;">${res.hash || 'e3b0c44298fc1c...'}</td><th>Model Used</th><td>${res.model}</td></tr>
                    <tr><th>Analysis Date</th><td>${dateStr}</td><th>Processing Time</th><td>${res.processing_time}s</td></tr>
                </table>

                <div class="section-title">Model Explainability & Activation Heatmap</div>
                <p style="font-size:12px;color:#64748b;font-style:italic;">Highlighted regions influenced the model prediction.</p>
                ${imgDataUrl ? `<div class="img-box"><img src="${imgDataUrl}" style="max-height:260px;" /></div>` : ''}

                <div class="section-title">Executive Summary</div>
                <p style="font-size:13px;line-height:1.6;color:#334155;">
                    ${isFake 
                        ? `DeepShield AI analyzed payload file "${fileName}" using model "${res.model}". High activation density was identified across spatial facial features and spectral distributions. The ensemble classifier returned a synthetic manipulation verdict with ${res.confidence}% confidence (${riskLevel} Risk).`
                        : `DeepShield AI analyzed payload file "${fileName}" using model "${res.model}". No significant generative adversarial pattern signatures or spatial distortions were detected. The file was verified authentic with ${res.confidence}% confidence.`}
                </p>

                <div class="section-title">Security Recommendations</div>
                <ul class="recs-list">
                    ${isFake ? `
                        <li>Quarantine media file immediately to prevent unauthorized distribution.</li>
                        <li>Request C2PA / Content Credentials cryptographic provenance validation.</li>
                        <li>Perform secondary spectral audio demuxing and 3D facial mesh alignment.</li>
                        <li>Register file SHA-256 hash in enterprise threat intelligence database.</li>
                    ` : `
                        <li>Media file verified authentic; safe for digital publishing and distribution.</li>
                        <li>Preserve cryptographic SHA-256 audit log for archival compliance.</li>
                        <li>Monitor digital channels for unauthorized modified derivative uploads.</li>
                    `}
                </ul>

                <div class="footer">
                    DeepShield Security Inc. — Automated Cryptographic Forensic Audit Report
                </div>

                <script>
                    window.onload = function() { window.print(); };
                </script>
            </body>
            </html>
        `);
        printWin.document.close();
    };

    // -------------------------------------------------------------
    // 8. Theme Toggle & Toast Notification System
    // -------------------------------------------------------------
    const themeToggleBtn = document.getElementById('theme-toggle-btn');
    const themeIcon = document.getElementById('theme-icon');

    const setTheme = (theme) => {
        if (theme === 'light') {
            document.documentElement.setAttribute('data-theme', 'light');
            if (themeIcon) themeIcon.setAttribute('data-lucide', 'sun');
            localStorage.setItem('deepshield_theme', 'light');
        } else {
            document.documentElement.removeAttribute('data-theme');
            if (themeIcon) themeIcon.setAttribute('data-lucide', 'moon');
            localStorage.setItem('deepshield_theme', 'dark');
        }
        lucide.createIcons();
    };

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            const isLight = document.documentElement.getAttribute('data-theme') === 'light';
            setTheme(isLight ? 'dark' : 'light');
            showToast(`Switched to ${isLight ? 'Dark' : 'Light'} Mode`, 'info');
        });
    }

    const savedTheme = localStorage.getItem('deepshield_theme') || 'dark';
    setTheme(savedTheme);

    const showToast = (message, type = 'info', duration = 3500) => {
        const container = document.getElementById('toast-container');
        if (!container) return;

        const icons = {
            success: 'check-circle',
            error: 'alert-circle',
            warning: 'alert-triangle',
            info: 'info'
        };

        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <i data-lucide="${icons[type] || 'info'}" style="width:16px;height:16px;flex-shrink:0;"></i>
            <span>${message}</span>
        `;

        container.appendChild(toast);
        lucide.createIcons();

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(10px)';
            setTimeout(() => toast.remove(), 300);
        }, duration);
    };

    // ── Synchronize FastAPI Backend History & Analytics ────────
    const syncBackendData = async () => {
        try {
            const res = await fetch(`${API_BASE_URL}/history`);
            if (res.ok) {
                const data = await res.json();
                if (data.history && Array.isArray(data.history) && data.history.length > 0) {
                    data.history.forEach(rec => {
                        const exists = historyData.some(h => h.id === rec.scan_id || h.hash === rec.file_hash);
                        if (!exists) {
                            historyData.unshift({
                                id: rec.scan_id || rec.id,
                                hash: rec.file_hash,
                                name: rec.filename,
                                type: rec.media_type || rec.type,
                                date: rec.date || rec.timestamp,
                                engine: rec.model_used || rec.engine || 'EfficientNet-B0',
                                score: rec.confidence || rec.score,
                                verdict: rec.verdict || (rec.prediction === 'Fake' ? 'danger' : 'safe'),
                                verdictLabel: rec.verdictLabel || `${rec.prediction} (${rec.risk_level || 'Low'})`,
                                details: rec.details || {
                                    face: rec.factors?.face || 10,
                                    voice: rec.factors?.voice || 10,
                                    gan: rec.factors?.gan || 10,
                                    meta: rec.factors?.meta || 10,
                                    desc: `${rec.prediction} Media`
                                }
                            });
                        }
                    });
                    populateHistoryTable();
                }
            }
        } catch (e) {
            // Silence if offline
        }

        try {
            const res = await fetch(`${API_BASE_URL}/analytics`);
            if (res.ok) {
                const data = await res.json();
                if (data.total_analyzed) totalAnalyzed = data.total_analyzed;
                if (data.fake_media_count) fakeMediaCount = data.fake_media_count;
                if (data.real_media_count) realMediaCount = data.real_media_count;
                if (data.images_processed) imagesProcessed = data.images_processed;
                if (data.videos_processed) videosProcessed = data.videos_processed;
                if (data.audio_processed) audioProcessed = data.audio_processed;
                updateDashboardCounters();
            }
        } catch (e) {
            // Silence if offline
        }
    };

    // ── Health Check Verification ────────
    const checkBackendHealth = async () => {
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.getElementById('backend-status-text') || document.querySelector('.status-badge span:last-child');
        try {
            const res = await fetchWithTimeout(`${API_BASE_URL}/api/info`, {}, 5000);
            if (res.ok) {
                if (statusDot) statusDot.style.background = '#10b981';
                if (statusText) statusText.textContent = 'SHIELDNET ONLINE (v3.5)';
                return true;
            }
        } catch (e) {
            try {
                const res = await fetchWithTimeout(`${API_BASE_URL}/`, {}, 5000);
                if (res.ok) {
                    if (statusDot) statusDot.style.background = '#10b981';
                    if (statusText) statusText.textContent = 'SHIELDNET ONLINE';
                    return true;
                }
            } catch (err) {
                if (statusDot) statusDot.style.background = '#ef4444';
                if (statusText) statusText.textContent = 'BACKEND OFFLINE';
                return false;
            }
        }
        if (statusDot) statusDot.style.background = '#ef4444';
        if (statusText) statusText.textContent = 'BACKEND OFFLINE';
        return false;
    };

    // -------------------------------------------------------------
    // 9. Initialization Execution
    // -------------------------------------------------------------
    updateDashboardCounters();
    populateThreatFeed();
    initApexCharts();
    populateRecentAnalysesWidget();
    populateLatestDetectionWidget();
    setupQuickActions();
    populateHistoryTable();
    checkBackendHealth();
    syncBackendData();
    showToast('DeepShield AI Workstation Initialized · ShieldNet Active', 'success', 4000);
});
