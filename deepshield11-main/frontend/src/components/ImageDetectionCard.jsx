import React, { useState } from 'react';

const API_BASE_URL = 'http://127.0.0.1:8001';

const ImageDetectionCard = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState('');

    const handleFileChange = (e) => {
        if (e.target.files && e.target.files[0]) {
            setSelectedFile(e.target.files[0]);
            setResult(null);
            setError('');
        }
    };

    const handleAnalyze = async () => {
        if (!selectedFile) {
            setError('Please select an image file first.');
            return;
        }

        setLoading(true);
        setError('');
        setResult(null);

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch(`${API_BASE_URL}/predict/image`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `Server error (Status ${response.status})`);
            }

            const data = await response.json();
            setResult(data);
        } catch (err) {
            console.error('Image Prediction Error:', err);
            setError(err.message || 'Network error or backend server unavailable.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="detection-card">
            <div className="card-header">
                <div className="icon-wrapper image-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <rect width="18" height="18" x="3" y="3" rx="2" ry="2"/>
                        <circle cx="9" cy="9" r="2"/>
                        <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>
                    </svg>
                </div>
                <div>
                    <h3 className="card-title">Image Detection</h3>
                    <p className="card-subtitle">Detect AI-generated or manipulated images</p>
                </div>
            </div>

            <div className="upload-area">
                <label htmlFor="image-file-input" className="upload-btn">
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                        <polyline points="17 8 12 3 7 8"/>
                        <line x1="12" x2="12" y1="3" y2="15"/>
                    </svg>
                    <span>{selectedFile ? selectedFile.name : "Upload Image"}</span>
                </label>
                <input
                    id="image-file-input"
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    style={{ display: 'none' }}
                />
            </div>

            <button
                className="analyze-btn"
                onClick={handleAnalyze}
                disabled={!selectedFile || loading}
            >
                {loading ? "Analyzing..." : "Analyze Image"}
            </button>

            {error && <div className="error-message">{error}</div>}

            <div className="result-placeholder">
                <div className="placeholder-row">
                    <span className="placeholder-label">Prediction</span>
                    <span className={`placeholder-value ${result ? (result.prediction.toLowerCase().includes('real') ? 'val-real' : 'val-fake') : ''}`}>
                        {result ? result.prediction : "--"}
                    </span>
                </div>
                <div className="placeholder-row">
                    <span className="placeholder-label">Confidence</span>
                    <span className="placeholder-value">
                        {result ? `${result.confidence} %` : "-- %"}
                    </span>
                </div>
                <div className="placeholder-row">
                    <span className="placeholder-label">Risk Level</span>
                    <span className="placeholder-value">
                        {result ? (result.risk_level ? `${result.risk_level} Risk` : (result.prediction.toLowerCase().includes('real') ? 'Low Risk' : 'High Risk')) : "--"}
                    </span>
                </div>
                <div className="placeholder-row">
                    <span className="placeholder-label">Processing Time</span>
                    <span className="placeholder-value">
                        {result ? `${result.processing_time || 0.45} s` : "-- s"}
                    </span>
                </div>
                <div className="placeholder-row">
                    <span className="placeholder-label">Model Used</span>
                    <span className="placeholder-value">
                        {result ? (result.model || "Vision Transformer") : "--"}
                    </span>
                </div>
                {result && result.explainability_images && result.explainability_images[0] && (
                    <div className="heatmap-preview-container" style={{ marginTop: '12px', textAlign: 'center' }}>
                        <span className="placeholder-label" style={{ display: 'block', marginBottom: '6px', fontSize: '11px', color: 'var(--text-muted)' }}>AI Explainability Heatmap</span>
                        <img src={result.explainability_images[0]} alt="Heatmap" style={{ width: '100%', maxHeight: '140px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.1)' }} />
                    </div>
                )}
            </div>
        </div>
    );
};

export default ImageDetectionCard;
