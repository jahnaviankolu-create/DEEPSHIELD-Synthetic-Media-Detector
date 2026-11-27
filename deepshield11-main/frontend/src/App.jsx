import React from 'react';
import ImageDetectionCard from './components/ImageDetectionCard';
import VideoDetectionCard from './components/VideoDetectionCard';
import AudioDetectionCard from './components/AudioDetectionCard';
import './App.css';

function App() {
    return (
        <div className="app-layout">
            {/* Header Navbar */}
            <header className="app-header">
                <div className="header-brand">
                    <div className="shield-logo">
                        <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>
                            <path d="m9 12 2 2 4-4"/>
                        </svg>
                    </div>
                    <div className="brand-text">
                        <h2>DeepShield</h2>
                        <span className="brand-badge">AI Detection Suite</span>
                    </div>
                </div>

                <div className="header-status">
                    <span className="status-indicator"></span>
                    <span className="status-text">System Active</span>
                </div>
            </header>

            {/* Main Content Area */}
            <main className="dashboard-container">
                <div className="dashboard-title-section">
                    <h1 className="dashboard-heading">Synthetic Media Analysis Dashboard</h1>
                    <p className="dashboard-subheading">Select a detection module below to inspect media files for deepfakes and AI synthesis.</p>
                </div>

                {/* 3 Upload Cards Grid */}
                <div className="cards-grid">
                    <ImageDetectionCard />
                    <VideoDetectionCard />
                    <AudioDetectionCard />
                </div>
            </main>
        </div>
    );
}

export default App;
