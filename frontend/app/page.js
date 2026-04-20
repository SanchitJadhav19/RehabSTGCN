'use client';

import { useState } from 'react';
import VideoUploadComponent from '../components/VideoUploadComponent';
import ScoreDisplayComponent from '../components/ScoreDisplayComponent';
import FrameConfidenceComponent from '../components/AttentionGraphComponent';

export default function Home() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileSelected = (selectedFile) => {
    setFile(selectedFile);
    setResult(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('video', file);

    try {
      const response = await fetch('http://localhost:8000/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Failed to analyze video');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="container">
      <header className="header">
        <h1>RehabSTGCN Assessment System</h1>
        <p className="subtitle">AI-powered clinical exercise quality scoring using STGCN-LSTM</p>
      </header>

      <div className="layout-grid">
        {/* Left Column: Upload & Controls */}
        <div className="left-panel">
          <div className="card">
            <h2>1. Upload Exercise Video</h2>
            <VideoUploadComponent onFileSelected={handleFileSelected} />
            
            <button 
              className={`analyze-btn ${!file || loading ? 'disabled' : ''}`}
              onClick={handleAnalyze}
              disabled={!file || loading}
            >
              {loading ? 'Analyzing... (this may take a minute)' : 'Analyze Movement Quality'}
            </button>
            
            {error && (
              <div className="error-message">
                ⚠️ {error}
              </div>
            )}
          </div>
          
          {/* Architecture info */}
          <div className="card info-card">
            <h3>How it works</h3>
            <ul className="feature-list">
              <li><strong>MediaPipe</strong> extracts 25 Kinect v2 skeletal joints</li>
              <li><strong>ST-GCN</strong> analyzes spatial joint relationships via graph convolutions</li>
              <li><strong>LSTM</strong> captures temporal movement patterns</li>
              <li><strong>Score</strong> predicts rehabilitation exercise quality (0-100)</li>
            </ul>
          </div>
        </div>

        {/* Right Column: Results */}
        <div className="right-panel">
          <div className="card result-card">
            {result ? (
              <div className="results-wrapper fadeIn">
                <ScoreDisplayComponent score={result.score} loading={false} />
                
                <div className="divider"></div>
                
                <FrameConfidenceComponent 
                  confidences={result.frame_confidences} 
                  score={result.score} 
                />
                
                <div className="meta-info">
                  <p>Analyzed <strong>{result.num_frames}</strong> frames from video.</p>
                  {result.video_info && (
                    <p>Resolution: {result.video_info.resolution} | FPS: {result.video_info.fps}</p>
                  )}
                </div>
              </div>
            ) : loading ? (
              <ScoreDisplayComponent loading={true} />
            ) : (
              <div className="empty-state">
                <p>Upload a rehabilitation exercise video and click Analyze to see results here.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
