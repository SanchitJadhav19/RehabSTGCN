'use client';

/**
 * ScoreDisplayComponent
 * 
 * Displays the predicted rehabilitation quality score as an animated
 * circular gauge with color coding. Score range: 0-100.
 * 
 * Props:
 *   score: number (0-100)
 *   loading: boolean
 */
export default function ScoreDisplayComponent({ score, loading }) {
  if (loading) {
    return (
      <div className="score-container">
        <div className="score-loading">
          <div className="spinner"></div>
          <p>Analyzing exercise...</p>
          <p className="loading-subtitle">Extracting skeleton &amp; running STGCN-LSTM model</p>
        </div>
      </div>
    );
  }

  if (score === null || score === undefined) return null;

  // Color based on score (0-100 range)
  const getColor = (s) => {
    if (s >= 70) return '#10b981';  // green — good
    if (s >= 40) return '#f59e0b';  // yellow — moderate
    return '#ef4444';              // red — needs improvement
  };

  const getLabel = (s) => {
    if (s >= 80) return 'Excellent';
    if (s >= 60) return 'Good';
    if (s >= 40) return 'Fair';
    if (s >= 20) return 'Needs Improvement';
    return 'Poor';
  };

  const color = getColor(score);
  const percentage = Math.min(Math.max(score / 100, 0), 1) * 100;

  // SVG circular gauge
  const radius = 70;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  return (
    <div className="score-container">
      <h2 className="score-title">Rehabilitation Quality Score</h2>

      <div className="score-gauge">
        <svg width="180" height="180" viewBox="0 0 180 180">
          {/* Background circle */}
          <circle
            cx="90" cy="90" r={radius}
            fill="none" stroke="#1e293b" strokeWidth="12"
          />
          {/* Score arc */}
          <circle
            cx="90" cy="90" r={radius}
            fill="none" stroke={color} strokeWidth="12"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            transform="rotate(-90 90 90)"
            style={{ transition: 'stroke-dashoffset 1.5s ease-out' }}
          />
        </svg>

        <div className="score-value" style={{ color }}>
          <span className="score-number">{score.toFixed(1)}</span>
          <span className="score-max">/100</span>
        </div>
      </div>

      <p className="score-label" style={{ color }}>{getLabel(score)}</p>
    </div>
  );
}
