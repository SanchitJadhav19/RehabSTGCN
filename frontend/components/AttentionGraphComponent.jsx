'use client';

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Filler,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Filler,
  Legend
);

/**
 * FrameConfidenceComponent
 * 
 * Displays the per-frame skeleton detection confidence using Chart.js.
 * Shows how reliably the pose was detected throughout the exercise video.
 * 
 * Props:
 *   confidences: list of floats (per-frame confidence 0-1)
 *   score: number
 */
export default function FrameConfidenceComponent({ confidences, score }) {
  if (!confidences || confidences.length === 0) return null;

  // Format data
  const labels = confidences.map((_, index) => index);
  const dataPoints = confidences.map(conf => conf * 100);

  // Find the peak confidence point
  const maxConfidence = Math.max(...dataPoints);
  const peakFrame = dataPoints.findIndex(d => d === maxConfidence);

  // Average confidence
  const avgConfidence = (dataPoints.reduce((sum, d) => sum + d, 0) / dataPoints.length).toFixed(1);

  // Define color theme based on score
  const getThemeColor = (s) => {
    if (s >= 70) return '#10b981'; // green
    if (s >= 40) return '#f59e0b'; // yellow
    return '#ef4444';             // red
  };
  const themeColor = score ? getThemeColor(score) : '#3b82f6';
  const themeColorOp = themeColor + '33'; // Add alpha

  const chartData = {
    labels,
    datasets: [
      {
        label: 'Confidence (%)',
        data: dataPoints,
        borderColor: themeColor,
        backgroundColor: themeColorOp,
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        pointHoverRadius: 6,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        backgroundColor: '#1e293b',
        titleColor: '#f8fafc',
        bodyColor: themeColor,
        borderColor: '#334155',
        borderWidth: 1,
        callbacks: {
          title: (items) => `Frame: ${items[0].label}`,
          label: (item) => `Confidence: ${item.raw.toFixed(1)}%`
        }
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Time (Frames)',
          color: '#94a3b8'
        },
        grid: {
          display: false,
          color: '#334155'
        },
        ticks: { color: '#94a3b8' }
      },
      y: {
        title: {
          display: true,
          text: 'Confidence (%)',
          color: '#94a3b8'
        },
        grid: {
          color: '#334155'
        },
        ticks: { color: '#94a3b8' },
        min: 0,
        max: 100
      }
    }
  };

  return (
    <div className="attention-container">
      <div className="attention-header">
        <h3>Skeleton Detection Confidence</h3>
        <p className="attention-subtitle">
          Average detection confidence: <strong>{avgConfidence}%</strong> across {dataPoints.length} frames.
          <br/>
          Best detection at <strong>frame {peakFrame}</strong>.
        </p>
      </div>

      <div className="chart-wrapper" style={{ width: '100%', height: 300, marginTop: '20px' }}>
        <Line options={options} data={chartData} />
      </div>
    </div>
  );
}
