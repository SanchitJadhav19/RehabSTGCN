'use client';

import { useState, useRef } from 'react';

/**
 * VideoUploadComponent (Stitch Module)
 * 
 * Drag-and-drop video upload with file selection fallback.
 * Shows video preview after selection.
 * 
 * Props:
 *   onFileSelected(file) — callback when a video file is chosen
 */
export default function VideoUploadComponent({ onFileSelected }) {
  const [isDragging, setIsDragging] = useState(false);
  const [preview, setPreview] = useState(null);
  const [fileName, setFileName] = useState('');
  const fileInputRef = useRef(null);

  const handleFile = (file) => {
    if (!file) return;
    if (!file.type.startsWith('video/')) {
      alert('Please upload a video file (MP4, AVI, MOV)');
      return;
    }
    setFileName(file.name);
    setPreview(URL.createObjectURL(file));
    onFileSelected(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => setIsDragging(false);

  const handleClick = () => fileInputRef.current?.click();

  const handleInputChange = (e) => handleFile(e.target.files[0]);

  return (
    <div className="upload-container">
      <div
        className={`upload-dropzone ${isDragging ? 'dragging' : ''} ${preview ? 'has-preview' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={handleClick}
      >
        {preview ? (
          <div className="preview-wrapper">
            <video
              src={preview}
              controls
              className="video-preview"
            />
            <p className="file-name">{fileName}</p>
            <p className="change-hint">Click to change video</p>
          </div>
        ) : (
          <div className="upload-prompt">
            <div className="upload-icon">📹</div>
            <h3>Drop your exercise video here</h3>
            <p>or click to browse files</p>
            <p className="supported-formats">MP4, AVI, MOV supported</p>
          </div>
        )}
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="video/*"
        onChange={handleInputChange}
        style={{ display: 'none' }}
      />
    </div>
  );
}
