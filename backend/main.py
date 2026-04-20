"""
==============================================================================
  FastAPI Backend for RehabSTGCN
==============================================================================

Endpoints:
  POST /api/predict  — Upload video → get rehabilitation quality score
  GET  /api/health   — Health check

Runs on: http://localhost:8000
==============================================================================
"""

import os
import sys
import tempfile
import traceback

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_model.predictor import RehabPredictor

# ============================================================================
# Initialize app and model
# ============================================================================
app = FastAPI(
    title="RehabSTGCN API",
    description="AI-powered rehabilitation exercise quality assessment using STGCN-LSTM",
    version="2.0.0"
)

# CORS — allow Next.js frontend (port 3000) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
PRETRAINED_DIR = os.path.join(PROJECT_ROOT, "ml_model", "pretrained")
predictor = None


@app.on_event("startup")
async def load_model():
    global predictor
    try:
        predictor = RehabPredictor(PRETRAINED_DIR)
        print("[Backend] STGCN-LSTM model loaded successfully!")
    except FileNotFoundError as e:
        print(f"[Backend] WARNING: {e}")
        print("[Backend] Ensure pretrained files exist in ml_model/pretrained/")
    except Exception as e:
        print(f"[Backend] Model load error: {e}")
        traceback.print_exc()


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/api/health")
async def health_check():
    """Check if the server and model are ready."""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "model_type": "STGCN-LSTM (TensorFlow)",
        "joints": 25,
        "framework": "Kinect v2 skeleton via MediaPipe"
    }


@app.post("/api/predict")
async def predict_exercise(video: UploadFile = File(...)):
    """
    Upload a video and get a rehabilitation exercise quality score.

    Args:
        video: Video file (MP4, AVI, MOV, etc.)

    Returns:
        JSON: {
            "score": float,              # Quality score
            "num_frames": int,           # Frames with detected pose
            "frame_confidences": list,   # Per-frame detection confidence
            "video_info": dict           # Video metadata
        }
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Ensure pretrained files exist in ml_model/pretrained/"
        )

    # Validate file type
    allowed_types = [
        'video/mp4', 'video/avi', 'video/quicktime',
        'video/x-msvideo', 'video/webm', 'application/octet-stream'
    ]
    if video.content_type and video.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {video.content_type}. Please upload MP4, AVI, or MOV."
        )

    # Save uploaded video to temp file
    temp_path = None
    try:
        suffix = os.path.splitext(video.filename)[1] or '.mp4'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await video.read()
            tmp.write(content)
            temp_path = tmp.name

        # Run full pipeline: Video → Skeleton → Score
        result = predictor.predict_from_video(temp_path)

        return JSONResponse(content={
            "score": result["score"],
            "num_frames": result["num_frames"],
            "frame_confidences": result["frame_confidences"],
            "video_info": {
                "fps": result["video_info"]["fps"],
                "total_frames": result["video_info"]["total_frames"],
                "resolution": f"{result['video_info']['width']}x{result['video_info']['height']}",
            }
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


# ============================================================================
# Run server
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    print("Starting RehabSTGCN Backend on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
