# RehabSTGCN

AI-powered clinical exercise quality scoring with temporal interpretability. 

This project uses a Next.js front-end and a FastAPI backend connected to an STGCN-LSTM machine learning pipeline. It analyzes video inputs using MediaPipe and evaluates the geometric graph structure of the skeleton using our custom AI model.

## Prerequisites
- Python 3.9 - 3.10 (Recommended due to TensorFlow/MediaPipe requirements)
- Node.js & npm

## Setting Up and Running Locally

Follow these steps to run the application on your computer:

### 1. Setup the Backend (Python)
The backend converts the video into a 25-joint skeleton array and leverages the STGCN-LSTM model to compute the rehabilitation score.

If using Windows (PowerShell):
```powershell
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r backend/requirements.txt
```

If using Mac/Linux (Bash):
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

**Run the Backend Server:**
```powershell
# Set Python path to ensure imports work correctly, and start the FastAPI server
$env:PYTHONPATH = "."
python backend/main.py
```
> The API will be active at `http://localhost:8000`

---

### 2. Setup the Frontend (Next.js)
The frontend provides the user interface for uploading videos and viewing the visual graphs and scores.

Open a **new terminal tab** without closing the backend:

```powershell
# Navigate into the frontend directory
cd frontend

# Install Node dependencies
npm install

# Start the Webpack Dev Server
npm run dev --webpack
```
> The web application will be accessible at `http://localhost:3000`

## Supported Formats
Videos must be in MP4, AVI, or MOV format. The MediaPipe algorithm will automatically extract coordinate data, frame count, and processing confidence.


$env:PYTHONPATH = "."; ..\STGCN-rehab-main\venv\Scripts\python.exe backend\main.py


cd frontend
npx next dev --webpack
