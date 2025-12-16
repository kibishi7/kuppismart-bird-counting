# Kuppismart Bird Counting & Weight Estimation

**ML/AI Engineering Assessment - Poultry Farm CCTV Analysis**

## Overview

This project processes fixed-camera poultry farm CCTV videos to detect, track, and analyze birds in real-time. The pipeline:

1. **Detects** individual birds using YOLOv8 pretrained model
2. **Tracks** birds across frames using ByteTrack algorithm to maintain stable IDs
3. **Counts** birds over time and generates frame-level statistics
4. **Estimates** bird weight as a pixel-area-based index (scalable to grams with calibration)
5. **Exposes** results via FastAPI service with video file upload support

## Architecture

### Detection & Tracking
- **Model**: YOLOv8n (nano, CPU-friendly)
- **Tracker**: ByteTrack (default YOLO tracker)
- **Framework**: Ultralytics YOLO

### Weight Estimation
- **Proxy**: Bounding box pixel area (width × height)
- **Aggregation**:
  - Per-frame: Sum of all bird areas
  - Per-track: Average area across all frames where bird appears
- **Unit**: Currently "index" (pixel area)
- **Calibration**: To convert to grams, need:
  - Camera height + reference object length → pixels-to-cm mapping
  - Labeled dataset (pixel area, true weight) → regression model

### API Service
- **Framework**: FastAPI (async, file upload support)
- **Endpoints**:
  - `GET /health` → returns `{"status": "ok"}`
  - `POST /analyze_video` → uploads video, returns JSON with counts, tracks, weights, artifacts

## Setup

### 1. Clone Repository
```bash
git clone https://github.com/kibishi7/kuppismart-bird-counting.git
cd kuppismart-bird-counting
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\\Scripts\\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
Obtain the poultry farm video from: [Google Forms Link](https://forms.gle/3aiJKdsWaFiDK2Hq5)

Save sample video as:
```
data/sample_farm.mp4
```

## Usage

### Option 1: Direct Analysis (No API)

```bash
python test_script.py
```

This will:
- Load `data/sample_farm.mp4`
- Run detection + tracking with fps_sample=2
- Save annotated video to `outputs/sample_farm_annotated.mp4`
- Save JSON response to `outputs/sample_response.json`

### Option 2: FastAPI Service

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then:
- Visit `http://127.0.0.1:8000/docs` for interactive API docs
- Or use curl:

```bash
curl -X POST "http://127.0.0.1:8000/analyze_video" \\
  -F "file=@data/sample_farm.mp4" \\
  -F "fps_sample=2" \\
  -F "conf_thresh=0.4" \\
  -F "iou_thresh=0.5" \\
  -o output_response.json
```

## Output Format

The analysis returns a JSON object:

```json
{
  "counts": {
    "timestamps": [0.0, 0.2, 0.4, ...],
    "counts": [35, 36, 35, ...]
  },
  "tracks_sample": [
    {
      "track_id": 7,
      "samples": [
        {
          "timestamp": 0.4,
          "bbox": [100, 150, 180, 230],
          "conf": 0.91
        }
      ]
    }
  ],
  "weight_estimates": {
    "per_frame": [
      {"timestamp": 0.0, "aggregate_weight_index": 45000.0}
    ],
    "per_track": [
      {"track_id": 7, "weight_index": 2500.0, "unit": "index", "confidence": 0.5}
    ],
    "unit": "index",
    "note": "Index based on bounding box area in pixels. To convert to grams, need calibration and labeled data."
  },
  "artifacts": {
    "annotated_video_path": "outputs/sample_farm_annotated.mp4"
  }
}
```

## Implementation Details

### Bird Counting
- Frame-by-frame detection + tracking
- Bird count = number of unique track IDs visible in each frame
- Annotated video overlay shows:
  - Bounding boxes (green)
  - Track IDs (top-left of each box)
  - Total count (red text, top-left corner)

### FPS Sampling
- Process every Nth frame to reduce computation (default N=1)
- Timestamps still reflect actual video time, not sampled frames

### Occlusions & ID Switches
- ByteTrack handles partial occlusions automatically
- ID switches rare with stable camera and non-rapid movement
- Track ends if bird disappears and is not re-detected

### Weight Estimation Calibration

To convert pixel area to grams:

1. **Camera Calibration**:
   - Measure camera height from ground
   - Place reference object (e.g., 30cm ruler) in scene
   - Map pixel distance to cm

2. **Collect Training Data**:
   - Manual annotations: pixel area + true bird weight (scale)
   - At least 50-100 samples per bird size category

3. **Train Regression**:
   ```python
   from sklearn.linear_model import LinearRegression
   model = LinearRegression()
   model.fit(X_pixel_area, y_true_weight_grams)
   ```

4. **Update `analysis.py`**:
   - Replace `weight_index` with regression prediction
   - Change unit to "grams"

## Project Structure

```
kuppismart-bird-counting/
├── main.py                    # FastAPI application
├── analysis.py                # Video analysis core logic
├── test_script.py             # Direct script for testing
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── data/
│   └── sample_farm.mp4        # Input video (download from Google Forms)
├── outputs/
│   ├── sample_farm_annotated.mp4  # Annotated output video
│   └── sample_response.json       # Sample JSON response
└── uploads/                   # Temporary upload folder (API)
```

## Dependencies

- **ultralytics** (YOLOv8 detection + tracking)
- **opencv-python** (video I/O)
- **fastapi** (HTTP API framework)
- **uvicorn[standard]** (ASGI server)
- **python-multipart** (file upload parsing)
- **numpy** (array operations)

All specified in `requirements.txt`.

## Performance Notes

- **CPU Mode**: YOLOv8n runs on CPU, ~5-15 FPS per frame
- **GPU Mode**: Use YOLOv8s or larger, enable CUDA for 20-60 FPS
- **Optimization**: FPS sampling (e.g., process every 2nd or 3rd frame) reduces latency
- **Model Download**: First run downloads ~40MB YOLOv8n model (~5min on first use)

## Submission

Deliverables included in this repo:
1. ✅ Full source code (`main.py`, `analysis.py`, `test_script.py`)
2. ✅ Comprehensive README.md
3. ✅ Annotated output video sample
4. ✅ Sample JSON response
5. ✅ `requirements.txt` for environment setup

## Author

**Assessment Submission** - Kuppismart ML/AI Engineering Internship  
**Submitted**: December 2025

---

*For questions or issues, refer to the implementation details above or contact the assessment team.*
