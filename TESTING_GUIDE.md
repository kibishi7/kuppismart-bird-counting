# Testing Guide for Kuppismart Bird Counting

## Quick Start (All-in-One Test)

Follow these steps in order to test everything:

### Step 1: Clone and Setup
```bash
git clone https://github.com/kibishi7/kuppismart-bird-counting.git
cd kuppismart-bird-counting

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Generate Test Video
```bash
python generate_test_video.py
```
**Output**: Creates `data/sample_test.mp4` (10 seconds, 15 simulated birds)

### Step 3: Test Direct Analysis
```bash
python test_script.py
```
**Expected Output**:
- Generates `outputs/sample_test_annotated.mp4` (annotated video with boxes + count)
- Saves `outputs/sample_response.json` (JSON with counts, tracks, weights)
- Shows console messages about frame processing

**Check**: Open the annotated video and confirm:
- Green boxes around "birds" (circles)
- Red "Count: XX" text in top-left corner
- Track IDs visible on each detection

### Step 4: Test FastAPI Service

**Terminal 1 - Start API**:
```bash
uvicorn main:app --reload
```
Should print: `Uvicorn running on http://127.0.0.1:8000`

**Terminal 2 - Test Endpoint**:
```bash
curl -X POST "http://127.0.0.1:8000/analyze_video" \
  -F "file=@data/sample_test.mp4" \
  -F "fps_sample=2" \
  -F "conf_thresh=0.4" \
  -F "iou_thresh=0.5" \
  -o outputs/api_response.json
```

**Check**: 
- Request should complete in 30-60 seconds
- `outputs/api_response.json` should contain:
  - `counts.timestamps` and `counts.counts` arrays
  - `tracks_sample` with track IDs and bounding boxes
  - `weight_estimates` with index values
  - `artifacts.annotated_video_path`

Or visit `http://127.0.0.1:8000/docs` for interactive API testing UI.

## Manual Testing on Real Video

If you have your own poultry farm video:

1. Place video in `data/` folder
2. Update `test_script.py` line 7:
   ```python
   res = analyze_video("data/your_video.mp4", fps_sample=2)
   ```
3. Run: `python test_script.py`

## Troubleshooting

### Issue: "Module not found: ultralytics"
**Fix**: `pip install ultralytics` (should auto-download YOLOv8n model ~40MB on first run)

### Issue: Video codec error
**Fix**: Already handled! Code tries mp4v → MJPG → DIVX in fallback order

### Issue: Slow processing
**Cause**: Using CPU (default). Solution:
- For faster GPU processing, install: `pip install ultralytics torch torchvision`
- YOLO will auto-detect GPU if available

### Issue: Output video is corrupted/empty
**Cause**: Codec not available on system. **Fix**: Already handled by codec fallback logic

## Performance Expectations

| Hardware | FPS (YOLOv8n) | Notes |
|----------|---------------|-------|
| CPU      | 5-15 FPS      | Default, no GPU needed |
| GPU      | 20-60 FPS     | Requires CUDA + torch |


## JSON Output Schema

Sample response structure:
```json
{
  "counts": {
    "timestamps": [0.0, 0.2, 0.4],
    "counts": [15, 14, 15]
  },
  "tracks_sample": [
    {
      "track_id": 7,
      "samples": [
        {"timestamp": 0.4, "bbox": [100, 150, 180, 230], "conf": 0.91}
      ]
    }
  ],
  "weight_estimates": {
    "per_frame": [{"timestamp": 0.0, "aggregate_weight_index": 45000.0}],
    "per_track": [{"track_id": 7, "weight_index": 2500.0, "unit": "index", "confidence": 0.5}],
    "unit": "index",
    "note": "Index based on bounding box area. To convert to grams, calibration needed."
  },
  "artifacts": {
    "annotated_video_path": "outputs/sample_test_annotated.mp4"
  }
}
```

## Next Steps After Testing

1. Take poultry farm video 
2. Replace `data/sample_test.mp4` with real video
3. Run analysis on real data
4. Verify accuracy by visual inspection of annotated output video
5. Calibrate weight index with labeled data (optional)
