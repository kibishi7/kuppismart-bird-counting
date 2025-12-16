from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import uuid
from analysis import analyze_video

app = FastAPI(
    title="Bird Counting & Weight Estimation API",
    description="ML/AI Assessment: Poultry farm CCTV analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "message": "API is running"}

@app.post("/analyze_video")
async def analyze_video_endpoint(
    file: UploadFile = File(...),
    fps_sample: int = Form(1),
    conf_thresh: float = Form(0.4),
    iou_thresh: float = Form(0.5),
):
    """Analyze uploaded video for bird counting and weight estimation."""
    os.makedirs("uploads", exist_ok=True)
    video_id = str(uuid.uuid4())
    input_path = os.path.join("uploads", f"{video_id}.mp4")

    with open(input_path, "wb") as f:
        data = await file.read()
        f.write(data)

    try:
        result = analyze_video(
            input_path=input_path,
            fps_sample=fps_sample,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
        )
        return result
    except Exception as e:
        return {"error": str(e), "status": "failed"}
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
