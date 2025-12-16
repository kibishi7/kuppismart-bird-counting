import cv2
import numpy as np
from ultralytics import YOLO
import os

model = YOLO("yolov8n.pt")

def analyze_video(
    input_path: str,
    fps_sample: int = 1,
    conf_thresh: float = 0.4,
    iou_thresh: float = 0.5,
):
    """Analyze video and return counts, tracks, and weight estimates."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs("outputs", exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    out_video_path = os.path.join("outputs", f"{base_name}_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_video_path, fourcc, original_fps / fps_sample, (width, height))

    timestamps, counts = [], []
    track_samples = {}
    per_frame_weight = []
    track_areas = {}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % fps_sample != 0:
            frame_idx += 1
            continue

        timestamp = frame_idx / original_fps
        results = model.track(
            source=frame,
            conf=conf_thresh,
            iou=iou_thresh,
            persist=True,
            verbose=False,
        )

        r = results[0]
        boxes = r.boxes
        frame_count = 0
        aggregate_area = 0.0

        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            ids = boxes.id
            if ids is not None:
                ids = ids.cpu().numpy()
            else:
                ids = np.arange(len(xyxy))

            for bbox, c, tid in zip(xyxy, confs, ids):
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1
                area = float(w * h)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ID {int(tid)}",
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

                frame_count += 1
                aggregate_area += area

                track_samples.setdefault(int(tid), [])
                if len(track_samples[int(tid)]) < 5:
                    track_samples[int(tid)].append({
                        "timestamp": float(timestamp),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "conf": float(c),
                    })

                track_areas.setdefault(int(tid), [])
                track_areas[int(tid)].append(area)

        cv2.putText(
            frame,
            f"Count: {frame_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )

        out.write(frame)
        timestamps.append(float(timestamp))
        counts.append(frame_count)
        per_frame_weight.append({
            "timestamp": float(timestamp),
            "aggregate_weight_index": float(aggregate_area)
        })

        frame_idx += 1

    cap.release()
    out.release()

    per_track_weight = [
        {
            "track_id": tid,
            "weight_index": float(np.mean(areas)),
            "unit": "index",
            "confidence": 0.5,
        }
        for tid, areas in track_areas.items()
    ]

    tracks_sample_list = [
        {"track_id": tid, "samples": samples}
        for tid, samples in track_samples.items()
    ]

    result = {
        "counts": {"timestamps": timestamps, "counts": counts},
        "tracks_sample": tracks_sample_list,
        "weight_estimates": {
            "per_frame": per_frame_weight,
            "per_track": per_track_weight,
            "unit": "index",
            "note": "Index based on bounding box area. To convert to grams, calibration needed.",
        },
        "artifacts": {"annotated_video_path": out_video_path},
    }
    return result
