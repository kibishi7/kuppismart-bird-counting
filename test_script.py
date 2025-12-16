"""Test script for local video analysis without API."""
import json
from analysis import analyze_video

def main():
    # Ensure data folder exists and has a sample video
    import os
    if not os.path.exists("data"):
        os.makedirs("data")
    
    video_path = "data/sample_farm.mp4"
    
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        print("Please download the poultry farm video from the Google Forms link.")
        print("Save it as data/sample_farm.mp4")
        return
    
    print("Starting video analysis...")
    print(f"Processing: {video_path}")
    print()
    
    # Run analysis with FPS sampling
    result = analyze_video(
        input_path=video_path,
        fps_sample=2,  # Process every 2nd frame for speed
        conf_thresh=0.4,
        iou_thresh=0.5,
    )
    
    print("Analysis complete!")
    print()
    
    # Display results summary
    print("=== RESULTS SUMMARY ===")
    counts_data = result["counts"]
    print(f"Total frames processed: {len(counts_data['counts'])}")
    print(f"Min bird count: {min(counts_data['counts'])}")
    print(f"Max bird count: {max(counts_data['counts'])}")
    print(f"Avg bird count: {sum(counts_data['counts']) / len(counts_data['counts']):.1f}")
    print()
    
    weight_data = result["weight_estimates"]
    print(f"Tracked unique birds: {len(weight_data['per_track'])}")
    if weight_data['per_track']:
        indices = [w['weight_index'] for w in weight_data['per_track']]
        print(f"Weight index range: {min(indices):.0f} - {max(indices):.0f}")
    print()
    
    artifacts = result["artifacts"]
    print(f"Output video: {artifacts['annotated_video_path']}")
    print()
    
    # Save JSON response
    os.makedirs("outputs", exist_ok=True)
    json_path = "outputs/sample_response.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"JSON response saved: {json_path}")
    
if __name__ == "__main__":
    main()
