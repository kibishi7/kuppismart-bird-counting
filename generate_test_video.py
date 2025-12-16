"""Generate a test video with simulated birds for local testing."""
import cv2
import numpy as np

def generate_test_video(output_path='data/sample_test.mp4', duration=10, fps=30):
    """
    Generate a synthetic test video with moving circles (representing birds).
    
    Args:
        output_path: Path to save the video
        duration: Video duration in seconds
        fps: Frames per second
    """
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    num_frames = duration * fps
    num_birds = 15
    
    # Initialize bird positions and velocities
    birds = []
    for i in range(num_birds):
        bird = {
            'x': np.random.randint(20, width - 20),
            'y': np.random.randint(20, height - 20),
            'vx': np.random.uniform(-3, 3),
            'vy': np.random.uniform(-3, 3),
            'radius': np.random.randint(5, 15)
        }
        birds.append(bird)
    
    print(f"Generating {num_frames} frames with {num_birds} birds...")
    
    for frame_idx in range(num_frames):
        # Create blank frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 200
        
        # Draw each bird
        for bird in birds:
            # Update position
            bird['x'] += bird['vx']
            bird['y'] += bird['vy']
            
            # Bounce off walls
            if bird['x'] - bird['radius'] < 0 or bird['x'] + bird['radius'] > width:
                bird['vx'] *= -1
            if bird['y'] - bird['radius'] < 0 or bird['y'] + bird['radius'] > height:
                bird['vy'] *= -1
            
            # Clamp position
            bird['x'] = np.clip(bird['x'], bird['radius'], width - bird['radius'])
            bird['y'] = np.clip(bird['y'], bird['radius'], height - bird['radius'])
            
            # Draw bird as a circle
            cv2.circle(frame, (int(bird['x']), int(bird['y'])), bird['radius'], (100, 100, 255), -1)
            cv2.circle(frame, (int(bird['x']), int(bird['y'])), bird['radius'], (0, 0, 0), 1)
        
        # Add text
        cv2.putText(frame, f'Frame: {frame_idx+1}/{num_frames}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Birds: {num_birds}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        
        out.write(frame)
        if (frame_idx + 1) % 30 == 0:
            print(f"  Generated {frame_idx + 1}/{num_frames} frames")
    
    out.release()
    print(f"✓ Test video saved to {output_path}")

if __name__ == '__main__':
    import os
    os.makedirs('data', exist_ok=True)
    generate_test_video()
