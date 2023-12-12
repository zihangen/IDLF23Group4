import cv2
import numpy as np
import os
# stack videos to compare frame by frame visually.
def process_videos(video_paths):
    # Open the video files
    caps = [cv2.VideoCapture(path) for path in video_paths]
    num = 0
    while True:
        frames = []
        rets = []
        num += 1
        s = np.round(num / 24,2)
        # Read a frame from each video
        for cap in caps:
            ret, frame = cap.read()
            frames.append(frame)
            rets.append(ret)
        # Check if any video has ended
        if not all(rets):
            print("Not all videos have the same length.")
            break

        # Compare frames
        # Example: comparing first video with each other video
        stacked = np.hstack(frames)
        
        cv2.imshow(f"second {s}",stacked)
        k = cv2.waitKey(0)
        # Press 'q' to quit
        if k & 0xFF == ord('q'):
            break

    # Release all captures
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

# Example usage
dir = "7850-111outvods"
video_paths = sorted([f for f in os.listdir(dir) if f.endswith(".mp4")]) # List of video paths
print(video_paths)

process_videos([os.path.join(dir, p) for p in video_paths])