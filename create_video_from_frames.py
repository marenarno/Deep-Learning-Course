import cv2
import os
from natsort import natsorted

'''
This script converts a folder of image frames into a video file.
Frames must be named in a way that allows for proper sorting (e.g., using frame numbers).
It reads all .png or .jpg files from the input directory, orders them naturally,
and writes them into a video using OpenCV.

Output:
- A single .mp4 video file containing all the frames in sequence.

Usage:
Run the script (change dir in code)
'''

def create_video_from_frames(
    frames_dir="/home/marenfa/dl2/frames_output",
    output_path="/home/marenfa/dl2/tracked_video.mp4",
    fps=30
):
    # Get and sort all image files
    frame_files = [f for f in os.listdir(frames_dir) if f.lower().endswith((".png", ".jpg"))]
    frame_files = natsorted(frame_files)

    if not frame_files:
        print("No images found in:", frames_dir)
        return

    # Read first frame to get dimensions
    first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    height, width, _ = first_frame.shape

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write all frames to video
    for file in frame_files:
        frame_path = os.path.join(frames_dir, file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            out.write(frame)
        else:
            print("Warning: could not read", frame_path)

    out.release()
    print("Video saved to:", output_path)


if __name__ == "__main__":
    create_video_from_frames()
