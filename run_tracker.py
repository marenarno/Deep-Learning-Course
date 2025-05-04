import os
from tracker import Tracker
from utils.image_utils import save_image
from utils.dataset_utils import load_frames
from config import PROJECT_DIR

'''
Runs object tracking on a sequence of image frames using two YOLO models:
one for general object detection (e.g., players) and one specifically for tracking the ball.

Steps:
1. Loads all frames from a specified input directory.
2. Initializes the Tracker class with two model weights.
3. Computes object tracks, optionally loading from a cached track file.
4. Interpolates ball positions to improve continuity.
5. Draws annotations (e.g., bounding boxes and IDs) on each frame.
6. Saves the annotated frames to an output directory.

Output:
- Annotated frames saved to specified output directory.
'''


INPUT_DIR = os.path.join(PROJECT_DIR, 'data', 'data', '4_annotate_1min_bodo_start', 'img1')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'frames_output')
BEST_MODEL_PATH = os.path.join(PROJECT_DIR, "src", "runs", "soccer_training", "ball_and_player", "weights", "best.pt")
BALL_MODEL_PATH = os.path.join(PROJECT_DIR, "src", "runs", "soccer_training", "only_ball", "weights", "best.pt")
CACHE_PATH = os.path.join(PROJECT_DIR, "output", "tracks.pkl")


def run_tracker_on_frames():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    frames, filenames = load_frames(INPUT_DIR)
    if not frames:
        print("No frames loaded. Exiting.")
        return

    tracker = Tracker(model_path=BEST_MODEL_PATH, ball_model_path=BALL_MODEL_PATH)

    tracks = tracker.get_object_tracks(
        frames,
        use_cache=True,
        cache_path=CACHE_PATH
    )

    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    annotated_frames = tracker.draw_annotations(frames, tracks)

    for index, frame in enumerate(annotated_frames):
        filename = os.path.basename(filenames[index])
        save_image(frame, OUTPUT_DIR, filename)

    print(f"Tracking complete. Annotated frames saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_tracker_on_frames()
