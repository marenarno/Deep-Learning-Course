from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
from utils.image_utils import get_center_of_bbox, draw_ellipse, draw_triangle

'''
This module defines the `Tracker` class for detecting and tracking football players and the ball
in a sequence of video frames using two YOLO models (one for players, one for the ball) and ByteTrack.

Key features:
- Batch detection of frames using YOLO
- Player tracking using ByteTrack
- Ball detection with fallback logic and scoring
- Position interpolation for smoother ball tracking
- Drawing of annotated frames
- Caching of tracking results to avoid redundant computation
'''

class Tracker:
    def __init__(self, model_path, ball_model_path, conf_thresh=0.3, iou_thresh=0.5):
        self.model = YOLO(model_path)
        self.model_ball = YOLO(ball_model_path)
        self.tracker = sv.ByteTrack()
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

    def detect_frames(self, frames, batch_size=20):
        detections = []
        for i in range(0, len(frames), batch_size):
            batch = self.model.predict(
                source=frames[i:i+batch_size],
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                imgsz=1024,
                classes=[1]
            )
            detections.extend(batch)
        return detections

    def get_object_tracks(self, frames, use_cache=False, cache_path=None, max_ball_distance=150):
        if use_cache and cache_path and os.path.exists(cache_path):
            return self._load_cached_tracks(cache_path)

        detections = self.detect_frames(frames)
        tracks = {"players": [], "ball": []}
        last_ball_position = None

        for index, detection in enumerate(detections):
            player_class_id, ball_class_id = self._get_class_ids(detection.names)

            supervision_detections = sv.Detections.from_ultralytics(detection)
            player_tracks = self.tracker.update_with_detections(supervision_detections)

            tracks["players"].append(self._extract_tracks(player_tracks, player_class_id))
            tracks["ball"].append({})

            best_ball_bbox = self._get_best_ball_bbox(supervision_detections, player_tracks, ball_class_id, last_ball_position, max_ball_distance)

            if best_ball_bbox is None:
                best_ball_bbox = self._fallback_ball_detection(frames[index], ball_class_id, last_ball_position, max_ball_distance)

            if best_ball_bbox:
                tracks["ball"][index][1] = {"bbox": best_ball_bbox}
                last_ball_position = get_center_of_bbox(best_ball_bbox)

        if cache_path:
            self._save_cached_tracks(tracks, cache_path)

        return tracks

    def interpolate_ball_positions(self, ball_tracks):
        boxes = [frame.get(1, {}).get('bbox', []) for frame in ball_tracks]
        dataframe = pd.DataFrame(boxes, columns=['x1', 'y1', 'x2', 'y2'])
        dataframe = dataframe.interpolate().bfill()
        return [{1: {"bbox": row.tolist()}} for _, row in dataframe.iterrows()]

    def draw_annotations(self, frames, tracks):
        annotated_frames = []
        for index, frame in enumerate(frames):
            image = frame.copy()
            for track_id, player in tracks["players"][index].items():
                image = draw_ellipse(image, player["bbox"], (0, 0, 255), track_id)
            for track_id, ball in tracks["ball"][index].items():
                image = draw_triangle(image, ball["bbox"], (0, 255, 0))
            annotated_frames.append(image)
        return annotated_frames

    def _get_class_ids(self, class_name_mapping):
        inverse_mapping = {v: k for k, v in class_name_mapping.items()}
        return inverse_mapping["player"], inverse_mapping["ball"]

    def _extract_tracks(self, tracked_objects, target_class_id):
        return {
            track_id: {"bbox": bbox.tolist()}
            for bbox, class_id, track_id in zip(tracked_objects.xyxy, tracked_objects.class_id, tracked_objects.tracker_id)
            if class_id == target_class_id
        }

    def _get_best_ball_bbox(self, supervision_detections, player_tracks, ball_class_id, last_position, max_distance):
        all_detections = list(zip(supervision_detections.xyxy, supervision_detections.class_id, supervision_detections.confidence)) + \
                         list(zip(player_tracks.xyxy, player_tracks.class_id, player_tracks.confidence))

        return self._select_best_ball_bbox(all_detections, ball_class_id, last_position, max_distance)

    def _fallback_ball_detection(self, frame, ball_class_id, last_position, max_distance):
        fallback = self.model_ball.predict(
            source=frame,
            conf=0.46,
            iou=0.3,
            imgsz=1920,
            classes=[ball_class_id],
            augment=True,
            agnostic_nms=True
        )[0]
        fallback_detections = sv.Detections.from_ultralytics(fallback)
        return self._select_best_ball_bbox(zip(fallback_detections.xyxy, fallback_detections.class_id, fallback_detections.confidence), ball_class_id, last_position, max_distance)

    def _select_best_ball_bbox(self, detections, ball_class_id, last_position, max_distance):
        best_score = -1
        best_bbox = None
        for bbox, class_id, confidence in detections:
            if class_id == ball_class_id:
                bbox_list = bbox.tolist()
                score = self._calculate_ball_score(confidence, bbox_list, last_position, max_distance)
                if score > best_score:
                    best_score = score
                    best_bbox = bbox_list
        return best_bbox

    def _calculate_ball_score(self, confidence, bbox, last_position, max_distance):
        if last_position is None:
            return confidence
        current_center = get_center_of_bbox(bbox)
        distance = np.linalg.norm(np.array(current_center) - np.array(last_position))
        distance_factor = max(0, 1 - distance / max_distance)
        return confidence * 0.6 + distance_factor * 0.4

    def _load_cached_tracks(self, path):
        with open(path, 'rb') as file:
            return pickle.load(file)

    def _save_cached_tracks(self, tracks, path):
        with open(path, 'wb') as file:
            pickle.dump(tracks, file)