import cv2
import numpy as np
import os

def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) / 2, (y1 + y2) / 2]

def get_bbox_width(bbox):
    x1, _, x2, _ = bbox
    return x2 - x1

def draw_ellipse(frame, bbox, color, track_id=None):
    y2 = int(bbox[3])
    x_c, _ = get_center_of_bbox(bbox)
    w = get_bbox_width(bbox)
    cv2.ellipse(frame, (int(x_c), y2), (int(w), int(w * 0.35)),
                0, -45, 235, color, 2, cv2.LINE_4)
    if track_id is not None:
        rw, rh = 40, 20
        x1 = int(x_c - rw / 2)
        y1 = int(y2 - rh / 2) + 15
        x2, y2r = x1 + rw, y1 + rh
        cv2.rectangle(frame, (x1, y1), (x2, y2r), color, cv2.FILLED)
        tx = x1 + 12 - (10 if track_id > 99 else 0)
        cv2.putText(frame, f"{track_id}", (tx, y1 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return frame

def draw_triangle(frame, bbox, color):
    y = int(bbox[1])
    x_c = int(get_center_of_bbox(bbox)[0])
    pts = np.array([[x_c, y], [x_c - 10, y - 20], [x_c + 10, y - 20]], dtype=np.int32)
    cv2.drawContours(frame, [pts], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [pts], 0, (0, 0, 0), 2)
    return frame

def draw_boxes_on_frame(image_path, label_path, class_names, output_path, img_width=1920, img_height=1080):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return

    with open(label_path, 'r') as f:
        for line in f:
            cls, x, y, w, h = map(float, line.strip().split())
            # Konverter YOLO format til pixel-koordinater
            x_center, y_center = x * img_width, y * img_height
            box_width, box_height = w * img_width, h * img_height
            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)
            class_name = class_names[int(cls)]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(output_path, image)
    print(f"Labeled image saved to: {output_path}")