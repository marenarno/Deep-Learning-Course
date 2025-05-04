import os
import shutil
import cv2

def save_image(image, output_dir, filename):
    """Saves an image to the specified directory with the given filename."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    cv2.imwrite(path, image)


def copy_files(src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir, filenames, prefix):
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    for filename in filenames:
        img_file = os.path.join(src_img_dir, filename.replace(".txt", ".jpg"))
        lbl_file = os.path.join(src_lbl_dir, filename)
        if os.path.exists(img_file) and os.path.exists(lbl_file):
            img_dst = os.path.join(dst_img_dir, f"{prefix}_{filename.replace('.txt', '.jpg')}")
            lbl_dst = os.path.join(dst_lbl_dir, f"{prefix}_{filename}")
            shutil.copy2(img_file, img_dst)
            shutil.copy2(lbl_file, lbl_dst)
def load_frames(input_dir):
    image_files = sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))
    )

    frames = []
    for file in image_files:
        img = cv2.imread(file)
        if img is None:
            print(f"Warning: Could not read {file}")
            continue
        frames.append(img)
    return frames, image_files