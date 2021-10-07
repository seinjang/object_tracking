import os
import cv2


def load_image(image_path, rgb=False):
    if os.path.isfile(image_path):
        if rgb:
            image = cv2.imread(image_path)[..., ::-1]  # OpenCV image (BGR to RGB)
        else:
            image = cv2.imread(image_path)
    else:
        raise ValueError(f"Please check {image_path} exists")

    return image


def get_landmark(landmarks, point):
    point_nums = {'face': range(18),
                  'eyes': range(36, 48),
                  'eyebrows': range(17, 27),
                  'nose': range(27, 36),
                  'lips': range(48, 68)}

    landmark = [landmarks[x] for x in point_nums[point]]

    return landmark


def write_result(image, faces, output_dir='./'):
    for i, (face, bbox) in enumerate(faces):
        x, y, w, h = bbox
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 125), 2)

        # write single face on local disk
        cv2.imwrite(os.path.join(output_dir, f'face_{i}.jpg'), face)

    # write full image with bounding box on the faces
    cv2.imwrite(os.path.join(output_dir, 'face_detected_image.jpg'), image)


def draw_bbox(image, faces, color=(0, 0, 125)):
    for face, bbox in faces:
        x, y, w, h = bbox
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)

    return image