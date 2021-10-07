import argparse
import cv2
import torch

from detectors.dlib_detector import DlibDetector
from commons.utils import load_image, write_result, get_landmark, draw_bbox


parser = argparse.ArgumentParser()

parser.add_argument('--input', required=True, help='input file path')
parser.add_argument('--output', required=False, default='/')


def object_detection(input, output, save=True, version='s'):
    yolo_version = f'yolov5{version}'

    # Model
    od_model = torch.hub.load('ultralytics/yolov5', yolo_version)

    # Images
    imgs = []
    for f in input:
        img = load_image(f, rgb=True)
        imgs.append(img)

    # Inference
    results = od_model(imgs, size=640)  # includes NMS

    # Results
    if save:
        results.save(output)  # or .show()

    # results.xyxy[0]  # img1 predictions (tensor)
    # results.pandas().xyxy[0]  # img1 predictions (pandas)
    return results


def face_detection(model, input, type, output):
    if type == 'image':
        for f in input:
            print(f)
            # read image
            img = load_image(f)

            # detect faces from the image
            faces, landmarks = DlibDetector().detect_face(img, model)

            # get specific landmark point (face, eyes, eyebrows, lips, nose)
            # face_line = get_landmark(landmarks[0], 'face')

            # write result image
            write_result(img, faces, output)

    elif type == 'video':
        for f in input:
            cap = cv2.VideoCapture(f)

            width = int(cap.get(3))
            height = int(cap.get(4))
            fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
            out = cv2.VideoWriter(output, fcc, 30, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()

                if ret:
                    faces, landmarks = DlibDetector().detect_face(frame, model)

                    frame_with_bbox = draw_bbox(frame, faces)

                    # write video
                    out.write(frame_with_bbox)

            cap.release()

    elif type == 'webcam':
        cap = cv2.VideoCapture(0)

        width = int(cap.get(3))
        height = int(cap.get(4))
        fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
        out = cv2.VideoWriter(output, fcc, 30, (width, height))

        while True:
            ret, frame = cap.read()

            if ret:
                faces, landmarks = DlibDetector().detect_face(frame, model)

                frame_with_bbox = draw_bbox(frame, faces)

                # write video
                out.write(frame_with_bbox)

                if cv2.waitKey(1) == ord('q'):
                    break

        cap.release()


def build_model(model_name):
    models = {'Dlib': DlibDetector().load_model}

    model = models.get(model_name)

    if model:
        model = model()
    else:
        raise ValueError(f"Invalid model_name '{model_name}', please check.")

    return model


if __name__ == '__main__':
    """
    Load Model
    """
    model = build_model('Dlib')

    """
    Select input type (Image / Video / Web cam) 
    """
    input_type = 'image'
    input = ['./sample_images/img.jpg']
    output = './results'

    print('face detection')
    face_detection(model, input, input_type, output)
    
    # print('opject detection')
    # object_detection(input, output, save=True, version='s')



