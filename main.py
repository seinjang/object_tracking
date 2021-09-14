import argparse
import cv2

from detectors.dlib_detector import DlibDetector
from commons.utils import load_image, write_result, get_landmark, draw_bbox


def build_model(model_name):
    models = {'Dlib': DlibDetector().load_model}

    model = models.get(model_name)

    if model:
        model = model()
    else:
        raise ValueError(f"Invalid model_name '{model_name}', please check.")

    return model


def face_detection(model, input, type, output):
    if type == 'image':
        print('image')
        # read image
        img = load_image(input)

        # detect faces from the image
        faces, landmarks = DlibDetector().detect_face(img, model)

        # get specific landmark point (face, eyes, eyebrows, lips, nose)
        # face_line = get_landmark(landmarks[0], 'face')

        # write result image
        write_result(img, faces, output)

    else:
        if type == 'video':
            cap = cv2.VideoCapture(input)
        elif type == 'webcam':
            cap = cv2.VideoCapture(0)

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


if __name__ == '__main__':
    """
    Load Model
    """
    model = build_model('Dlib')

    """
    Select input type (Image / Video / Web cam) 
    """
    input_type = 'image'





