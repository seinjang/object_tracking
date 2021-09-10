# DLIB
import dlib # dlib-19.22.0
import gdown # gdown-3.13.0
import bz2
import imutils # imutils-0.5.4
import sys

from os import path
from os import makedirs


class DlibDetector:
    def __init__(self):
        #self.img_size = 224
        self.weight_dir = './detectors/weights'
        self.dlib_weight = 'shape_predictor_68_face_landmarks.dat'

        # create dir
        if not path.isdir(self.weight_dir):
            makedirs(self.weight_dir, exist_ok=True)

    def load_model(self):
        bz2_file = path.join(self.weight_dir, self.dlib_weight + '.bz2')
        output_file = path.join(self.weight_dir, self.dlib_weight)

        # check if dlib weight file exist, download if not
        if not path.isfile(path.join(self.weight_dir, self.dlib_weight)):
            print(f"download weight file '{self.dlib_weight}' under 'detectors/weights'")

            # download weight file
            download_url = f"http://dlib.net/files/{self.dlib_weight}.bz2"
            gdown.download(download_url, bz2_file, quiet=False)

            # write wegith file on local
            zipfile = bz2.BZ2File(bz2_file)
            data = zipfile.read()
            open(output_file, 'wb').write(data)

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(output_file)

        dlib_model = {'face_detector': detector,
                      'shape_predictor': predictor}

        return dlib_model

    def detect_face(self, image, dlib_model):
        faces = []
        landmarks = []

        # load models
        detector = dlib_model['face_detector']
        predictor = dlib_model['shape_predictor']

        # image resize
        #image = imutils.resize(image, width=self.img_size)

        # detect face
        dets = detector(image, 1)

        num_faces = len(dets)
        if num_faces == 0:
            print("Sorry, there were no faces found in image")
            sys.exit()
        else:
            for idx, d in enumerate(dets):
                left = d.left()
                right = d.right()
                top = d.top()
                bottom = d.bottom()

                cropped_face = image[top:bottom, left:right]
                bbox = [left, top, right - left, bottom - top]
                shape = predictor(image, dets[idx])

                detected_face = dlib.get_face_chip(image, shape, size=cropped_face.shape[0])

                faces.append((detected_face, bbox))
                landmarks.append([(shape.part(i).x, shape.part(i).y) for i in range(68)])

        return faces, landmarks
