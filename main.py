from detectors.dlib_detector import DlibDetector
from commons.utils import load_image, write_result, get_landmark


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
    Load Image
    """
    image = load_image('./img.jpg')

    """
    Detect faces from the image
    """
    faces, landmarks = DlibDetector().detect_face(image, model)

    """
    Get specific landmark point (face, eyes, eyebrows, lips, nose)
    """
    face_line = get_landmark(landmarks[0], 'face')
