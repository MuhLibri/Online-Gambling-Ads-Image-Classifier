import cv2
import numpy as np
from paddleocr import PaddleOCR
import easyocr


EASYOCR_READER = easyocr.Reader(['id'])
PADDLEOCR_READER = PaddleOCR(use_angle_cls=True, lang='id')


def extract_text(image_input, ocr_engine='paddleocr'):
    """
    Args:
        image_input (str or numpy.ndarray): Path to the image file or image in numpy array format (BGR).
        ocr_engine (str): Choice of OCR engine, 'paddleocr' or 'easyocr'. Defaults to 'paddleocr'.

    Returns:
        str: A combination of all detected text, separated by spaces.
             Returns an empty string if no text is detected.
    """
    if ocr_engine not in ['paddleocr', 'easyocr']:
        raise ValueError("ocr_engine must be 'paddleocr' or 'easyocr'")

    if isinstance(image_input, np.ndarray):
        img = image_input
    else:
        raise TypeError("image_input must be a numpy.ndarray")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    detected_texts = []
    if ocr_engine == 'paddleocr':
        result = PADDLEOCR_READER.predict(img_rgb)
        if result and result[0] is not None:
            detected_texts = result[0]['rec_texts']

    elif ocr_engine == 'easyocr':
        result = EASYOCR_READER.readtext(img_rgb)
        if result:
            for (_, text, _) in result:
                detected_texts.append(text)

    final_text = " ".join(detected_texts)

    if len(final_text.strip()) < 5:
        return "tidak ada teks"

    return final_text
