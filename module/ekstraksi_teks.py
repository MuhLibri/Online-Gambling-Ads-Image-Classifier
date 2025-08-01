"""
Modul untuk mengekstrak teks dari gambar menggunakan PaddleOCR atau EasyOCR.
"""

import cv2
import numpy as np
from paddleocr import PaddleOCR
import easyocr

# Inisialisasi pembaca OCR
# EasyOCR diinisialisasi sekali untuk efisiensi karena waktu startup yang lama.
EASYOCR_READER = easyocr.Reader(['id'])  # Ganti atau tambahkan bahasa sesuai kebutuhan, misal ['id', 'en']
# Inisialisasi PaddleOCR di sini untuk memastikan state bersih setiap kali dipanggil.
paddle_ocr_reader = PaddleOCR(use_angle_cls=True, lang='id')


def extract_text(image_input, ocr_engine='paddleocr'):
    """
    Mengekstrak teks dari gambar menggunakan PaddleOCR atau EasyOCR.

    Args:
        image_input (str atau numpy.ndarray): Path ke file gambar atau gambar dalam format numpy array (BGR).
        ocr_engine (str): Pilihan OCR engine, 'paddleocr' atau 'easyocr'. Defaultnya 'paddleocr'.

    Returns:
        str: Gabungan semua teks yang terdeteksi, dipisahkan dengan spasi.
             Mengembalikan string kosong jika tidak ada teks yang terdeteksi.
    """
    if ocr_engine not in ['paddleocr', 'easyocr']:
        raise ValueError("ocr_engine harus berupa 'paddleocr' atau 'easyocr'")

    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"Gambar tidak ditemukan di path: {image_input}")
    elif isinstance(image_input, np.ndarray):
        img = image_input
    else:
        raise TypeError("image_input harus berupa path (str) atau numpy.ndarray")

    # PaddleOCR dan EasyOCR lebih baik bekerja dengan gambar RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    detected_texts = []
    if ocr_engine == 'paddleocr':
        # PaddleOCR menerima numpy array (RGB)
        # Format output untuk PaddleOCR v3.1+
        result = paddle_ocr_reader.predict(img_rgb)
        if result and result[0] is not None:
            # 'rec_texts' berisi daftar semua teks yang terdeteksi
            detected_texts = result[0]['rec_texts']

    elif ocr_engine == 'easyocr':
        # EasyOCR menerima numpy array (RGB)
        result = EASYOCR_READER.readtext(img_rgb)
        if result:
            for (_, text, _) in result:
                detected_texts.append(text)

    final_text = " ".join(detected_texts)

    # Jika teks yang diekstrak terlalu pendek, anggap tidak ada teks.
    if len(final_text.strip()) < 5:
        return "tidak ada teks"

    return final_text


