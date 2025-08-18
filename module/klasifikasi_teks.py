"""
Modul untuk klasifikasi teks menggunakan model BERT (EasyOCR atau PaddleOCR).
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Tentukan path absolut ke direktori root proyek
# Ini membuat path menjadi independen dari lokasi eksekusi skrip
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Definisikan path ke model-model yang telah dilatih
PADDLEOCR_BERT_PATH = os.path.join(PROJECT_ROOT, "model", "OCR_BERT", "PaddleOCR_BERT")
EASYOCR_BERT_PATH = os.path.join(PROJECT_ROOT, "model", "OCR_BERT", "EasyOCR_BERT")

# Muat tokenizer dan model hanya sekali saat modul diimpor
try:
    EASYOCR_TOKENIZER = AutoTokenizer.from_pretrained(EASYOCR_BERT_PATH)
    EASYOCR_MODEL = AutoModelForSequenceClassification.from_pretrained(EASYOCR_BERT_PATH)

    PADDLEOCR_TOKENIZER = AutoTokenizer.from_pretrained(PADDLEOCR_BERT_PATH)
    PADDLEOCR_MODEL = AutoModelForSequenceClassification.from_pretrained(PADDLEOCR_BERT_PATH)
except Exception as e:
    raise RuntimeError(f"Gagal memuat model atau tokenizer. Pastikan path sudah benar dan file model ada. Error: {e}")

def classify_text(text, model_choice='easyocr_bert'):
    """
    Mengklasifikasikan sebuah kalimat menggunakan model BERT yang dipilih.

    Args:
        text (str): Kalimat yang akan diklasifikasikan.
        model_choice (str): Pilihan model, 'easyocr_bert' atau 'paddleocr_bert'. 
                              Defaultnya 'easyocr_bert'.

    Returns:
        int: Hasil klasifikasi (0 atau 1).
    
    Raises:
        ValueError: Jika model_choice tidak valid.
    """
    if model_choice == 'easyocr_bert':
        tokenizer = EASYOCR_TOKENIZER
        model = EASYOCR_MODEL
    elif model_choice == 'paddleocr_bert':
        tokenizer = PADDLEOCR_TOKENIZER
        model = PADDLEOCR_MODEL
    else:
        raise ValueError("Pilihan model tidak valid. Gunakan 'easyocr_bert' atau 'paddleocr_bert'.")

    # Tokenisasi input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    # Lakukan inferensi
    model.eval()  # Set model ke mode evaluasi
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Dapatkan prediksi
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()

    return predicted_class_id
