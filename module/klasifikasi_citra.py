"""
Modul untuk klasifikasi citra menggunakan model CNN (EfficientNet-B0 atau ResNet-50).
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import cv2

# Tentukan path absolut ke direktori root proyek
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Definisikan path ke model-model yang telah dilatih
EFFICIENTNET_PATH = os.path.join(PROJECT_ROOT, "model", "CNN", "EfficientNet_B0(1e-4).pt")
RESNET_PATH = os.path.join(PROJECT_ROOT, "model", "CNN", "ResNet50(1e-4).pt")

# Tentukan device (CPU atau GPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_model(model_type, path, num_classes=1):
    """Memuat model dan memodifikasi layer terakhir agar sesuai dengan training."""
    model = None
    if model_type == 'efficientnet':
        # Inisialisasi model tanpa bobot pre-trained
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        # Arsitektur classifier harus sama persis dengan saat training
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    elif model_type == 'resnet':
        # Inisialisasi model tanpa bobot pre-trained
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        # Arsitektur classifier harus sama persis dengan saat training
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )
    else:
        raise ValueError("model_type harus 'efficientnet' atau 'resnet'")

    # Muat state dict dengan weights_only=True untuk keamanan
    # Pastikan file model .pt ada di path yang benar
    if not os.path.exists(path):
        raise FileNotFoundError(f"File model tidak ditemukan di: {path}")
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# Muat model hanya sekali saat modul diimpor
try:
    EFFICIENTNET_MODEL = _load_model('efficientnet', EFFICIENTNET_PATH)
    RESNET_MODEL = _load_model('resnet', RESNET_PATH)
except Exception as e:
    raise RuntimeError(f"Gagal memuat model CNN. Pastikan path sudah benar dan file model ada. Error: {e}")


# Definisikan transformasi gambar (sesuaikan dengan yang digunakan saat training)
# Ukuran input untuk EfficientNet-B0 dan ResNet adalah 224x224
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def classify_image(image_input, model_choice='efficientnet'):
    """
    Mengklasifikasikan sebuah gambar menggunakan model CNN yang dipilih.

    Args:
        image_input (str atau numpy.ndarray): Path ke file gambar atau gambar 
                                             dalam format numpy array (BGR atau RGB).
        model_choice (str): Pilihan model, 'efficientnet' atau 'resnet'. 
                            Defaultnya 'efficientnet'.

    Returns:
        int: Hasil klasifikasi (0 atau 1).
    
    Raises:
        ValueError: Jika model_choice tidak valid.
        FileNotFoundError: Jika path gambar tidak ditemukan.
        TypeError: Jika tipe input tidak didukung.
    """
    if model_choice == 'efficientnet':
        model = EFFICIENTNET_MODEL
    elif model_choice == 'resnet':
        model = RESNET_MODEL
    else:
        raise ValueError("Pilihan model tidak valid. Gunakan 'efficientnet' atau 'resnet'.")

    # Buka gambar dan konversi ke PIL Image
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Gambar tidak ditemukan di path: {image_input}")
        image = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, np.ndarray):
        # Konversi numpy array (dari OpenCV BGR atau lainnya) ke RGB
        image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)).convert('RGB')
    else:
        raise TypeError("image_input harus berupa path (str) atau numpy.ndarray")

    # Terapkan transformasi dan tambahkan batch dimension
    image_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    # Lakukan inferensi
    with torch.no_grad():
        output = model(image_tensor)
        # Terapkan fungsi sigmoid untuk mendapatkan probabilitas
        predicted_prob = torch.sigmoid(output).item()
        print(f"Predicted Probability: {predicted_prob:.4f}")
        # Konversi probabilitas ke kelas biner (0 atau 1)
        predicted = int(predicted_prob >= 0.5)
    
    return predicted
