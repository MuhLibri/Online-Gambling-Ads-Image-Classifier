import tkinter as tk
from gui import AppGUI
import os
import tempfile
import numpy as np
from PIL import Image
from io import BytesIO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# OCR
import easyocr
from paddleocr import PaddleOCR

# BERT
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import tempfile


# === LABEL MAP ===
label_map = {
    0: "Iklan judi online",
    1: "Iklan non judi online"
}


# === LOAD MODELS ===
# CNN Models
cnn_models = {
    "efficientnet": load_model("models/CNN/efficientnet_model(20).keras"),
    "resnet": load_model("models/CNN/resnet_model(15).keras")
}

# BERT Models
bert_models = {
    "easyocr": {
        "model": TFBertForSequenceClassification.from_pretrained("models/BERT/bert_easyocr"),
        "tokenizer": BertTokenizer.from_pretrained("models/BERT/bert_easyocr")
    },
    "paddleocr": {
        "model": TFBertForSequenceClassification.from_pretrained("models/BERT/bert_paddleocr"),
        "tokenizer": BertTokenizer.from_pretrained("models/BERT/bert_paddleocr")
    }
}

# OCR Engines
ocr_engines = {
    "easyocr": easyocr.Reader(['id']),
    "paddleocr": PaddleOCR(use_angle_cls=True, lang='id')
}


# === IMAGE PROCESSING ===
def preprocess_image_for_cnn(img_path, cnn_model_name):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    if cnn_model_name == "efficientnet":
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    elif cnn_model_name == "resnet":
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    return img_array


def convert_image_to_jpg_in_memory(img_path):
    img = Image.open(img_path).convert('RGB')
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


# === OCR EXTRACT ===
def extract_text(image_path, engine="easyocr"):
    img_bytes = convert_image_to_jpg_in_memory(image_path)

    if engine == "easyocr":
        result = ocr_engines["easyocr"].readtext(img_bytes, detail=0)
        return " ".join(result)

    elif engine == "paddleocr":
        try:
            # Try direct path first (most stable for PaddleOCR)
            result = ocr_engines["paddleocr"].ocr(image_path, cls=True)
            lines = [line[1][0] for line in result[0]]
            return " ".join(lines)
        except Exception:
            # If error, fallback to temp file (sometimes helps on Windows)
            img = Image.open(image_path).convert("RGB")
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
                img.save(tmp_path)
            try:
                # Re-instantiate PaddleOCR to avoid internal state/caching issues
                paddleocr_instance = PaddleOCR(use_angle_cls=True, lang='id')
                result = paddleocr_instance.ocr(tmp_path, cls=True)
                lines = [line[1][0] for line in result[0]]
                return " ".join(lines)
            finally:
                os.remove(tmp_path)

    return ""


# === BERT CLASSIFY ===
def classify_text(text, engine="easyocr"):
    tokenizer = bert_models[engine]["tokenizer"]
    model = bert_models[engine]["model"]
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=512)
    outputs = model(inputs)
    probs = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]
    pred = int(np.argmax(probs))  # 0: Judi, 1: Non-Judi
    return pred, probs


# === MAIN ENTRY ===
def classify_image(img_path, mode, cnn_model, ocr_engine, show_result, show_text):
    pred_cnn = None
    pred_bert = None

    if mode in ("cnn", "both"):
        img = preprocess_image_for_cnn(img_path, cnn_model)
        model = cnn_models[cnn_model]
        raw_pred = model.predict(img)[0][0]
        pred_cnn = int(raw_pred > 0.5)
        print(f"CNN Predict: {raw_pred:.4f} => {label_map[pred_cnn]}")

    if mode in ("ocr_bert", "both"):
        text = extract_text(img_path, engine=ocr_engine)
        print("OCR Text:", text)
        if text.strip():
            pred_bert, _ = classify_text(text, engine=ocr_engine)
        else:
            pred_bert = 0
        print("BERT Predict:", label_map[pred_bert])

    # Combine result
    if mode == "both":
        final_pred = int(pred_cnn == 1 and pred_bert == 1)
    elif mode == "cnn":
        final_pred = pred_cnn
    elif mode == "ocr_bert":
        final_pred = pred_bert
    else:
        final_pred = 0

    is_judi = final_pred == 0
    result_text = "🟥 Iklan Judi Online" if is_judi else "🟩 Iklan Non Judi Online"
    print("Final Prediction:", result_text)

    if show_result:
        show_result(result_text, is_judi)

    if show_text and mode in ("ocr_bert", "both"):
        show_text(text)


if __name__ == "__main__":
    import tkinter as tk
    from gui import AppGUI
    root = tk.Tk()
    app = AppGUI(root, classify_image)
    root.mainloop()
