import tkinter as tk
from module.gui import AppGUI
from module.klasifikasi_citra import classify_image as classify_image_cnn
from module.ekstraksi_teks import extract_text
from module.klasifikasi_teks import classify_text as classify_text_bert
from module.fusi import fuse_results
import cv2

# === LABEL MAP ===
label_map = {
    0: "Iklan Judi Online",
    1: "Iklan Non Judi Online"
}

# === FUNGSI UTAMA KLASIFIKASI ===
def process_and_classify(image_path, mode, cnn_model, ocr_engine, show_result, show_text, cache):
    """
    Fungsi utama untuk memproses dan mengklasifikasikan gambar berdasarkan mode yang dipilih.
    Fungsi ini mengintegrasikan semua modul: klasifikasi citra, ekstraksi teks, 
    klasifikasi teks, dan fusi, serta menggunakan cache untuk optimasi.
    """
    try:
        # Baca gambar sekali untuk digunakan di semua modul
        image_data = cv2.imread(image_path)
        if image_data is None:
            raise FileNotFoundError(f"Tidak dapat membaca gambar dari: {image_path}")

        pred_cnn = None
        pred_bert = None
        extracted_text = ""

        # 1. Klasifikasi Citra (CNN)
        if mode in ("cnn", "both"):
            cache_key_cnn = f"cnn_{cnn_model}"
            if cache_key_cnn in cache:
                pred_cnn = cache[cache_key_cnn]
                print(f"CNN Predict (from cache): Logit={pred_cnn} => Label='{label_map.get(pred_cnn, 'Tidak Diketahui')}'")
            else:
                pred_cnn = classify_image_cnn(image_data, model_choice=cnn_model)
                cache[cache_key_cnn] = pred_cnn
                print(f"CNN Predict: Logit={pred_cnn} => Label='{label_map.get(pred_cnn, 'Tidak Diketahui')}'")

        # 2. Ekstraksi Teks (OCR) dan Klasifikasi Teks (BERT)
        if mode in ("ocr_bert", "both"):
            bert_model_choice = 'easyocr_bert' if ocr_engine == 'easyocr' else 'paddleocr_bert'
            
            # Cek cache untuk hasil OCR
            cache_key_ocr = f"ocr_{ocr_engine}"
            if cache_key_ocr in cache:
                extracted_text = cache[cache_key_ocr]
                print(f"OCR Text (from cache, using {ocr_engine}): {extracted_text}")
            else:
                extracted_text = extract_text(image_data, ocr_engine=ocr_engine)
                cache[cache_key_ocr] = extracted_text
                print(f"OCR Text (using {ocr_engine}): {extracted_text}")

            # Cek cache untuk hasil klasifikasi teks
            cache_key_bert = f"bert_{bert_model_choice}"
            if extracted_text.strip() and extracted_text != "tidak ada teks":
                if cache_key_bert in cache:
                    pred_bert = cache[cache_key_bert]
                    print(f"BERT Predict (from cache): Logit={pred_bert} => Label='{label_map.get(pred_bert, 'Tidak Diketahui')}'")
                else:
                    pred_bert = classify_text_bert(extracted_text, model_choice=bert_model_choice)
                    cache[cache_key_bert] = pred_bert
                    print(f"BERT Predict: Logit={pred_bert} => Label='{label_map.get(pred_bert, 'Tidak Diketahui')}'")
            else:
                pred_bert = 1 # Jika tidak ada teks, asumsikan bukan judi
                extracted_text = "tidak ada teks" # Pastikan teks konsisten
                print(f"BERT Predict: No text found, defaulting to Logit=1")


        # 3. Fusi Hasil
        final_pred = 1 # Default ke "Bukan Judi"
        if mode == "both":
            # Fusi hanya dilakukan jika kedua prediktor ada
            if pred_cnn is not None and pred_bert is not None:
                final_pred = fuse_results(text_logit=pred_bert, image_logit=pred_cnn)
        elif mode == "cnn":
            final_pred = pred_cnn if pred_cnn is not None else 1
        elif mode == "ocr_bert":
            final_pred = pred_bert if pred_bert is not None else 1

        # 4. Tampilkan Hasil
        is_judi = (final_pred == 0)
        result_text = f"🟥 {label_map[0]}" if is_judi else f"🟩 {label_map[1]}"
        print(f"Final Prediction: {result_text}")

        if show_result:
            show_result(result_text, is_judi)

        if show_text and mode in ("ocr_bert", "both"):
            show_text(extracted_text)

    except Exception as e:
        # Menampilkan error di GUI jika terjadi masalah
        error_message = f"Terjadi Error: {e}"
        print(error_message)
        if show_result:
            show_result(error_message, True) # Tampilkan dengan latar merah

# === MAIN ENTRY POINT ===
if __name__ == "__main__":
    root = tk.Tk()
    # Berikan fungsi process_and_classify ke AppGUI
    app = AppGUI(root, process_and_classify)
    root.mainloop()
