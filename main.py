import tkinter as tk
from modules.gui import AppGUI
from modules.image_classification import classify_image as classify_image_cnn
from modules.text_extraction import extract_text
from modules.text_classification import classify_text as classify_text_bert
from modules.fusion import fuse_results
import cv2

# === LABEL MAP ===
label_map = {
    0: "Online Gambling Ad",
    1: "Non-Online Gambling Ad"
}

# === MAIN CLASSIFICATION FUNCTION ===
def process_and_classify(image_path, mode, cnn_model, ocr_engine, show_result, show_text, cache):
    """
    Main function to process and classify images based on the selected mode.
    This function integrates all modules: image classification, text extraction, 
    text classification, and fusion, and uses a cache for optimization.
    """
    try:
        # Read the image once to be used in all modules
        image_data = cv2.imread(image_path)
        if image_data is None:
            raise FileNotFoundError(f"Could not read image from: {image_path}")

        pred_cnn = None
        pred_bert = None
        extracted_text = ""

        # 1. Image Classification (CNN)
        if mode in ("cnn", "both"):
            cache_key_cnn = f"cnn_{cnn_model}"
            if cache_key_cnn in cache:
                pred_cnn = cache[cache_key_cnn]
                print(f"CNN Predict (from cache): Logit={pred_cnn} => Label='{label_map.get(pred_cnn, 'Unknown')}'")
            else:
                pred_cnn = classify_image_cnn(image_data, model_choice=cnn_model)
                cache[cache_key_cnn] = pred_cnn
                print(f"CNN Predict: Logit={pred_cnn} => Label='{label_map.get(pred_cnn, 'Unknown')}'")

        # 2. Text Extraction (OCR) and Text Classification (BERT)
        if mode in ("ocr_bert", "both"):
            bert_model_choice = 'easyocr_bert' if ocr_engine == 'easyocr' else 'paddleocr_bert'
            
            # Check cache for OCR results
            cache_key_ocr = f"ocr_{ocr_engine}"
            if cache_key_ocr in cache:
                extracted_text = cache[cache_key_ocr]
                print(f"OCR Text (from cache, using {ocr_engine}): {extracted_text}")
            else:
                extracted_text = extract_text(image_data, ocr_engine=ocr_engine)
                cache[cache_key_ocr] = extracted_text
                print(f"OCR Text (using {ocr_engine}): {extracted_text}")

            # Check cache for text classification results
            cache_key_bert = f"bert_{bert_model_choice}"
            if extracted_text.strip() and extracted_text != "no text":
                if cache_key_bert in cache:
                    pred_bert = cache[cache_key_bert]
                    print(f"BERT Predict (from cache): Logit={pred_bert} => Label='{label_map.get(pred_bert, 'Unknown')}'")
                else:
                    pred_bert = classify_text_bert(extracted_text, model_choice=bert_model_choice)
                    cache[cache_key_bert] = pred_bert
                    print(f"BERT Predict: Logit={pred_bert} => Label='{label_map.get(pred_bert, 'Unknown')}'")
            else:
                pred_bert = 1 # If no text, assume not gambling
                extracted_text = "no text" # Ensure consistent text
                print(f"BERT Predict: No text found, defaulting to Logit=1")


        # 3. Fusion of Results
        final_pred = 1 # Default to "Not Gambling"
        if mode == "both":
            # Fusion is only performed if both predictors exist
            if pred_cnn is not None and pred_bert is not None:
                final_pred = fuse_results(text_logit=pred_bert, image_logit=pred_cnn)
        elif mode == "cnn":
            final_pred = pred_cnn if pred_cnn is not None else 1
        elif mode == "ocr_bert":
            final_pred = pred_bert if pred_bert is not None else 1

        # 4. Display Results
        is_judi = (final_pred == 0)
        result_text = f"🟥 {label_map[0]}" if is_judi else f"🟩 {label_map[1]}"
        print(f"Final Prediction: {result_text}")

        if show_result:
            show_result(result_text, is_judi)

        if show_text and mode in ("ocr_bert", "both"):
            show_text(extracted_text)

    except Exception as e:
        # Display error in GUI if a problem occurs
        error_message = f"An error occurred: {e}"
        print(error_message)
        if show_result:
            show_result(error_message, True) # Show with a red background

# === MAIN ENTRY POINT ===
if __name__ == "__main__":
    root = tk.Tk()
    app = AppGUI(root, process_and_classify)
    root.mainloop()
