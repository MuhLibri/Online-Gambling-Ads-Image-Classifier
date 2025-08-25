import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PADDLEOCR_BERT_PATH = os.path.join(PROJECT_ROOT, "models", "OCR_BERT", "PaddleOCR_BERT")
EASYOCR_BERT_PATH = os.path.join(PROJECT_ROOT, "models", "OCR_BERT", "EasyOCR_BERT")

try:
    EASYOCR_TOKENIZER = AutoTokenizer.from_pretrained(EASYOCR_BERT_PATH)
    EASYOCR_MODEL = AutoModelForSequenceClassification.from_pretrained(EASYOCR_BERT_PATH)

    PADDLEOCR_TOKENIZER = AutoTokenizer.from_pretrained(PADDLEOCR_BERT_PATH)
    PADDLEOCR_MODEL = AutoModelForSequenceClassification.from_pretrained(PADDLEOCR_BERT_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model or tokenizer. Ensure the path is correct and the model files exist. Error: {e}")


def classify_text(text, model_choice='easyocr_bert'):
    """
    Args:
        text (str): The sentence to be classified.
        model_choice (str): Model choice, 'easyocr_bert' or 'paddleocr_bert'. 
                              Defaults to 'easyocr_bert'.

    Returns:
        int: Classification result (0 or 1).
    
    Raises:
        ValueError: If model_choice is invalid.
    """
    if model_choice == 'easyocr_bert':
        tokenizer = EASYOCR_TOKENIZER
        model = EASYOCR_MODEL
    elif model_choice == 'paddleocr_bert':
        tokenizer = PADDLEOCR_TOKENIZER
        model = PADDLEOCR_MODEL
    else:
        raise ValueError("Invalid model choice. Use 'easyocr_bert' or 'paddleocr_bert'.")

    # Tokenize the input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    # Perform inference
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the prediction
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()

    return predicted_class_id
