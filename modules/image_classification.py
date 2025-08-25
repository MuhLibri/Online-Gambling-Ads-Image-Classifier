import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import cv2


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
EFFICIENTNET_PATH = os.path.join(PROJECT_ROOT, "models", "CNN", "EfficientNet-B0_final_model.pt")
RESNET_PATH = os.path.join(PROJECT_ROOT, "models", "CNN", "ResNet-50_final_model.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(model_type, path, num_classes=1):
    model = None
    if model_type == 'efficientnet':
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    elif model_type == 'resnet':
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )
    else:
        raise ValueError("model_type must be 'efficientnet' or 'resnet'")

    # Load the state dict with weights_only=True for security
    # Ensure the .pt model file exists at the correct path
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


try:
    EFFICIENTNET_MODEL = _load_model('efficientnet', EFFICIENTNET_PATH)
    RESNET_MODEL = _load_model('resnet', RESNET_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load CNN models. Ensure the path is correct and the model files exist. Error: {e}")

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def classify_image(image_input, model_choice='efficientnet'):
    """
    Args:
        image_input (str or numpy.ndarray): Path to the image file or image 
                                             in numpy array format (BGR or RGB).
        model_choice (str): Model choice, 'efficientnet' or 'resnet'. 
                            Defaults to 'efficientnet'.

    Returns:
        int: Classification result (0 or 1).
    
    Raises:
        ValueError: If model_choice is invalid.
        TypeError: If the input type is not supported.
    """
    if model_choice == 'efficientnet':
        model = EFFICIENTNET_MODEL
    elif model_choice == 'resnet':
        model = RESNET_MODEL
    else:
        raise ValueError("Invalid model choice. Use 'efficientnet' or 'resnet'.")

    if isinstance(image_input, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)).convert('RGB')
    else:
        raise TypeError("image_input must be a path (str) or numpy.ndarray")

    image_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)
        # Apply sigmoid function to get probability
        predicted_prob = torch.sigmoid(output).item()
        print(f"Predicted Probability: {predicted_prob:.4f}")
        # Convert probability to a binary class (0 or 1)
        predicted = int(predicted_prob >= 0.5)
    
    return predicted
