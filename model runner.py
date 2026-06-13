import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import os
import sys

# Set UTF-8 encoding for standard output
sys.stdout.reconfigure(encoding='utf-8')


# Function to load Keras models from a specified path
def load_keras_model(model_path):
    return load_model(model_path)

# Define the image preprocessing function
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))  # Resize to the size required by the models
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to predict with all models and display disease classification
def predict(image_tensor, resnet, efficientnet, mobilenetv2, class_labels):
    resnet_pred = resnet.predict(image_tensor)
    efficientnet_pred = efficientnet.predict(image_tensor)
    mobilenetv2_pred = mobilenetv2.predict(image_tensor)
    
    # Get the predicted class for each model
    resnet_class = class_labels[np.argmax(resnet_pred)]
    efficientnet_class = class_labels[np.argmax(efficientnet_pred)]
    mobilenetv2_class = class_labels[np.argmax(mobilenetv2_pred)]
    
    return resnet_class, efficientnet_class, mobilenetv2_class

# Function to select an image file
def select_image():
    root = tk.Tk()
    root.withdraw()  # Hide the Tkinter window
    image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    return image_path
                                            
# Main function to run the script
def main():
    # Specify the paths for the models
    resnet_path = r"C:\Users\D. Naga sujith kumar\Desktop\projects\skin disease ML\Skin Detection Model Files Multi-Label Classification\final_trained_model.keras"
    efficientnet_path = r"C:\Users\D. Naga sujith kumar\Desktop\projects\skin disease ML\Skin Detection Model Files Multi-Label Classification EfficientNet\final_trained_model_efficientnet.keras"
    mobilenetv2_path = r"C:\Users\D. Naga sujith kumar\Desktop\projects\skin disease ML\Skin Detection Model Files Multi-Label Classification MobileNetV2\final_trained_model_mobilenetv2.keras"
    
    # Load the models using their respective paths
    resnet = load_keras_model(resnet_path)
    efficientnet = load_keras_model(efficientnet_path)
    mobilenetv2 = load_keras_model(mobilenetv2_path)
    
    # Define the possible classes (diseases)
    class_labels = ['Acrochordon', 'Lichenoid Keratosis', 'Vascular Lesions']
    
    # Prompt the user to select an image
    image_path = r"C:\Users\D. Naga sujith kumar\Desktop\achrocordan.jpeg"
    
    if image_path:
        print(f"Selected image: {image_path}")
        # Preprocess the selected image
        image_tensor = preprocess_image(image_path)
        
        # Get predictions from all models
        resnet_class, efficientnet_class, mobilenetv2_class = predict(image_tensor, resnet, efficientnet, mobilenetv2, class_labels)
        
        # Print predictions
        print("ResNet Prediction:", resnet_class)
        print("EfficientNet Prediction:", efficientnet_class)
        print("MobileNetV2 Prediction:", mobilenetv2_class)
    else:
        print("No image selected.")

if __name__ == "__main__":
    main()
