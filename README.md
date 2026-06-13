# Comparative-Analysis-of-Resnet-Efficient-Net-MobileNetV2

## Overview

This project presents a comparative analysis of three state-of-the-art deep learning architectures for automated skin disease classification:

* ResNet50
* MobileNetV2
* EfficientNet

The objective is to classify skin lesion images into multiple categories using transfer learning and evaluate the performance of each model based on various classification metrics.

This work was developed as part of a research study focused on improving the accuracy and reliability of skin disease detection using deep learning techniques.

---

## Dataset

The dataset used in this project was obtained from the **International Skin Imaging Collaboration (ISIC)** archive.

### Data Preparation

The dataset was processed using **Roboflow** for:

* Image resizing
* Dataset organization
* Data augmentation
* Training/Validation/Test splitting

### Dataset Structure

```
dataset/
│
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
│
├── valid/
│   ├── class_1/
│   ├── class_2/
│   └── ...
│
└── test/
    ├── class_1/
    ├── class_2/
    └── ...
```

The dataset folder contains three main directories:

* Train
* Valid
* Test

Each class folder contains corresponding skin lesion images.

---

## Models Implemented

### 1. ResNet50

Residual Neural Network architecture utilizing skip connections for efficient gradient propagation and improved classification performance.

### 2. MobileNetV2

Lightweight convolutional neural network optimized for efficient deployment while maintaining competitive accuracy.

### 3. EfficientNet

Scalable architecture that balances network depth, width, and resolution for superior performance with fewer parameters.

---

## Project Structure

```
Skin-Disease-Classification/
│
├── dataset/
│   ├── train/
│   ├── valid/
│   └── test/
│
├── models/
│   ├── ResNet50_Model.py
│   ├── MobileNetV2_Model.py
│   └── EfficientNet_Model.py
│
├── ModelRunner.py
│
├── results/
│   ├── confusion_matrix/
│   ├── graphs/
│   └── metrics/
│
├── requirements.txt
│
└── README.md
```

---

## Features

* Skin disease image classification
* Dataset preprocessing using Roboflow
* Transfer learning implementation
* Training and validation monitoring
* Comparative model evaluation
* Accuracy and loss visualization
* Confusion matrix generation
* Classification report generation

---

## Evaluation Metrics

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Training Loss
* Validation Loss

---

## Required Libraries

Install all dependencies before running the project.

```bash
pip install tensorflow
pip install keras
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install pandas
pip install opencv-python
pip install pillow
```

Or install everything at once:

```bash
pip install tensorflow keras numpy matplotlib seaborn scikit-learn pandas opencv-python pillow
```

---

## Imports Used

```python
import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import EfficientNetB0

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import seaborn as sns
```

---

## Dataset Directory Configuration

Update the dataset paths according to your local system.

```python
train_dir = "dataset/train"
valid_dir = "dataset/valid"
test_dir  = "dataset/test"
```

Example Windows Path:

```python
train_dir = r"C:\Users\Username\dataset\train"
valid_dir = r"C:\Users\Username\dataset\valid"
test_dir = r"C:\Users\Username\dataset\test"
```

---

## Running Individual Models

### ResNet50

```bash
python ResNet50_Model.py
```

### MobileNetV2

```bash
python MobileNetV2_Model.py
```

### EfficientNet

```bash
python EfficientNet_Model.py
```

---

## Running Comparative Evaluation

Execute the model runner:

```bash
python ModelRunner.py
```

The Model Runner evaluates all models and generates performance metrics for comparison.

---

## Results

The project compares:

* ResNet50
* MobileNetV2
* EfficientNet

based on classification performance and computational efficiency.

Generated outputs include:

* Accuracy graphs
* Loss graphs
* Confusion matrices
* Classification reports

---

## Research Objective

To analyze and compare multiple deep learning architectures for skin disease classification and identify the most effective model for automated diagnosis support systems.

---

## Author

**D. Naga Sujith Kumar**

B.Tech Information Technology

VIT Vellore

---

## Acknowledgements

* International Skin Imaging Collaboration (ISIC)
* Roboflow
* TensorFlow
* Keras
* Scikit-Learn
* OpenCV
