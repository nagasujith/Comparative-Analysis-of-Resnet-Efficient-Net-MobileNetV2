import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve, auc, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Paths for your dataset
test_csv_path = r"C:\Users\D. Naga sujith kumar\Desktop\Capstone\Skin Disease Dataset V2.v2i.multiclass\test\_classes.csv"
test_img_dir = r"C:\Users\D. Naga sujith kumar\Desktop\Capstone\Skin Disease Dataset V2.v2i.multiclass\test"

# Image size and batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Custom Data Generator (from your previous code)
class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, img_dir, batch_size, img_size, augment=False):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.indices = self.dataframe.index.values
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = self.dataframe.iloc[batch_indices]
        return self.__data_generation(batch_data)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def __data_generation(self, batch_data):
        X = np.empty((self.batch_size, *self.img_size, 3))
        y = np.zeros((self.batch_size, len(batch_data.columns) - 1))  # Adjust based on the number of labels

        for i, (_, row) in enumerate(batch_data.iterrows()):
            img_path = os.path.join(self.img_dir, row['filename'])
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=self.img_size)
            img = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalize
            X[i,] = img
            y[i, :] = row[1:].values  # Assuming the first column is 'filename'

        return X, y

# Load the test dataset
test_df = pd.read_csv(test_csv_path)

# Define the test data generator
test_generator = CustomDataGenerator(test_df, test_img_dir, BATCH_SIZE, IMG_SIZE)

# Load the trained model
model = load_model(r"C:\Users\D. Naga sujith kumar\Desktop\Capstone\final_trained_model_efficientnet.keras")

# Generate predictions and true labels
y_true = []
y_pred = []
y_pred_prob = []

for i in range(len(test_generator)):
    X, y = test_generator[i]  # Get a batch of test data
    predictions = model.predict(X)  # Make predictions
    y_true.extend(y)  # True labels
    y_pred_prob.extend(predictions)  # Store predicted probabilities
    y_pred.extend((predictions > 0.5).astype(int))  # Convert probabilities to binary predictions

# Convert to NumPy arrays for easier handling
y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_pred_prob = np.array(y_pred_prob)

# Classification report (precision, recall, F1 score for each class)
print(classification_report(y_true, y_pred, target_names=['acrochordon', 'lichenoid_keratosis', 'vascular_lesion']))

# Plot Precision-Recall Curve and ROC Curve for each label
for i in range(y_true.shape[1]):  # Loop over each class
    precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred_prob[:, i])
    fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)

    # Precision-Recall Curve
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.title(f"Precision-Recall Curve (Label {i})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(f"PR_curve_label_{i}.png")
    plt.show()

    # ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (Label {i})')
    plt.legend(loc="lower right")
    plt.savefig(f"ROC_curve_label_{i}.png")
    plt.show()
