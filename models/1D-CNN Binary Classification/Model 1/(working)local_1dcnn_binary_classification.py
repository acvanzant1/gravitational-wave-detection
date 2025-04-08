import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout, MaxPool1D, ReLU, Flatten, BatchNormalization
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import tensorflow.keras.backend as K
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import time
start_time = time.time()

def focal_loss(gamma=1.0, alpha=0.5):
    def focal_loss_fixed(y_true, y_pred):
        y_true = K.cast(y_true, dtype='int32')
        y_true = K.one_hot(y_true, num_classes=2)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return K.sum(loss, axis=1)
    return focal_loss_fixed

# Define constants
samples_per_class = 5000
no_of_classes = 2

# Load datasets
noise_df = pd.read_csv("Downloads/Gravitational-Wave-Detection-Using-Deep-Learning/raw_data_files/Final_Merged_Noise_Reduced_No_Abs.csv", header=None)
data_df = pd.read_csv("Downloads/Gravitational-Wave-Detection-Using-Deep-Learning/raw_data_files/Final_BBH_Merged_Noise_Signal_Reduced_No_ABS.csv", header=None)

# Check for missing values
print("\nMissing Values in Noise Data:", noise_df.isnull().sum().sum())
print("Missing Values in Signal Data:", data_df.isnull().sum().sum())

# Fill missing values with median
noise_df.fillna(noise_df.median(), inplace=True)
data_df.fillna(data_df.median(), inplace=True)

# Remove outliers beyond 4 standard deviations
def remove_outliers(df, name="DataFrame"):
    mean = df.mean()
    std = df.std()
    clipped_df = df.clip(lower=mean - 4 * std, upper=mean + 4 * std, axis=1)
    print(f"{name}: {len(clipped_df)} rows after clipping outliers.")
    return clipped_df

noise_df = remove_outliers(noise_df, name="Noise Data")
data_df = remove_outliers(data_df, name="Signal Data")

# Print statistics after outlier removal
print("\nAfter Outlier Removal - Noise Data Stats:\n", noise_df.describe())
print("\nAfter Outlier Removal - Signal Data Stats:\n", data_df.describe())

# Define labels
noise_y = np.zeros((noise_df.shape[0], 1), dtype=int)  # Label all noise as 0
data_y = np.ones((data_df.shape[0], 1), dtype=int)  # Label all signals as 1

# Combine datasets
X = np.concatenate((noise_df.values, data_df.values), axis=0)
Y = np.concatenate((noise_y, data_y), axis=0)  # Ensure Y is a column vector

# Sanity checks before training
print("\nFinal Data Summary:")
print("X Shape:", X.shape)
print("Y Shape:", Y.shape)
print("Class Distribution:", dict(zip(*np.unique(Y, return_counts=True))))
print("X Sample:", X[:5])
print("Y Sample:", Y[:5])

# Verify class balance before training
unique, counts = np.unique(Y, return_counts=True)
print("Class Distribution:", dict(zip(unique, counts)))

# Ensure X and Y have the same number of samples
min_samples = min(X.shape[0], Y.shape[0])
X = X[:min_samples]
Y = Y[:min_samples]
Y = Y.reshape(-1)  # Convert to 1D integer labels
print(Y.shape)  # Debugging check
Y = Y.reshape(-1, 1)  # Ensure Y is a column vector

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Ensure X and Y stay separate
X = np.expand_dims(X, axis=-1)

# Ensure Y is 1D for sparse categorical cross-entropy
Y = Y.squeeze().astype(int)  # Convert Y to integer labels

# Compute class weights based on training data
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(Y), y=Y)
class_weights = dict(enumerate(class_weights))

# Sanity check before training
print("\nFinal Data Summary:")
print("X Shape:", X.shape)
print("Y Shape:", Y.shape)
print("Class Distribution:", dict(zip(*np.unique(Y, return_counts=True))))
print("X Sample:", X[:5])
print("Y Sample:", Y[:5])

# Verify class balance before training
unique, counts = np.unique(Y, return_counts=True)
print("Class Distribution:", dict(zip(unique, counts)))

# Print samples of X and Y before training
print("Sample X:", X[:5])
print("Sample Y:", Y[:5])

# Define model
model = Sequential([
    Conv1D(64, 32, input_shape=(X.shape[1], 1)),
    BatchNormalization(),
    ReLU(),
    MaxPool1D(4),

    Conv1D(128, 64),
    BatchNormalization(),
    ReLU(),
    MaxPool1D(4),

    Conv1D(256, 64),
    BatchNormalization(),
    ReLU(),
    MaxPool1D(4),

    Conv1D(512, 128),
    BatchNormalization(),
    ReLU(),
    MaxPool1D(4),

    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, amsgrad=True, clipnorm=1.0),
    loss=focal_loss(gamma=2.0, alpha=0.75),
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Train model
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

# Re-train the model on training split
history = model.fit(
    X_train, Y_train,
    batch_size=32,
    epochs=10,
    validation_split=0.2,
    class_weight=class_weights  # <- use computed weights here
)

# Predict on test data
val_preds = model.predict(X_test)
val_pred_labels = np.argmax(val_preds, axis=1)

print("Predicted class distribution:", np.bincount(val_pred_labels))
print("\nClassification Report:")
print(classification_report(Y_test, val_pred_labels, digits=4))

# Compute and print precision, recall, and F1-score
precision = precision_score(Y_test, val_pred_labels)
recall = recall_score(Y_test, val_pred_labels)
f1 = f1_score(Y_test, val_pred_labels)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Plot training & validation accuracy and loss
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('cnn_training_results.png')
plt.show()

# Confusion matrix
cm = confusion_matrix(Y_test, val_pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Noise', 'Signal'], yticklabels=['Noise', 'Signal'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('cnn_confusion_matrix.png')
plt.show()

end_time = time.time()
total_time = end_time - start_time
print(f"\nTotal Runtime: {total_time:.2f} seconds")