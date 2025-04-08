import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# RESNET IMPORTS 
import time
import matplotlib
matplotlib.use('agg')

start_time = time.time()

directory = ""
noise_df = pd.read_csv(directory + "Final_Merged_Noise_Reduced_No_Abs.csv", header=None)
data_BBH_df = pd.read_csv(directory + "Final_BBH_Merged_Noise_Signal_Reduced_No_ABS.csv", header=None)

# Check for missing values
if noise_df.isnull().values.any() or data_BBH_df.isnull().values.any():
    raise ValueError("Missing values found in the data.")

# Outlier clipping
noise_df = noise_df.clip(lower=noise_df.quantile(0.01), upper=noise_df.quantile(0.99), axis=1)
data_BBH_df = data_BBH_df.clip(lower=data_BBH_df.quantile(0.01), upper=data_BBH_df.quantile(0.99), axis=1)

# Z-score normalization
noise = (noise_df - noise_df.mean()) / noise_df.std()
data_BBH = (data_BBH_df - data_BBH_df.mean()) / data_BBH_df.std()

X = np.concatenate((noise.values.astype(np.float32), data_BBH.values.astype(np.float32)), axis=0)
print(len(noise_df.index))
print(X.shape)

samples_per_class = 5000
no_of_classes = 2
y = [int(i / samples_per_class) for i in range(samples_per_class * no_of_classes)]
y = tf.keras.utils.to_categorical(y)
print(y)

X = np.expand_dims(X, axis=-1)
print(X.shape)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

print(X_train.shape, y_train.shape)

class Classifier_RESNET:
    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, load_weights=False):
        self.output_directory = output_directory
        if build:
            self.model = self.build_model(input_shape, nb_classes)
            if verbose:
                self.model.summary()
            self.verbose = verbose
            if load_weights:
                self.model.load_weights(self.output_directory + '/model_init.hdf5')
            else:
                self.model.save_weights(self.output_directory + 'model_init.weights.h5')

    def build_model(self, input_shape, nb_classes):
        n_feature_maps = 64
        input_layer = keras.layers.Input(input_shape)

        # BLOCK 1
        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)
        output_block_1 = keras.layers.Dropout(0.3)(output_block_1)

        # BLOCK 2
        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)
        output_block_2 = keras.layers.Dropout(0.3)(output_block_2)

        # BLOCK 3
        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        shortcut_y = keras.layers.BatchNormalization()(output_block_2)
        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)
        output_block_3 = keras.layers.Dropout(0.3)(output_block_3)

        # FINAL (GAP + Dense)
        gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)
        output_layer = keras.layers.Dense(nb_classes, activation='softmax',
                                          kernel_regularizer=regularizers.l2(0.001))(gap_layer)

        optimizer = Adam(learning_rate=0.001, decay=0.0)
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def fit(self, x_train, y_train, x_val, y_val, BATCH_SIZE=32, nb_epochs=5):
        start_time = time.time()
        callbacks = []
        history = self.model.fit(
            x_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=nb_epochs,
            validation_data=(x_val, y_val),
            callbacks=callbacks
        )
        keras.backend.clear_session()
        return history

RESNET = Classifier_RESNET(directory, (16384, 1), 2, verbose=True)

BATCH_SIZE = 32
EPOCHS = 10
history = RESNET.fit(X_train, y_train, X_val, y_val, BATCH_SIZE, EPOCHS)

import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history.get('val_accuracy', None)
loss = history.history['loss']
val_loss = history.history.get('val_loss', None)

plt.figure(figsize=(13, 6))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy', marker='o')
if val_acc is not None:
    plt.plot(val_acc, label='Validation Accuracy', marker='x', linestyle='dashed')
plt.legend()
plt.title('Training & Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss', marker='o')
if val_loss is not None:
    plt.plot(val_loss, label='Validation Loss', marker='x', linestyle='dashed')
plt.legend()
plt.title('Training & Validation Loss')

plt.savefig("models/ResNet Binary Classification/Model 1/resnet_training_results.png", dpi=300, bbox_inches='tight')
print("Training results saved to 'resnet_training_results.png'.")

pred = RESNET.model.predict(X)
y_pred = np.argmax(pred, axis=1)
y_true = np.argmax(y, axis=1)

precision = precision_score(y_true, y_pred, average='binary')
recall = recall_score(y_true, y_pred, average='binary')
score = f1_score(y_true, y_pred, average='binary')

print(f'Precision: {precision:.3f}')
print(f'Recall: {recall:.3f}')
print(f'F1-Score: {score:.3f}')

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.savefig("models/ResNet Binary Classification/Model 1/resnet_confusion_matrix.png", dpi=300, bbox_inches='tight')
print("Confusion matrix saved to 'resnet_confusion_matrix.png'.")

end_time = time.time()
total_time = end_time - start_time
print(f"Total runtime: {total_time:.2f} seconds")