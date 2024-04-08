import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from keras.applications import EfficientNetB0
from tensorflow import keras
from pathlib import Path
from keras.models import clone_model
from tensorflow.keras.callbacks import EarlyStopping

#I used this online resource in assisting with my code:
#https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/

# A link to the directory that will be used for this (change if small or large train)
data_dir = Path(r'C:\Users\Devon Scheg\Documents\Academics\Classes\ECE 500\Assignments\MiniProject\data\cropped')

print('Model done on scrubbed set 00,01...')
# Specify image params
batch_size = 32
img_height = 224
img_width = 224

# Splitting the train and test by 80/20 split 
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=122,
  image_size=(img_height, img_width),
  batch_size=batch_size)



val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=122,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Define data augmentation parameters
data_augmentation = keras.Sequential(
  [
        keras.layers.RandomRotation(factor=0.15),
        keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        keras.layers.RandomFlip(),
        keras.layers.RandomContrast(factor=0.1),

  ]
)

# Apply data augmentation to the dataset
num_classes = len(class_names)

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), tf.keras.utils.to_categorical(y, num_classes)))
val_ds = val_ds.map(lambda x, y: (x, tf.keras.utils.to_categorical(y, num_classes)))

# def unfreeze_model(model1):
#     # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
#     for layer in model1.layers[-20:]:
#         if not isinstance(layer, keras.layers.BatchNormalization):
#             layer.trainable = False

#unfreeze_model = model1

def build_model(num_classes):
    inputs = keras.layers.Input(shape=(224, 224, 3))
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = keras.layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = keras.layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = keras.layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model

model = build_model(num_classes=num_classes)


#Creating an early stop
#Reference used https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
implement_early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=3,
    verbose=1,
    mode='auto',
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0
)

hist = model.fit(train_ds, epochs=20, validation_data=val_ds, callbacks=[implement_early_stop])

# Save the best model after training
model.save("B0_best_trained_scrubbedset_model.keras")
