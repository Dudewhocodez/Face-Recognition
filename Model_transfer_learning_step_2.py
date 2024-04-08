#Model 3 extension and find tuning
#Define the model that is to be fine tuned
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from keras.applications import EfficientNetB0
from tensorflow import keras
from pathlib import Path
from keras.models import clone_model

#I used this online resource for help on unfreezing my pretrained efficientnet
#https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/


data_dir = Path(r'C:\Users\Devon Scheg\Documents\Academics\Classes\ECE 500\Assignments\MiniProject\data\cropped')
model = tf.keras.models.load_model("B0_best_trained_scrubbedset_model.keras")


# Specify image params
batch_size = 32
img_height = 224
img_width = 224

#Fine tuning model
def unfreeze_model(model):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = True

    optimizer = keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )


unfreeze_model(model)


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
        keras.layers.RandomFlip("horizontal",
                                input_shape=(img_height, img_width, 3)),
        keras.layers.RandomRotation(0.3),
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomContrast(0.4),  # Adding RandomContrast augmentation
        keras.layers.RandomBrightness(0.2),  # Adding RandomBrightness augmentation
        keras.layers.RandomTranslation(0.15, 0.2),  # Adding RandomTranslation augmentation
  ]
)

# Apply data augmentation to the dataset
num_classes = len(class_names)

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), tf.keras.utils.to_categorical(y, num_classes)))
val_ds = val_ds.map(lambda x, y: (x, tf.keras.utils.to_categorical(y, num_classes)))


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


model.save("B0_best_trained__fine_tuned_scrubbedset_model.keras")
