import os
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input


#I used the following to assist in making my code:  https://stackoverflow.com/questions/65535101/how-to-make-a-prediction-on-this-trained-model
# Load the trained model
model_path = r'Large_cropped_Step_3_trained_model.keras'
model = load_model(model_path)

# Directory where the test images are stored
images_dir = r'data\Test_2\test'

# Load category mapping
category_df = pd.read_csv(r'C:\Users\Devon Scheg\Documents\Academics\Classes\ECE 500\Assignments\MiniProject\category.csv')
#This is standard to the csv file
category_to_name = dict(zip(category_df['Category Number'], category_df['Category Name']))

# List of test image filenames
test_image_filenames = sorted(os.listdir(images_dir), key=lambda x: int(os.path.splitext(x)[0]))

height = 224
width = 224

# Predict and map predictions to category names
predicted_labels = []


for i, filename in enumerate(test_image_filenames):
    print(f"Processing image {i+1}/{len(test_image_filenames)}: {filename}")
    #preparation of the image
    img = image.load_img(os.path.join(images_dir, filename), target_size=(height, width)) #model was trainied on 224, 224
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #model was trained on RGB
    img_array = image.img_to_array(img) #We need to flatten our image into an array
    img_array_expanded_dims = np.expand_dims(img_array, axis=0) #The model predicts the input to have dimensions (batch_size, width, height, bands) 

    #now making the prediciton phase
    prediction = model.predict(img_array_expanded_dims, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_labels.append(category_to_name.get(predicted_class, "Unknown"))

# Process the filenames to remove the .jpg extension
ids = [os.path.splitext(filename)[0] for filename in test_image_filenames]

# Create a DataFrame for submission with the new column headers
submission_df = pd.DataFrame({
    'Id': ids,
    'Category': predicted_labels
})
# Save the DataFrame to a CSV file
submission_csv_path = r'C:\Users\Devon Scheg\Documents\Academics\Classes\ECE 500\Assignments\MiniProject\data\submission_14.csv'
submission_df.to_csv(submission_csv_path, index=False)

print(f"Submission file saved to {submission_csv_path}")
