import cv2
import shutil 
import os
#Important for test folder
#This will be my preprocessing face code:
#Thank you to the online cascadeclassifer help and also this user tutorial https://www.youtube.com/watch?v=kwKfWBb6frs&list=LL&index=1
# Path to the directory containing the class folders
data_dir = r'C:\Users\Devon Scheg\Documents\Academics\Classes\ECE 500\Assignments\MiniProject\data\train'

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def get_cropped_image_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x,y,w,h) in faces:
        roi_gray  = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return img[y:y+h, x:x+w]

# Path to the directory where cropped images will be saved
path_to_cr_data = r'C:\Users\Devon Scheg\Documents\Academics\Classes\ECE 500\Assignments\MiniProject\data\cropped_try2'

# Initialize count for total cropped faces
count = 0
#I used the following to help with finding faces: https://github.com/codebasics/py/blob/master/DataScience/CelebrityFaceRecognition/model/data_cleaning.ipynb
# Iterate through the directories containing images
for entry in os.scandir(data_dir):
    if entry.is_dir():
        img_dir = entry.path
        celebrity_number = os.path.basename(img_dir)
        print(celebrity_number)
        
        # Iterate through the images in the current directory
        for entry in os.scandir(img_dir):
            roi_color = get_cropped_image_if_2_eyes(entry.path)   
            if roi_color is not None:
                # Create the cropped folder if it does not exist
                cropped_folder = os.path.join(path_to_cr_data, celebrity_number)
                if not os.path.exists(cropped_folder):
                    os.makedirs(cropped_folder)
                    print("Generating cropped images in folder:", cropped_folder)
                
                # Save the cropped image
                cropped_file_name = f"{os.path.splitext(os.path.basename(entry.path))[0]}.jpg"
                cropped_file_path = os.path.join(cropped_folder,cropped_file_name)
                cv2.imwrite(cropped_file_path, roi_color) 
                
                # Increment the count for total cropped faces
                count += 1 

print(f'Finished with finding faces, total count: {count}')
