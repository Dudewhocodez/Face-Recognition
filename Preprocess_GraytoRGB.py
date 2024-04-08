import cv2
from pathlib import Path

#I used this link for some referencing https://stackoverflow.com/questions/21596281/how-does-one-convert-a-grayscale-image-to-rgb-in-opencv-python
# Function to check if an image is in RGB format
def is_rgb(image_path):
    # Read the image
    img = cv2.imread(str(image_path))
    
    # Check if the image is not None and has 3 channels
    return img is not None and img.ndim == 3 and img.shape[2] == 3

# Function to convert grayscale image to RGB
def grayscale_to_rgb(image_path):
    # Read the grayscale image
    grayscale_img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    # Check if the image was loaded successfully
    if grayscale_img is None:
        print(f"Error: Unable to load image '{image_path}'")
        return None
    
    # Convert grayscale image to RGB
    rgb_img = cv2.cvtColor(grayscale_img, cv2.COLOR_GRAY2RGB)
    
    return rgb_img

# Path to the directory containing images
data_dir = Path(r'data\test')
# Count the total number of images processed
total_images_processed = 0
# Go through each subdirectory and convert grayscale images to RGB
for subdir in data_dir.iterdir():
    if subdir.is_dir():
        for image_path in subdir.glob('*.jpg'):
            #check if it is already RGB
            if is_rgb(image_path):
                print(f"Skipping image '{image_path}' it is RGB format")
                continue
            #convert grayscale to rgb
            rgb_img = grayscale_to_rgb(image_path)
            if rgb_img is not None:
                # Overwrite the grayscale image with the RGB version
                cv2.imwrite(str(image_path), rgb_img)
                print(f"Converted and saved: {image_path}")
                total_images_processed += 1


# Print completion message
print(f"All images processed. Total images processed: {total_images_processed}")

print("\nAll images loaded successfully.")
