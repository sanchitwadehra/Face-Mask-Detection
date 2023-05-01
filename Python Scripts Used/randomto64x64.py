import os
from PIL import Image

# Set the input and output folders
input_folder = "C:/Users/WADEHRA/Downloads/Face Mask/Code/data/not standardized/with_mask"
output_folder = "C:/Users/WADEHRA/Downloads/Face Mask/Code/data/standardized/with_mask"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop over all files in the input folder
for file_name in os.listdir(input_folder):
    # Check if the file is an image
    if file_name.endswith(".jpg") or file_name.endswith(".jpeg") or file_name.endswith(".png"):
        # Open the image file
        image_path = os.path.join(input_folder, file_name)
        with Image.open(image_path) as image:
            # Convert the image to RGB mode
            image = image.convert("RGB")
            # Resize the image to 64x64 pixels
            image = image.resize((64, 64))
            # Save the resized image to the output folder
            output_path = os.path.join(output_folder, file_name.replace(".", "_64x64."))
            output_path = output_path.replace(".jpeg", ".jpg")
            image.save(output_path)
            print(f"Saved {output_path}")
    else:
        print(f"{file_name} is not an image file")


