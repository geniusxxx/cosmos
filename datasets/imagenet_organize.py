import os
import shutil

# Define the source directory
source_dir = "/your/directory/to/imagenet/data/val_images"

# Check if the source directory exists
if not os.path.exists(source_dir):
    print(f"Source directory {source_dir} does not exist.")
    exit()

# Get a list of all JPEG files in the source directory
jpeg_files = [f for f in os.listdir(source_dir) if f.endswith('.JPEG')]

# Process each JPEG file
for jpeg_file in jpeg_files:
    # Extract the class name from the file name
    # Example) ILSVRC2012_val_00012508_n01843065.JPEG => n01843065
    class_name = jpeg_file.split('_')[-1].split('.')[0]
    
    # Define the target directory for this class
    target_dir = os.path.join(source_dir, class_name)
    
    # Create the target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Define the source and target file paths
    source_file = os.path.join(source_dir, jpeg_file)
    target_file = os.path.join(target_dir, jpeg_file)
    
    # Move the file to the target directory
    shutil.move(source_file, target_file)

print("Files have been classified and moved to their respective directories.")