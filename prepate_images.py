import os
import shutil

def process_images(target_dir):
    # Walk through the directory
    for root, _, files in os.walk(target_dir):
        for file in files:
            # Check if the file is an image (add more extensions if needed)
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                file_path = os.path.join(root, file)
                base_name, extension = os.path.splitext(file)
                target_folder = os.path.join(root, base_name)
                
                # Create the target folder if it doesn't exist
                os.makedirs(target_folder, exist_ok=True)
                
                # Move and rename the image
                new_file_path = os.path.join(target_folder, f'original{extension}')
                shutil.move(file_path, new_file_path)
                
                print(f'Moved {file_path} to {new_file_path}')

if __name__ == "__main__":
    #target_directory = "C:/Users/ontologist3/daikan-private/image-segmentation/noNoise_0604"
    target_directory = "C:/Users/ontologist3/daikan-private/image-segmentation/remove_difficult_0607"
    process_images(target_directory)
    print("Processing completed.")
