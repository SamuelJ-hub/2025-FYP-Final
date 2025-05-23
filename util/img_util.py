import cv2
import random
import os

class ImageDataLoader:
    def __init__(self, directory, file_extension=".png", suffix="", shuffle=False, transform=None):
        self.directory = directory
        self.file_extension = file_extension
        self.suffix = suffix # Useful to distinguish images and masks (e.g., "_mask")
        self.shuffle = shuffle
        self.transform = transform # For future transformations (e.g., data augmentation)

        self.file_list = self._get_image_files()

        if not self.file_list:
            # Do not raise error here, it will be handled by Data_Loader.
            # If the directory is empty or contains no files, ImageDataLoader will be empty.
            print(f"Warning: No files found with extension '{file_extension}' and suffix '{suffix}' in directory: {directory}")

        if self.shuffle:
            random.shuffle(self.file_list)

        self.current_index = 0

    def _get_image_files(self):
        """
        Populates self.file_list with the full paths of image/mask files
        that match the specified extension and suffix.
        """
        found_files = []
        if os.path.isdir(self.directory):
            for filename in os.listdir(self.directory):
                # Check extension and suffix
                if filename.endswith(self.file_extension) and filename.endswith(self.suffix + self.file_extension):
                    full_path = os.path.join(self.directory, filename)
                    if os.path.isfile(full_path): # Ensure it's a file
                        found_files.append(full_path)
        return sorted(found_files) # Sort for reproducibility if not shuffled

    def __len__(self):
        return len(self.file_list)

    def __iter__(self):
        # Reset index for a new iteration cycle
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index < len(self.file_list):
            file_path = self.file_list[self.current_index]
            self.current_index += 1
            
            # Here you may want to read the file directly, or just return the path
            # For your Data_Loader, only the paths are needed, since Data_Loader
            # already checks the dataframe.
            return file_path
        else:
            raise StopIteration

# Utility functions for reading images/masks (still useful)
def read_image_bgr(file_path):
    """
    Reads an image file into BGR format using OpenCV.
    Returns None if the image cannot be loaded.
    """
    img_bgr = cv2.imread(file_path)
    return img_bgr

def read_mask_grayscale(file_path):
    """
    Reads a mask file in grayscale format using OpenCV.
    Returns None if the mask cannot be loaded.
    """
    mask_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return mask_gray

def save_image_file(img_rgb, file_path):
    """
    Saves an RGB image to a file in BGR format (for OpenCV compatibility).
    """
    try:
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        success = cv2.imwrite(file_path, img_bgr)
        if not success:
            print(f"Failed to save the image to {file_path}")
        return success
    except Exception as e:
        print(f"Error saving the image: {e}")
        return False