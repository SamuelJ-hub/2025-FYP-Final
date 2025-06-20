# This script provides utility functions for common image and mask
# operations like reading, saving, and a basic hair removal attempt.

import cv2

# Helper functions for reading images/masks
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

def removeHair(img_org, img_gray, kernel_size=25, threshold=10, radius=3):
    """
    Attempts to remove hair from an image using black-hat filtering and inpainting.

    Args:
        img_org (numpy.ndarray): The original color image (BGR).
        img_gray (numpy.ndarray): The grayscale version of the original image.
        kernel_size (int, optional): Size for the morphological kernel. Defaults to 25.
        threshold (int, optional): Threshold value for hair contour intensification. Defaults to 10.
        radius (int, optional): Radius for the inpainting algorithm. Defaults to 3.

    Returns:
        tuple: (blackhat_image, thresholded_image, inpainted_image)
               The black-hat result, the thresholded hair mask, and the image with hair inpainted.
    """
    # kernel for the morphological filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))

    # perform the blackHat filtering on the grayscale image to find the hair countours
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)

    # intensify the hair countours in preparation for the inpainting algorithm
    _, thresh = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)

    # inpaint the original image depending on the mask
    img_out = cv2.inpaint(img_org, thresh, radius, cv2.INPAINT_TELEA)

    return blackhat, thresh, img_out
