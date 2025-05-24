import cv2
import numpy as np

def analyze_hair_amount(image, low_threshold_pct=0.5, high_threshold_pct=2.5):
    """
    Analyzes the amount of hair in an image and classifies it into three levels.

    This function uses a black-hat morphological operation to detect dark, hair-like
    structures. It calculates the percentage of the image covered by these structures
    and classifies it into a level from 0 to 2.

    Args:
        image (np.ndarray): The input image in BGR format (as loaded by OpenCV).
        low_threshold_pct (float): The percentage of hair coverage below which is
                                   classified as 'low hair' (level 0). Default is 0.5%.
        high_threshold_pct (float): The percentage of hair coverage above which is
                                    classified as 'high hair' (level 2). Default is 2.5%.

    Returns:
        dict: A dictionary containing the hair level (0, 1, or 2) and the raw
              hair coverage percentage. e.g., {'hair_level': 1, 'hair_coverage_pct': 1.7}
    """
    if image is None:
        return {'hair_level': np.nan, 'hair_coverage_pct': np.nan}

    # 1. Convert image to grayscale, which is required for the black-hat operation.
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Perform the black-hat filtering to create a mask of dark, hair-like structures.
    #    A kernel size of 25x25 is chosen as it's larger than the typical hair width.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)

    # 3. Threshold the black-hat image to create a clean binary mask of the hair.
    #    A value of 15 is a robust starting point for dermoscopic images.
    _, hair_mask = cv2.threshold(blackhat, 15, 255, cv2.THRESH_BINARY)

    # 4. Quantify the hair by calculating its percentage coverage of the total image area.
    #    This makes the metric robust to images of different resolutions.
    total_pixels = image.shape[0] * image.shape[1]
    hair_pixels = cv2.countNonZero(hair_mask)
    hair_coverage_percentage = (hair_pixels / total_pixels) * 100

    # 5. Classify the hair amount into the three levels based on the thresholds.
    hair_level = 0
    if hair_coverage_percentage >= high_threshold_pct:
        hair_level = 2  # A lot of hair (like in img_1453.png)
    elif hair_coverage_percentage >= low_threshold_pct:
        hair_level = 1  # Some hair (like in img_1487.png)
    else:
        hair_level = 0  # Low or no hair (like in img_1443.png)

    return {'hair_level': hair_level, 'hair_coverage_pct': hair_coverage_percentage}