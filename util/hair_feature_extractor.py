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

    # 1. Convert image to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Perform black-hat filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)

    # 3. Threshold the black-hat image
    _, hair_mask = cv2.threshold(blackhat, 15, 255, cv2.THRESH_BINARY)

    # 4. Quantify hair coverage
    total_pixels = image.shape[0] * image.shape[1]
    hair_pixels = cv2.countNonZero(hair_mask)
    hair_coverage_percentage = (hair_pixels / total_pixels) * 100

    # 5. Classify hair amount
    hair_level = 0
    if hair_coverage_percentage >= high_threshold_pct:
        hair_level = 2
    elif hair_coverage_percentage >= low_threshold_pct:
        hair_level = 1
    else:
        hair_level = 0

    return {'hair_level': hair_level, 'hair_coverage_pct': hair_coverage_percentage}
        hair_level = 0  # Low or no hair (like in img_1443.png)

    return {'hair_level': hair_level, 'hair_coverage_pct': hair_coverage_percentage}
