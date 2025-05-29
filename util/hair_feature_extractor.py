# This script defines a function to analyze the amount of hair in an image
# and classify it into levels based on coverage percentage.
import cv2
import numpy as np

def analyze_hair_amount(image, low_threshold_pct=3, high_threshold_pct=10):
    """
    Analyzes an image to estimate hair presence and categorizes it.

    It converts the image to grayscale, applies black-hat filtering to highlight
    hair-like structures, thresholds the result, and then calculates the
    percentage of hair coverage to assign a hair level (0, 1, or 2).

    Args:
        image (numpy.ndarray): The input color image (BGR format).
        low_threshold_pct (float, optional): Lower percentage threshold for 'some hair' (level 1). Defaults to 3.
        high_threshold_pct (float, optional): Higher percentage threshold for 'a lot of hair' (level 2). Defaults to 10.

    Returns:
        dict: A dictionary with 'hair_level' (0, 1, or 2) and 
              'hair_coverage_pct' (float percentage), or NaNs if input is invalid.
    """
    if image is None:
        return {'hair_level': np.nan, 'hair_coverage_pct': np.nan}

    # 1. Convert image to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Perform black-hat filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)

    # 3. Threshold the black-hat image
    _, hair_mask = cv2.threshold(blackhat, 30, 255, cv2.THRESH_BINARY)

    # 4. Quantify hair coverage
    total_pixels = image.shape[0] * image.shape[1]
    hair_pixels = cv2.countNonZero(hair_mask)
    if total_pixels == 0:
        hair_coverage_percentage = np.nan
    else:
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