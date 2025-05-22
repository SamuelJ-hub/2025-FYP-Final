import cv2
import numpy as np
from math import pi
from skimage.transform import rotate # Ensure skimage.transform.rotate is imported
from skimage import morphology


# --- Helper Functions ---
def crop(mask):
    mid = find_midpoint_v4(mask)
    y_nonzero, x_nonzero = np.nonzero(mask)
    y_lims = [np.min(y_nonzero), np.max(y_nonzero)]
    x_lims = np.array([np.min(x_nonzero), np.max(x_nonzero)])
    x_dist = max(np.abs(x_lims - mid))
    x_lims = [mid - x_dist, mid + x_dist]
    # Ensure x_lims are within mask bounds
    x_start = max(0, int(x_lims[0]))
    x_end = min(mask.shape[1], int(x_lims[1]))
    y_start = max(0, int(y_lims[0]))
    y_end = min(mask.shape[0], int(y_lims[1]))
    return mask[y_start:y_end, x_start:x_end]


def find_midpoint_v4(mask):
    summed = np.sum(mask, axis=0)
    half_sum = np.sum(summed) / 2
    # Handle empty mask case
    if np.sum(summed) == 0:
        return 0
    for i, n in enumerate(np.add.accumulate(summed)):
        if n > half_sum:
            return i
    return 0 # Fallback in case loop finishes without returning


# --- Asymmetry Function ---
def get_asymmetry(mask):
    """
    Calculates rotational asymmetry based on the provided method.
    It rotates the mask, crops it, and compares it to its flipped version.
    Returns a dictionary for consistency.
    """
    scores = []
    # Create a copy of the mask to avoid modifying the original during rotation
    current_mask_copy = mask.copy()
    for _ in range(6):
        segment = crop(current_mask_copy) # Use the rotated mask copy
        
        # Ensure segment is not empty before calculating sum and xor
        if np.sum(segment) == 0:
            scores.append(np.nan) # Append NaN if segment is empty
            current_mask_copy = rotate(current_mask_copy, 30) # Rotate the copy for next iteration
            continue

        # Flipping horizontally (axis=1) is common for left-right asymmetry.
        # If you want vertical, change axis=0.
        xor_diff = np.sum(np.logical_xor(segment, np.flip(segment, axis=1)))
        score = xor_diff / np.sum(segment)
        scores.append(score)
        current_mask_copy = rotate(current_mask_copy, 30) # Rotate the copy for next iteration

    # Filter out NaNs if any segments were empty
    valid_scores = [s for s in scores if not np.isnan(s)]
    if valid_scores:
        return {'rotational_asymmetry_score': sum(valid_scores) / len(valid_scores)}
    else:
        return {'rotational_asymmetry_score': np.nan} # Return NaN if all segments were empty

# --- Border (Compactness) Feature ---
def compactness_score(mask):
    """
    Calculates the compactness score of the lesion from its mask.
    A score closer to 0 indicates a perfect circle (less irregular/more compact).
    """
    A = np.sum(mask) # Area is sum of pixels in binary mask

    if A == 0: # Handle empty mask
        return np.nan

    # Use morphology.binary_erosion to estimate perimeter
    struct_el = morphology.disk(2) # Disk of radius 2 pixels
    mask_eroded = morphology.binary_erosion(mask, struct_el)
    perimeter_pixels = np.sum(mask - mask_eroded) # Pixels that are on the border

    if perimeter_pixels == 0: # Handle case of 1x1 mask or similar where erosion removes all
        return np.nan

    # The formula (4*pi*A)/(l**2) gives 1 for a perfect circle.
    # Your function returns 1 - compactness, which gives 0 for perfect circle, and higher for irregularity.
    compactness = (4 * pi * A) / (perimeter_pixels ** 2)
    score = round(1 - compactness, 3) # Score higher for irregularity

    return score # Return this as the feature value


def get_color_features(image, mask):
    """
    Extracts mean and standard deviation of BGR color channels from the masked lesion area.
    """
    binary_mask = (mask > 0).astype(np.uint8)
    lesion_pixels = image[binary_mask == 1]

    if lesion_pixels.size == 0:
        return {'mean_color_B': np.nan, 'mean_color_G': np.nan, 'mean_color_R': np.nan,
                'std_color_B': np.nan, 'std_color_G': np.nan, 'std_color_R': np.nan}

    mean_b, mean_g, mean_r = np.mean(lesion_pixels, axis=0)
    std_b, std_g, std_r = np.std(lesion_pixels, axis=0)

    features = {
        'mean_color_B': mean_b,
        'mean_color_G': mean_g,
        'mean_color_R': mean_r,
        'std_color_B': std_b,
        'std_color_G': std_g,
        'std_color_R': std_r
    }
    return features


# --- Combined ABC Feature Extraction ---
def extract_all_abc_features(image, mask):
    """
    Combines all selected ABC feature extraction functions into one.
    """
    all_features = {}
    
    # Asymmetry
    # Pass a copy of the mask because get_asymmetry modifies it internally by rotating
    asymmetry_feats = get_asymmetry(mask.copy())
    all_features.update(asymmetry_feats)

    # Border - using compactness_score
    # Pass the original mask to get compactness
    border_feats = {'compactness_score': compactness_score(mask)} # Store it directly as a dict
    all_features.update(border_feats)

    # Color
    color_feats = get_color_features(image, mask)
    all_features.update(color_feats)
    
    return all_features


# image = cv2.imread(r'/Users/samuel/Desktop/ITU/Project in Data Science/final_lesion_data/images/PAT_8_15_820.png')
# mask = cv2.imread(r'/Users/samuel/Desktop/ITU/Project in Data Science/final_lesion_data/masks/PAT_8_15_820_mask.png', cv2.IMREAD_GRAYSCALE)

# mask = (mask > 0).astype(np.uint8)

# features = extract_all_abc_features(image, mask)
# print(features)


