# util/features_extractor.py

import numpy as np
from math import pi
from skimage.transform import rotate
from skimage import morphology
# Import your existing helper functions if they are still in this file
# For example, if crop and find_midpoint_v4 are here:
# from .your_helper_module import crop, find_midpoint_v4 # Or define them here

# --- NEW: Import the hair analysis function ---
from util.hair_feature_extractor import analyze_hair_amount


# --- Helper Functions (keep them here or move to another util file if preferred) ---
def crop(mask):
    # Make sure this function is robust, e.g., handles empty nonzeros
    y_nonzero, x_nonzero = np.nonzero(mask)
    if y_nonzero.size == 0 or x_nonzero.size == 0: # Handle empty mask after rotation/cropping
        return np.array([[]], dtype=mask.dtype) # Return an empty array of same type

    mid = find_midpoint_v4(mask)
    y_lims = [np.min(y_nonzero), np.max(y_nonzero)]
    x_lims_arr = np.array([np.min(x_nonzero), np.max(x_nonzero)]) # Renamed to avoid conflict
    x_dist = max(np.abs(x_lims_arr - mid))
    x_lims_calc = [mid - x_dist, mid + x_dist] # Renamed
    x_start = max(0, int(x_lims_calc[0]))
    x_end = min(mask.shape[1], int(x_lims_calc[1]))
    y_start = max(0, int(y_lims[0]))
    y_end = min(mask.shape[0], int(y_lims[1]))

    # Ensure indices are valid before slicing
    if y_start >= y_end or x_start >= x_end:
        return np.array([[]], dtype=mask.dtype)
    return mask[y_start:y_end, x_start:x_end]


def find_midpoint_v4(mask):
    summed = np.sum(mask, axis=0)
    total_sum = np.sum(summed)
    if total_sum == 0:
        return mask.shape[1] // 2 # Return center if mask is empty
    half_sum = total_sum / 2
    for i, n in enumerate(np.add.accumulate(summed)):
        if n >= half_sum: # Changed to >= to ensure it always finds a point
            return i
    return mask.shape[1] -1 # Fallback: return last index if not found (shouldn't happen if total_sum > 0)


# --- Asymmetry Function ---
def get_asymmetry(mask):
    scores = []
    current_mask_copy = mask.astype(np.uint8).copy() # Ensure mask is uint8 for rotate
    for _ in range(6):
        segment = crop(current_mask_copy)
        if segment.size == 0 or np.sum(segment) == 0:
            scores.append(np.nan)
            current_mask_copy = rotate(current_mask_copy, 30, resize=False, preserve_range=True, mode='constant', cval=0).astype(np.uint8)
            continue
        xor_diff = np.sum(np.logical_xor(segment, np.flip(segment, axis=1)))
        score = xor_diff / np.sum(segment)
        scores.append(score)
        current_mask_copy = rotate(current_mask_copy, 30, resize=False, preserve_range=True, mode='constant', cval=0).astype(np.uint8)

    valid_scores = [s for s in scores if not np.isnan(s)]
    if valid_scores:
        return {'rotational_asymmetry_score': sum(valid_scores) / len(valid_scores)}
    else:
        return {'rotational_asymmetry_score': np.nan}

# --- Border (Compactness) Feature ---
def compactness_score(mask):
    A = np.sum(mask)
    if A == 0: return np.nan
    struct_el = morphology.disk(2)
    mask_eroded = morphology.binary_erosion(mask, struct_el)
    perimeter_pixels = np.sum(mask - mask_eroded)
    if perimeter_pixels == 0: return np.nan # or a value indicating extreme compactness/small size
    
    # Polsby-Popper compactness: (4*pi*A)/(P^2). 1 for circle, <1 for others.
    # To make higher values mean more irregular, you can use 1/compactness or (P^2)/(4*pi*A)
    # Your original formula was 1 - compactness_value, which gives 0 for circle.
    # Let's stick to a common definition where higher is more irregular.
    # Compactness (Polsby-Popper) = (4 * pi * A) / (perimeter_pixels ** 2)
    # Irregularity = 1 / Compactness (Polsby-Popper) or (perimeter_pixels ** 2) / (4 * pi * A)
    # The provided code was: score = round(1 - compactness_value, 3)
    # This means score is 0 for circle, higher for irregular. Let's keep it for consistency with your previous features.
    compactness_value = (4 * pi * A) / (perimeter_pixels ** 2)
    score = round(1 - compactness_value, 3)
    return score


# --- Color Features ---
def get_color_features(image, mask):
    binary_mask_bool = (mask > 0) # Ensure mask is boolean for indexing
    lesion_pixels = image[binary_mask_bool]

    if lesion_pixels.size == 0:
        return {'mean_color_B': np.nan, 'mean_color_G': np.nan, 'mean_color_R': np.nan,
                'std_color_B': np.nan, 'std_color_G': np.nan, 'std_color_R': np.nan}

    # OpenCV loads images as BGR, so means/stds will be in that order
    mean_colors = np.mean(lesion_pixels, axis=0)
    std_colors = np.std(lesion_pixels, axis=0)

    features = {
        'mean_color_B': mean_colors[0],
        'mean_color_G': mean_colors[1],
        'mean_color_R': mean_colors[2],
        'std_color_B': std_colors[0],
        'std_color_G': std_colors[1],
        'std_color_R': std_colors[2]
    }
    return features


# --- Combined Feature Extraction ---
def extract_all_features(image, mask): # Renamed to reflect it does more than ABC
    """
    Combines all feature extraction functions into one master function.
    Ensures mask is binary (0 or 1) for feature functions.
    """
    all_features = {}

    # Ensure mask is binary (0 or 1) for consistency if not already
    binary_mask_01 = (mask > 0).astype(np.uint8)

    # Asymmetry
    all_features.update(get_asymmetry(binary_mask_01.copy())) # Pass copy as it's modified

    # Border
    all_features.update({'compactness_score': compactness_score(binary_mask_01)})

    # Color
    all_features.update(get_color_features(image, binary_mask_01)) # image, and the 0/1 mask

    # --- Hair Feature (Imported) ---
    all_features.update(analyze_hair_amount(image)) # Pass the original BGR image

    return all_features
