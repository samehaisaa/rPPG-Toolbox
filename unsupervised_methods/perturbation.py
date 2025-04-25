"""
Perturbation methods for uncertainty estimation in rPPG signal processing.
This module contains functions for applying various perturbations to video inputs
to assess the robustness and uncertainty of rPPG algorithms.
"""

import numpy as np
import cv2
from scipy import ndimage


def add_gaussian_noise(frames, noise_std_fraction=0.01):
    """
    Add Gaussian noise to video frames.
    
    Args:
        frames: Video frames array of shape (T, H, W, C)
        noise_std_fraction: Standard deviation of noise as fraction of signal magnitude
    
    Returns:
        Perturbed frames with added Gaussian noise
    """
    # Calculate signal magnitude
    signal_magnitude = np.mean(np.abs(frames))
    noise_std = noise_std_fraction * signal_magnitude
    
    # Generate and add noise
    noise = np.random.normal(0, noise_std, frames.shape)
    return frames + noise


def apply_blur(frames, kernel_size=3):
    """
    Apply Gaussian blur to video frames.
    
    Args:
        frames: Video frames array of shape (T, H, W, C)
        kernel_size: Size of the Gaussian kernel (odd number)
    
    Returns:
        Blurred frames
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    blurred_frames = np.zeros_like(frames)
    for i in range(len(frames)):
        blurred_frames[i] = cv2.GaussianBlur(frames[i], (kernel_size, kernel_size), 0)
    
    return blurred_frames


def apply_brightness_change(frames, brightness_factor=0.1):
    """
    Apply brightness variation to video frames.
    
    Args:
        frames: Video frames array of shape (T, H, W, C)
        brightness_factor: Factor by which to adjust brightness (positive or negative)
    
    Returns:
        Frames with adjusted brightness
    """
    # Calculate adjustment value based on frame intensity
    mean_intensity = np.mean(frames)
    adjustment = mean_intensity * brightness_factor
    
    # Apply brightness adjustment, ensuring values stay in valid range
    adjusted_frames = frames + adjustment
    adjusted_frames = np.clip(adjusted_frames, 0, 255)
    
    return adjusted_frames


def apply_random_crop(frames, crop_fraction=0.1):
    """
    Apply random cropping and resize back to original size.
    
    Args:
        frames: Video frames array of shape (T, H, W, C)
        crop_fraction: Fraction of width/height to crop (0-1)
    
    Returns:
        Randomly cropped and resized frames
    """
    T, H, W, C = frames.shape
    
    # Calculate crop dimensions
    crop_h = int(H * (1 - crop_fraction))
    crop_w = int(W * (1 - crop_fraction))
    
    # Random crop offsets
    top = np.random.randint(0, H - crop_h + 1)
    left = np.random.randint(0, W - crop_w + 1)
    
    # Crop and resize
    cropped_frames = np.zeros_like(frames)
    for i in range(T):
        # Crop frame
        crop = frames[i, top:top+crop_h, left:left+crop_w]
        # Resize back to original dimensions
        cropped_frames[i] = cv2.resize(crop, (W, H))
    
    return cropped_frames


def apply_rotation(frames, angle_range=5):
    """
    Apply random rotation to video frames.
    
    Args:
        frames: Video frames array of shape (T, H, W, C)
        angle_range: Maximum rotation angle in degrees (+/-)
    
    Returns:
        Rotated frames
    """
    # Generate random angle within range
    angle = np.random.uniform(-angle_range, angle_range)
    
    rotated_frames = np.zeros_like(frames)
    for i in range(len(frames)):
        # Apply rotation using scipy
        for c in range(frames.shape[3]):  # Process each channel
            rotated_frames[i, :, :, c] = ndimage.rotate(
                frames[i, :, :, c], 
                angle, 
                reshape=False, 
                mode='nearest'
            )
    
    return rotated_frames


def apply_color_jitter(frames, factor=0.1):
    """
    Apply random color channel perturbation to video frames.
    
    Args:
        frames: Video frames array of shape (T, H, W, C)
        factor: Maximum color channel adjustment factor (0-1)
    
    Returns:
        Color jittered frames
    """
    # Generate random factors for each color channel
    r_factor = 1.0 + np.random.uniform(-factor, factor)
    g_factor = 1.0 + np.random.uniform(-factor, factor)
    b_factor = 1.0 + np.random.uniform(-factor, factor)
    
    perturbed_frames = frames.copy()
    
    # Apply color jitter
    if frames.shape[3] >= 3:  # Ensure we have at least 3 channels (RGB)
        perturbed_frames[:, :, :, 0] = np.clip(frames[:, :, :, 0] * r_factor, 0, 255)
        perturbed_frames[:, :, :, 1] = np.clip(frames[:, :, :, 1] * g_factor, 0, 255)
        perturbed_frames[:, :, :, 2] = np.clip(frames[:, :, :, 2] * b_factor, 0, 255)
    
    return perturbed_frames


def apply_compression_artifacts(frames, quality=80):
    """
    Apply JPEG compression artifacts to frames.
    
    Args:
        frames: Video frames array of shape (T, H, W, C)
        quality: JPEG compression quality (0-100, lower means more artifacts)
    
    Returns:
        Frames with compression artifacts
    """
    compressed_frames = np.zeros_like(frames)
    
    # Define JPEG compression parameters
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    
    for i in range(len(frames)):
        # Convert to uint8 if not already
        frame_uint8 = frames[i].astype(np.uint8)
        
        # Compress and decompress
        _, encoded_img = cv2.imencode('.jpg', frame_uint8, encode_param)
        compressed_frames[i] = cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)
    
    return compressed_frames


def get_perturbation_function(perturbation_type):
    """
    Get the perturbation function based on the specified type.
    
    Args:
        perturbation_type: String identifier for the perturbation type
    
    Returns:
        Function that applies the specified perturbation
    """
    perturbation_functions = {
        'gaussian_noise': add_gaussian_noise,
        'blur': apply_blur,
        'brightness': apply_brightness_change,
        'crop': apply_random_crop,
        'rotation': apply_rotation,
        'color_jitter': apply_color_jitter,
        'compression': apply_compression_artifacts
    }
    
    if perturbation_type in perturbation_functions:
        return perturbation_functions[perturbation_type]
    else:
        raise ValueError(f"Unknown perturbation type: {perturbation_type}. "
                        f"Available types: {list(perturbation_functions.keys())}") 
