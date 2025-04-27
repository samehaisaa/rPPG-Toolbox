# The Chrominance Method from: De Haan, G., & Jeanne, V. (2013). 
# Robust pulse rate from chrominance-based rPPG. IEEE Transactions on Biomedical Engineering, 60(10), 2878-2886. 
# DOI: 10.1109/TBME.2013.2266196
import numpy as np
import math
from scipy import signal
import unsupervised_methods.utils as utils
from unsupervised_methods.perturbation import get_perturbation_function, add_gaussian_noise
import os
import cv2

def _calculate_bvp_from_rgb(RGB, FS, LPF, HPF, WinSec):
    """Core CHROM BVP calculation logic for a given RGB signal."""
    FN = RGB.shape[0]
    NyquistF = 1/2*FS
    B, A = signal.butter(3, [LPF/NyquistF, HPF/NyquistF], 'bandpass')

    WinL = math.ceil(WinSec*FS)
    if(WinL % 2):
        WinL = WinL+1
    NWin = math.floor((FN-WinL//2)/(WinL//2))
    if NWin <= 0:
        # Handle cases where the signal is too short for the window
        print(f"Warning: Signal length {FN} too short for window processing. Returning zeros.")
        # Estimate total length based on expected overlap if NWin was at least 1
        estimated_totallen = WinL if NWin == 0 else (WinL // 2) * (NWin + 1) 
        return np.zeros(estimated_totallen)
        
    WinS = 0
    WinM = int(WinS+WinL//2)
    WinE = WinS+WinL
    totallen = (WinL//2)*(NWin+1)
    S = np.zeros(totallen)

    for i in range(NWin):
        # Ensure window bounds are within RGB array dimensions
        if WinE > FN:
             print(f"Warning: Window end {WinE} exceeds signal length {FN}. Adjusting.")
             WinE = FN
             # Recalculate WinL based on available data, might need adjustment
             current_WinL = WinE - WinS 
             if current_WinL < 2: # Need at least 2 points for std dev
                 print("Window too small after adjustment. Skipping.")
                 break
             if current_WinL % 2:
                 current_WinL -= 1 # Ensure even window length if possible
                 WinE -=1
             if current_WinL < WinL: # If adjusted window is smaller
                 #Option 1: Skip this window (safer)
                 # break
                 # Option 2: Process with smaller window (might introduce artifacts)
                 hann_win = signal.windows.hann(current_WinL)
                 RGB_win = RGB[WinS:WinE, :]
                 Xs_win = np.squeeze(3*RGB_win[:, 0]-2*RGB_win[:, 1])
                 Ys_win = np.squeeze(1.5*RGB_win[:, 0]+RGB_win[:, 1]-1.5*RGB_win[:, 2])
             else:
                 hann_win = signal.windows.hann(WinL)
                 RGB_win = RGB[WinS:WinE, :]
                 Xs_win = np.squeeze(3*RGB_win[:, 0]-2*RGB_win[:, 1])
                 Ys_win = np.squeeze(1.5*RGB_win[:, 0]+RGB_win[:, 1]-1.5*RGB_win[:, 2])

        else: # Normal case
             hann_win = signal.windows.hann(WinL)
             RGB_win = RGB[WinS:WinE, :]
             Xs_win = np.squeeze(3*RGB_win[:, 0]-2*RGB_win[:, 1])
             Ys_win = np.squeeze(1.5*RGB_win[:, 0]+RGB_win[:, 1]-1.5*RGB_win[:, 2])
             current_WinL = WinL

        RGBBase = np.mean(RGB_win, axis=0)
        RGBNorm = np.zeros_like(RGB_win)
        # Avoid division by zero
        if np.any(RGBBase == 0):
             print(f"Warning: Zero mean RGB value encountered in window {i}. Skipping normalization.")
             RGBNorm = RGB_win # Or handle differently, e.g., add small epsilon
        else:
             for temp in range(current_WinL):
                  RGBNorm[temp] = np.true_divide(RGB_win[temp], RGBBase)
        
        # Use the original window size Xs, Ys for filtering if possible?
        # This part needs careful thought: Use Xs_win, Ys_win for consistency
        Xs = np.squeeze(3*RGBNorm[:, 0]-2*RGBNorm[:, 1])
        Ys = np.squeeze(1.5*RGBNorm[:, 0]+RGBNorm[:, 1]-1.5*RGBNorm[:, 2])

        # Check for sufficient data points for filtering
        if len(Xs) <= 3 * max(len(B), len(A)):
            print(f"Warning: Not enough data points ({len(Xs)}) for filtfilt in window {i}. Skipping filtering.")
            Xf = Xs
            Yf = Ys
        else:
            Xf = signal.filtfilt(B, A, Xs, axis=0)
            Yf = signal.filtfilt(B, A, Ys)
        
        # Avoid division by zero in std calculation
        std_Yf = np.std(Yf)
        if std_Yf == 0:
            print(f"Warning: Zero standard deviation for Yf in window {i}. Setting Alpha to 0.")
            Alpha = 0
        else:
            Alpha = np.std(Xf) / std_Yf
            
        SWin = Xf-Alpha*Yf
        SWin = np.multiply(SWin, hann_win) # Use hann_win corresponding to current_WinL

        # Adjust overlap calculation based on current_WinL
        win_overlap = current_WinL // 2
        if WinS + win_overlap > totallen or WinS + current_WinL > totallen:
             # Adjust slice indices if they exceed the allocated 'S' array size
             end_idx1 = min(WinS + win_overlap, totallen)
             end_idx2 = min(WinS + current_WinL, totallen)
             len1 = end_idx1 - WinS
             len2 = end_idx2 - (WinS + win_overlap)
             S[WinS : end_idx1] += SWin[:len1]
             if len2 > 0:
                 S[WinS + win_overlap : end_idx2] = SWin[len1 : len1+len2]
        else:
             S[WinS:WinS + win_overlap] += SWin[:win_overlap]
             S[WinS + win_overlap : WinS + current_WinL] = SWin[win_overlap:]
        
        # Update window boundaries for next iteration
        WinS += win_overlap
        WinM = WinS + win_overlap # Recalculate WinM
        WinE = WinS + current_WinL # Update WinE based on current window size

        # Break if WinS goes beyond the signal length processed so far in S
        if WinS >= totallen:
            break
            
    # Trim S to actual computed length if shortened by window adjustments
    actual_len = WinS # The last start position marks the end of computed signal
    return S[:actual_len]

def process_video(frames):
    """Process video frames to get RGB signals."""
    return utils.process_video(frames)

def CHROME_DEHAAN(frames, FS, n_perturbations=10, noise_std_fraction=0.01, perturbation_type='gaussian_noise', perturbation_params=None, save_path=None):
    """
    Enhanced CHROM method with uncertainty quantification via input perturbation.
    
    Args:
        frames: Video frames array of shape (T, H, W, C)
        FS: Sampling frequency
        n_perturbations: Number of perturbations to run
        noise_std_fraction: Standard deviation of noise as fraction of signal (used for gaussian_noise)
        perturbation_type: Type of perturbation to apply ('gaussian_noise', 'blur', 'brightness', etc.)
        perturbation_params: Dictionary of parameters for the perturbation function
        save_path: Path to save perturbed video frames. If None, frames are not saved.
        
    Returns:
        Dictionary containing BVP signal, confidence bands, perturbed signals, and frames
    """
    LPF = 0.7
    HPF = 2.5
    WinSec = 1.6

    # Get perturbation function
    try:
        perturbation_func = get_perturbation_function(perturbation_type)
    except ValueError:
        print(f"Warning: Unknown perturbation type '{perturbation_type}'. Falling back to gaussian noise.")
        perturbation_func = add_gaussian_noise
        perturbation_params = {'noise_std_fraction': noise_std_fraction}
    
    # Initialize default parameters if not provided
    if perturbation_params is None:
        if perturbation_type == 'gaussian_noise':
            perturbation_params = {'noise_std_fraction': noise_std_fraction}
        else:
            perturbation_params = {}

    # Process original video
    original_frames = frames.copy()
    RGB_original = process_video(original_frames)
    if RGB_original.size == 0:
        print("Error: process_video returned empty RGB array.")
        return {'BVP': np.array([]), 'confidence_bands': np.array([[], []]), 
                'perturbed_signals': np.array([]), 'perturbed_frames': []}
    
    # Calculate original BVP
    original_bvp = _calculate_bvp_from_rgb(RGB_original, FS, LPF, HPF, WinSec)
    if original_bvp.size == 0:
        print("Warning: Original BVP calculation resulted in empty array.")
        return {'BVP': np.array([]), 'confidence_bands': np.array([[], []]), 
                'perturbed_signals': np.array([]), 'perturbed_frames': []}
    
    perturbed_bvp_signals = [original_bvp]  # Start with original BVP
    perturbed_frames_list = [original_frames]  # Store original frames
    
    # Generate and process perturbed signals
    for i in range(n_perturbations):
        # Apply perturbation to the frames
        perturbed_frames = perturbation_func(original_frames.copy(), **perturbation_params)
        
        # Process perturbed frames
        RGB_perturbed = process_video(perturbed_frames)
        if RGB_perturbed.size == 0:
            print(f"Warning: Perturbed frames processing failed for perturbation {i}. Skipping.")
            continue
        
        # Calculate BVP from perturbed RGB
        bvp_perturbed = _calculate_bvp_from_rgb(RGB_perturbed, FS, LPF, HPF, WinSec)
        if bvp_perturbed.size == 0:
            print(f"Warning: BVP calculation failed for perturbation {i}. Skipping.")
            continue
        
        # Ensure perturbed BVP has same length as original via padding/truncation if needed
        if len(bvp_perturbed) != len(original_bvp):
            print(f"Warning: Perturbed BVP length ({len(bvp_perturbed)}) differs from original ({len(original_bvp)}). Adjusting.")
            # Simple truncation/padding - more sophisticated alignment might be better
            new_bvp = np.zeros_like(original_bvp)
            common_length = min(len(bvp_perturbed), len(original_bvp))
            new_bvp[:common_length] = bvp_perturbed[:common_length]
            bvp_perturbed = new_bvp
            
        perturbed_bvp_signals.append(bvp_perturbed)
        perturbed_frames_list.append(perturbed_frames)
        
    if not perturbed_bvp_signals:
         print("Error: No BVP signals generated.")
         return {'BVP': np.array([]), 'confidence_bands': np.array([[], []]), 
                 'perturbed_signals': np.array([]), 'perturbed_frames': []}
         
    # Calculate mean BVP and confidence bands
    perturbed_bvp_signals = np.array(perturbed_bvp_signals)
    mean_bvp = np.mean(perturbed_bvp_signals, axis=0)
    ci_lower = np.percentile(perturbed_bvp_signals, 2.5, axis=0)
    ci_upper = np.percentile(perturbed_bvp_signals, 97.5, axis=0)
    confidence_bands = np.array([ci_lower, ci_upper])
    
    # Save perturbed frames if path provided
    if save_path:
        try:
            os.makedirs(save_path, exist_ok=True)
            for i, frames in enumerate(perturbed_frames_list):
                # Convert frames to uint8 if not already
                if frames.dtype != np.uint8:
                    frames = (frames * 255).astype(np.uint8)
                
                # Save as video using cv2.VideoWriter
                name = 'original' if i == 0 else f'perturbed_{i}'
                video_path = os.path.join(save_path, f'{name}.avi')
                
                # Get frame dimensions
                T, H, W, C = frames.shape
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(video_path, fourcc, FS, (W, H))
                
                for frame in frames:
                    # Convert to BGR for OpenCV
                    if C == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame)
                out.release()
                
            print(f"Saved {len(perturbed_frames_list)} videos to {save_path}")
        except Exception as e:
            print(f"Error saving perturbed videos: {e}")

    return {
        'BVP': mean_bvp,  # Mean BVP signal from perturbations
        'confidence_bands': confidence_bands,  # Lower and upper confidence bounds
        'perturbed_signals': perturbed_bvp_signals, # Return all generated signals
        'perturbed_frames': perturbed_frames_list if save_path is None else None, # Return frames only if not saved
        'perturbation_type': perturbation_type  # Include the type of perturbation used
    }
    
