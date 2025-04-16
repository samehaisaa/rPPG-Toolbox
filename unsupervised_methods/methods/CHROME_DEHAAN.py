# The Chrominance Method from: De Haan, G., & Jeanne, V. (2013). 
# Robust pulse rate from chrominance-based rPPG. IEEE Transactions on Biomedical Engineering, 60(10), 2878-2886. 
# DOI: 10.1109/TBME.2013.2266196
import numpy as np
import math
from scipy import signal
import unsupervised_methods.utils as utils

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

def CHROME_DEHAAN(frames, FS, n_perturbations=10, noise_std_fraction=0.01):
    """Enhanced CHROM method with uncertainty quantification via input perturbation."""
    LPF = 0.7
    HPF = 2.5
    WinSec = 1.6

    RGB_original = process_video(frames)
    if RGB_original.size == 0:
        print("Error: process_video returned empty RGB array.")
        return {'BVP': np.array([]), 'confidence_bands': np.array([[], []]), 'perturbed_signals': np.array([])}
        
    rgb_mean = np.mean(np.abs(RGB_original)) # Estimate signal magnitude
    noise_std = noise_std_fraction * rgb_mean # Scale noise to signal
    
    perturbed_bvp_signals = []
    
    # Calculate original BVP first
    original_bvp = _calculate_bvp_from_rgb(RGB_original, FS, LPF, HPF, WinSec)
    if original_bvp.size == 0:
        print("Warning: Original BVP calculation resulted in empty array.")
        # Decide how to handle this - return empty or try perturbations?
        # For now, return empty based on original failure.
        return {'BVP': np.array([]), 'confidence_bands': np.array([[], []]), 'perturbed_signals': np.array([])}
        
    perturbed_bvp_signals.append(original_bvp)
    
    # Generate and process perturbed signals
    for _ in range(n_perturbations):
        noise = np.random.normal(0, noise_std, RGB_original.shape)
        RGB_perturbed = RGB_original + noise
        bvp_perturbed = _calculate_bvp_from_rgb(RGB_perturbed, FS, LPF, HPF, WinSec)
        
        # Ensure perturbed BVP has same length as original via padding/truncation if needed
        if len(bvp_perturbed) != len(original_bvp):
            print(f"Warning: Perturbed BVP length ({len(bvp_perturbed)}) differs from original ({len(original_bvp)}). Adjusting.")
            # Simple truncation/padding - more sophisticated alignment might be better
            new_bvp = np.zeros_like(original_bvp)
            common_length = min(len(bvp_perturbed), len(original_bvp))
            new_bvp[:common_length] = bvp_perturbed[:common_length]
            bvp_perturbed = new_bvp
            
        perturbed_bvp_signals.append(bvp_perturbed)
        
    if not perturbed_bvp_signals:
         print("Error: No BVP signals generated.")
         return {'BVP': np.array([]), 'confidence_bands': np.array([[], []]), 'perturbed_signals': np.array([])}
         
    # Calculate mean BVP and confidence bands
    perturbed_bvp_signals = np.array(perturbed_bvp_signals)
    mean_bvp = np.mean(perturbed_bvp_signals, axis=0)
    ci_lower = np.percentile(perturbed_bvp_signals, 2.5, axis=0)
    ci_upper = np.percentile(perturbed_bvp_signals, 97.5, axis=0)
    confidence_bands = np.array([ci_lower, ci_upper])

    return {
        'BVP': mean_bvp,  # Mean BVP signal from perturbations
        'confidence_bands': confidence_bands,  # Lower and upper confidence bounds
        'perturbed_signals': perturbed_bvp_signals # Return all generated signals
    }

def process_video(frames):
    "Calculates the average value of each frame."
    RGB = []
    for frame in frames:
        # Added check for empty frame
        if frame is None or frame.size == 0:
            print("Warning: Encountered empty frame, skipping.")
            continue 
        sum_val = np.sum(np.sum(frame, axis=0), axis=0)
        # Check for potential division by zero if frame dimensions are 0
        num_pixels = frame.shape[0] * frame.shape[1]
        if num_pixels == 0:
             print("Warning: Frame with zero pixels encountered, skipping.")
             continue
        RGB.append(sum_val / num_pixels)
    return np.asarray(RGB)
    
