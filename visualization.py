import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

def plot_bvp_with_confidence(bvp_signal, confidence_bands, fs, gt_bvp_signal=None, title="BVP Signal with Perturbation Confidence Bands", save_path=None):
    """
    Plots the BVP signal with confidence bands derived from input perturbation,
    optionally including the ground truth BVP signal.

    Args:
        bvp_signal (np.array): The mean BVP signal.
        confidence_bands (np.array): A (2, N) array with lower and upper confidence bounds.
        fs (float): The sampling frequency of the signal.
        gt_bvp_signal (np.array, optional): The ground truth BVP signal. Defaults to None.
        title (str): The title for the plot.
        save_path (str, optional): Path to save the figure. If None, displays the plot.
    """
    if bvp_signal is None or bvp_signal.size == 0 or confidence_bands is None or confidence_bands.shape[0] != 2 or confidence_bands.shape[1] != len(bvp_signal):
        print(f"Warning: Invalid input for plot_bvp_with_confidence ({title}). Skipping plot.")
        return
        
    time_vector = np.arange(len(bvp_signal)) / fs
    
    plt.figure(figsize=(12, 4))
    plt.plot(time_vector, bvp_signal, label='Mean Predicted BVP', color='blue')
    plt.fill_between(time_vector, 
                     confidence_bands[0], 
                     confidence_bands[1], 
                     color='lightblue', 
                     alpha=0.5, 
                     label='95% Confidence Interval (Perturbation)')
                     
    # Plot Ground Truth BVP if provided and valid
    if gt_bvp_signal is not None:
        if len(gt_bvp_signal) == len(bvp_signal):
             # Normalize GT signal for visualization purposes if needed (optional)
             # gt_bvp_normalized = (gt_bvp_signal - np.mean(gt_bvp_signal)) / np.std(gt_bvp_signal)
             # bvp_normalized = (bvp_signal - np.mean(bvp_signal)) / np.std(bvp_signal)
             # plt.plot(time_vector, bvp_normalized, label='Mean Predicted BVP (Norm)', color='blue')
             # plt.plot(time_vector, gt_bvp_normalized, label='Ground Truth BVP (Norm)', color='green', linestyle=':')
             plt.plot(time_vector, gt_bvp_signal, label='Ground Truth BVP', color='green', linestyle=':')
        else:
             print(f"Warning: GT BVP signal length ({len(gt_bvp_signal)}) does not match predicted BVP length ({len(bvp_signal)}) for plot '{title}'. Skipping GT plot.")
    
    plt.xlabel("Time (s)")
    plt.ylabel("BVP Amplitude")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        # print(f"BVP confidence plot saved to: {save_path}") # Reduce print frequency
        plt.close() # Close the plot after saving
    else:
        plt.show()

def plot_hr_distribution(perturbed_hrs, mean_hr=None, gt_hr=None, title="HR Distribution from Perturbed Signals", save_path=None):
    """
    Plots the distribution of heart rates calculated from perturbed signals for a single window.

    Args:
        perturbed_hrs (list or np.array): A list/array of HR values from perturbations.
        mean_hr (float, optional): The HR calculated from the mean BVP signal.
        gt_hr (float, optional): The ground truth HR for comparison.
        title (str): The title for the plot.
        save_path (str, optional): Path to save the figure. If None, displays the plot.
    """
    if perturbed_hrs is None or len(perturbed_hrs) == 0:
        print("Warning: No perturbed HRs provided for plot_hr_distribution. Skipping plot.")
        return

    perturbed_hrs = np.array(perturbed_hrs).flatten() # Ensure it's a flat array
    perturbed_hrs = perturbed_hrs[~np.isnan(perturbed_hrs)] # Remove NaNs

    if len(perturbed_hrs) == 0:
        print("Warning: No valid perturbed HRs after removing NaNs. Skipping plot.")
        return

    plt.figure(figsize=(8, 6))
    sns.histplot(perturbed_hrs, kde=True, label='Perturbed HR Distribution', stat='density')
    
    min_hr, max_hr = np.min(perturbed_hrs), np.max(perturbed_hrs)
    plot_min = min_hr - 5
    plot_max = max_hr + 5
    
    if mean_hr is not None and not np.isnan(mean_hr):
        plt.axvline(mean_hr, color='red', linestyle='--', label=f'Mean BVP HR: {mean_hr:.2f}')
        plot_min = min(plot_min, mean_hr - 5)
        plot_max = max(plot_max, mean_hr + 5)
        
    if gt_hr is not None and not np.isnan(gt_hr):
        plt.axvline(gt_hr, color='green', linestyle=':', label=f'Ground Truth HR: {gt_hr:.2f}')
        plot_min = min(plot_min, gt_hr - 5)
        plot_max = max(plot_max, gt_hr + 5)
        
    plt.xlabel("Heart Rate (BPM)")
    plt.ylabel("Density")
    plt.title(title)
    plt.xlim(plot_min, plot_max)
    plt.legend()
    plt.grid(True, axis='x')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"HR distribution plot saved to: {save_path}")
        plt.close() # Close the plot after saving
    else:
        plt.show() 
