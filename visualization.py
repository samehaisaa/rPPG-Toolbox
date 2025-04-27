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
    
    # Normalize signals for better comparison
    bvp_normalized = (bvp_signal - np.mean(bvp_signal)) / np.std(bvp_signal)
    confidence_bands_normalized = np.zeros_like(confidence_bands)
    confidence_bands_normalized[0] = (confidence_bands[0] - np.mean(bvp_signal)) / np.std(bvp_signal)
    confidence_bands_normalized[1] = (confidence_bands[1] - np.mean(bvp_signal)) / np.std(bvp_signal)
    
    # Plot normalized predicted BVP and confidence bands
    plt.plot(time_vector, bvp_normalized, label='Mean Predicted BVP', color='blue', linewidth=1.5)
    plt.fill_between(time_vector, 
                     confidence_bands_normalized[0], 
                     confidence_bands_normalized[1], 
                     color='lightblue', 
                     alpha=0.3, 
                     label='95% Confidence Interval')
                     
    # Plot Ground Truth BVP if provided and valid
    error_metrics = {}
    if gt_bvp_signal is not None:
        if len(gt_bvp_signal) == len(bvp_signal):
            # Same length - proceed normally
            gt_bvp_normalized = (gt_bvp_signal - np.mean(gt_bvp_signal)) / np.std(gt_bvp_signal)
            plt.plot(time_vector, gt_bvp_normalized, label='Ground Truth BVP',
                    color='green', linestyle='-', linewidth=2, alpha=0.7)
            
            # Calculate error metrics
            error_metrics['MAE'] = np.mean(np.abs(bvp_normalized - gt_bvp_normalized))
            error_metrics['RMSE'] = np.sqrt(np.mean((bvp_normalized - gt_bvp_normalized)**2))
            error_metrics['Pearson'] = np.corrcoef(bvp_normalized, gt_bvp_normalized)[0,1]
            
            # Add error metrics to plot
            metrics_text = f'MAE: {error_metrics["MAE"]:.3f}\nRMSE: {error_metrics["RMSE"]:.3f}\nPearson: {error_metrics["Pearson"]:.3f}'
            plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes,
                    verticalalignment='top', fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        else:
            # Different lengths - use the minimum length and show a note
            min_length = min(len(gt_bvp_signal), len(bvp_signal))
            print(f"Warning: GT BVP signal length ({len(gt_bvp_signal)}) does not match predicted BVP length ({len(bvp_signal)}) for plot '{title}'. Using first {min_length} samples for comparison.")
            
            # Trim both signals to minimum length
            gt_bvp_signal_trimmed = gt_bvp_signal[:min_length]
            bvp_signal_trimmed = bvp_signal[:min_length]
            time_vector_trimmed = time_vector[:min_length] if len(time_vector) > min_length else time_vector
            
            # Normalize ground truth
            gt_bvp_normalized = (gt_bvp_signal_trimmed - np.mean(gt_bvp_signal_trimmed)) / np.std(gt_bvp_signal_trimmed)
            
            # Need to re-normalize bvp_signal for consistent comparison
            bvp_normalized_trimmed = (bvp_signal_trimmed - np.mean(bvp_signal_trimmed)) / np.std(bvp_signal_trimmed)
            
            plt.plot(time_vector_trimmed, gt_bvp_normalized, label='Ground Truth BVP (trimmed)',
                    color='green', linestyle='-', linewidth=2, alpha=0.7)
            
            # Calculate error metrics on trimmed data
            error_metrics['MAE'] = np.mean(np.abs(bvp_normalized_trimmed - gt_bvp_normalized))
            error_metrics['RMSE'] = np.sqrt(np.mean((bvp_normalized_trimmed - gt_bvp_normalized)**2))
            error_metrics['Pearson'] = np.corrcoef(bvp_normalized_trimmed, gt_bvp_normalized)[0,1]
            
            # Add error metrics to plot with a note about trimming
            metrics_text = f'MAE: {error_metrics["MAE"]:.3f}\nRMSE: {error_metrics["RMSE"]:.3f}\nPearson: {error_metrics["Pearson"]:.3f}\n(on trimmed data)'
            plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes,
                    verticalalignment='top', fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized BVP Amplitude")
    plt.title(title)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
    return error_metrics if error_metrics else None

def plot_hr_distribution(hr_values, mean_hr=None, gt_hr=None, title="Heart Rate Distribution", save_path=None):
    """
    Plots the distribution of heart rate values from perturbed signals.

    Args:
        hr_values (list or np.array): List of heart rate values.
        mean_hr (float, optional): Mean heart rate value to highlight. Defaults to None.
        gt_hr (float, optional): Ground truth heart rate to highlight. Defaults to None.
        title (str): The title for the plot.
        save_path (str, optional): Path to save the figure. If None, displays the plot.
    """
    if hr_values is None or len(hr_values) == 0:
        print(f"Warning: Empty heart rate data for {title}. Skipping plot.")
        return
        
    # Filter out NaN values
    hr_values = np.array([hr for hr in hr_values if not np.isnan(hr)])
    
    if len(hr_values) == 0:
        print(f"Warning: All HR values are NaN for {title}. Skipping plot.")
        return
        
    plt.figure(figsize=(10, 6))
    
    # Calculate bin edges with step size of 1 BPM
    min_hr = np.floor(np.min(hr_values))
    max_hr = np.ceil(np.max(hr_values))
    bins = np.arange(min_hr, max_hr + 2) - 0.5  # +2 to include max value, -0.5 to center bins
    
    # Plot histogram with fixed bin size of 1 BPM
    plt.hist(hr_values, bins=bins, color='blue', alpha=0.7, label='HR Distribution')
    
    # Add mean line from perturbed values
    dist_mean = np.mean(hr_values)
    plt.axvline(dist_mean, color='blue', linestyle='-', 
                label=f'Distribution Mean: {dist_mean:.2f} BPM')
    
    # Add mean HR if provided
    if mean_hr is not None and not np.isnan(mean_hr):
        plt.axvline(mean_hr, color='red', linestyle='-', 
                    label=f'Predicted Mean HR: {mean_hr:.2f} BPM')
    
    # Add ground truth HR if provided
    if gt_hr is not None and not np.isnan(gt_hr):
        plt.axvline(gt_hr, color='green', linestyle='-',
                   label=f'Ground Truth HR: {gt_hr:.2f} BPM')
        
        # Show error between mean and ground truth
        if mean_hr is not None and not np.isnan(mean_hr):
            error = abs(mean_hr - gt_hr)
            plt.text(0.02, 0.95, f'Absolute Error: {error:.2f} BPM', 
                    transform=plt.gca().transAxes, fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel("Heart Rate (BPM)")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_perturbation_comparison(results_dict, item_id, window_idx=0, save_path=None):
    """
    Plots a comparison of different perturbation types for the same data item and window.
    
    Args:
        results_dict (dict): Dictionary with perturbation type as keys and result items as values.
        item_id (str): ID of the data item to compare.
        window_idx (int): Index of the window to compare.
        save_path (str, optional): Path to save the figure. If None, displays the plot.
    """
    perturbation_types = list(results_dict.keys())
    
    # Filter to get only the specified item_id
    filtered_results = {}
    for pert_type, results in results_dict.items():
        for result in results:
            if result['item_id'] == item_id and len(result['windows']) > window_idx:
                filtered_results[pert_type] = result
                break
    
    if not filtered_results:
        print(f"Warning: No data found for item_id {item_id} and window {window_idx}. Skipping comparison plot.")
        return
        
    # Set up figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Heart Rate distributions
    colors = plt.cm.tab10(np.linspace(0, 1, len(filtered_results)))
    
    for i, (pert_type, result) in enumerate(filtered_results.items()):
        window = result['windows'][window_idx]
        hr_method_key = [k for k in window.keys() if k.startswith('perturbed_hr_')][0]
        hr_values = np.array([hr for hr in window[hr_method_key] if not np.isnan(hr)])
        
        if len(hr_values) == 0:
            continue
            
        # Plot KDE
        sns.kdeplot(hr_values, ax=ax1, color=colors[i], label=f"{pert_type.replace('_', ' ').title()}")
        
        # Add mean and CI
        mean_hr = np.mean(hr_values)
        ci_lower = np.percentile(hr_values, 2.5)
        ci_upper = np.percentile(hr_values, 97.5)
        ax1.axvline(mean_hr, color=colors[i], linestyle='-', alpha=0.7)
        ax1.axvspan(ci_lower, ci_upper, alpha=0.1, color=colors[i])
        
    # If ground truth available, add it to the plot
    gt_hr = None
    for result in filtered_results.values():
        if 'hr_label' in result['windows'][window_idx]:
            gt_hr = result['windows'][window_idx]['hr_label']
            if not np.isnan(gt_hr):
                ax1.axvline(gt_hr, color='green', linestyle='-', linewidth=2, 
                          label=f'Ground Truth: {gt_hr:.2f} BPM')
            break
    
    ax1.set_xlabel("Heart Rate (BPM)")
    ax1.set_ylabel("Density")
    ax1.set_title(f"Heart Rate Distribution Comparison - {item_id}, Window {window_idx}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Uncertainty metrics
    pert_names = []
    mean_hrs = []
    ci_widths = []
    errors = []
    
    for pert_type, result in filtered_results.items():
        window = result['windows'][window_idx]
        hr_method_key = [k for k in window.keys() if k.startswith('perturbed_hr_')][0]
        hr_values = np.array([hr for hr in window[hr_method_key] if not np.isnan(hr)])
        
        if len(hr_values) == 0:
            continue
            
        mean_hr = np.mean(hr_values)
        ci_lower = np.percentile(hr_values, 2.5)
        ci_upper = np.percentile(hr_values, 97.5)
        ci_width = ci_upper - ci_lower
        
        pert_names.append(pert_type.replace('_', ' ').title())
        mean_hrs.append(mean_hr)
        ci_widths.append(ci_width)
        
        if gt_hr is not None and not np.isnan(gt_hr):
            errors.append(abs(mean_hr - gt_hr))
        else:
            errors.append(0)
    
    # Bar chart for CI widths
    x = np.arange(len(pert_names))
    width = 0.35
    
    ax2.bar(x - width/2, ci_widths, width, label='CI Width (BPM)', color='skyblue')
    
    if gt_hr is not None and not np.isnan(gt_hr):
        ax2.bar(x + width/2, errors, width, label='Absolute Error (BPM)', color='salmon')
    
    ax2.set_xlabel("Perturbation Type")
    ax2.set_ylabel("Value (BPM)")
    ax2.set_title("Uncertainty Metrics by Perturbation Type")
    ax2.set_xticks(x)
    ax2.set_xticklabels(pert_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close() # Close the plot after saving
    else:
        plt.show()

def plot_uncertainty_vs_error(results, window_idx=0, save_path=None):
    """
    Plots the relationship between prediction uncertainty (CI width) and error.
    
    Args:
        results (list): List of result items from different runs.
        window_idx (int): Window index to analyze.
        save_path (str, optional): Path to save the figure. If None, displays the plot.
    """
    ci_widths = []
    errors = []
    perturbation_types = []
    
    for result in results:
        if len(result['windows']) <= window_idx:
            continue
            
        window = result['windows'][window_idx]
        
        # Skip if necessary data is missing
        if 'hr_label' not in window or np.isnan(window['hr_label']):
            continue
            
        hr_method_key = [k for k in window.keys() if k.startswith('perturbed_hr_')][0]
        if hr_method_key not in window or not window[hr_method_key]:
            continue
            
        hr_values = np.array([hr for hr in window[hr_method_key] if not np.isnan(hr)])
        if len(hr_values) == 0:
            continue
            
        # Calculate confidence interval and width
        ci_lower = np.percentile(hr_values, 2.5)
        ci_upper = np.percentile(hr_values, 97.5)
        ci_width = ci_upper - ci_lower
        
        # Calculate prediction error
        mean_hr = np.mean(hr_values)
        gt_hr = window['hr_label']
        error = abs(mean_hr - gt_hr)
        
        ci_widths.append(ci_width)
        errors.append(error)
        perturbation_types.append(result.get('perturbation_type', 'unknown'))
    
    if not ci_widths:
        print("Warning: No valid data for uncertainty vs error plot. Skipping.")
        return
        
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    
    # Use different colors for different perturbation types
    unique_types = list(set(perturbation_types))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
    color_map = {ptype: colors[i] for i, ptype in enumerate(unique_types)}
    
    for i, (width, err, ptype) in enumerate(zip(ci_widths, errors, perturbation_types)):
        plt.scatter(width, err, color=color_map[ptype], label=ptype if ptype not in plt.gca().get_legend_handles_labels()[1] else "")
    
    # Add trend line
    if len(ci_widths) > 1:
        z = np.polyfit(ci_widths, errors, 1)
        p = np.poly1d(z)
        plt.plot(sorted(ci_widths), p(sorted(ci_widths)), "r--", alpha=0.8, label="Trend Line")
        
        # Calculate correlation
        corr = np.corrcoef(ci_widths, errors)[0, 1]
        plt.text(0.02, 0.95, f"Correlation: {corr:.3f}", transform=plt.gca().transAxes,
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel("Confidence Interval Width (BPM)")
    plt.ylabel("Absolute Error (BPM)")
    plt.title("Relationship Between Uncertainty and Prediction Error")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close() # Close the plot after saving
    else:
        plt.show() 
