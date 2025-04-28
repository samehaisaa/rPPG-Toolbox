#!/usr/bin/env python
"""
Script to train and test the SupervisedUncertaintyWrapper for rPPG signal uncertainty estimation.

This script:
1. Trains the SupervisedUncertaintyWrapper on a dataset with ground truth
2. Tests it on new videos to produce PPG signals with uncertainty
3. Visualizes the results
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from yacs.config import CfgNode as CN
import torch
from torch.utils.data import DataLoader
from config import get_config
from dataset import data_loader
from unsupervised_methods.supervised_uncertainty_wrapper import SupervisedUncertaintyWrapper
import glob
from visualization import plot_bvp_with_confidence


def train_uncertainty_model(config, data_loader, output_model_path):
    """
    Train the SupervisedUncertaintyWrapper model using the provided data loader.
    
    Args:
        config: Configuration object
        data_loader: Data loader containing video data and ground truth
        output_model_path: Path to save the trained model
        
    Returns:
        Trained SupervisedUncertaintyWrapper instance
    """
    print("===Training Supervised Uncertainty Model===")
    
    # Initialize SupervisedUncertaintyWrapper with sampling frequency from config
    wrapper = SupervisedUncertaintyWrapper(fs=config.UNSUPERVISED.DATA.FS)
    
    # Extract video paths and ground truth signals from data loader
    video_paths = []
    gt_signals = []
    
    # Process each batch from the data loader
    for batch_idx, batch in enumerate(data_loader["unsupervised"]):
        # Extract video data and ground truth from batch
        video_data = batch[0].cpu().numpy()
        gt_bvp = batch[1].cpu().numpy()
        
        # For each item in the batch
        for idx in range(video_data.shape[0]):
            # Create a temporary video file
            video_frames = video_data[idx]
            video_name = f"temp_video_{batch_idx}_{idx}.mp4"
            video_path = os.path.join("preprocessed_data", "temp_videos", video_name)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            
            # Save frames as a video file
            _save_video(video_frames, video_path)
            
            # Add to lists
            video_paths.append(video_path)
            gt_signals.append(gt_bvp[idx])
    
    print(f"Collected {len(video_paths)} videos for training.")
    
    # Train the model using all videos at once (more efficient than chunks for supervised approach)
    wrapper.train(video_paths, gt_signals, output_model_path)
    
    # Clean up temporary video files
    for video_path in video_paths:
        if os.path.exists(video_path):
            os.remove(video_path)
    
    return wrapper


def _save_video(frames, output_path):
    """
    Save numpy array of frames as a video file.
    
    Args:
        frames (np.ndarray): Video frames of shape (T, H, W, C)
        output_path (str): Path to save the video file
    """
    import cv2
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width = frames.shape[1:3]
    fps = 30  # Default fps
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames
    for frame in frames:
        # Convert to BGR format for OpenCV
        bgr_frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
    
    out.release()


def test_uncertainty_model(config, model_path, test_video_path):
    """
    Test the trained SupervisedUncertaintyWrapper model on a test video.
    
    Args:
        config: Configuration object
        model_path: Path to the trained model
        test_video_path: Path to the test video file
        
    Returns:
        Tuple of (timestamps, ppg_signal, uncertainties)
    """
    print(f"===Testing Supervised Uncertainty Model on {test_video_path}===")
    
    try:
        # Load trained model
        print("Loading model...")
        wrapper = SupervisedUncertaintyWrapper(fs=config.UNSUPERVISED.DATA.FS, model_path=model_path)
        
        # Process test video in smaller chunks to save memory
        print("Processing test video in chunks...")
        chunk_size = 10  # Process 10 seconds at a time
        total_duration = None
        all_timestamps = []
        all_ppg_signals = []
        all_uncertainties = []
        
        # Get video duration
        import cv2
        cap = cv2.VideoCapture(test_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = frame_count / fps
        cap.release()
        
        print(f"Video duration: {total_duration:.2f} seconds")
        
        # Process video in chunks
        current_time = 0
        while current_time < total_duration:
            end_time = min(current_time + chunk_size, total_duration)
            print(f"Processing chunk from {current_time:.1f}s to {end_time:.1f}s...")
            
            try:
                # Process current chunk
                timestamps, ppg_signal, uncertainties = wrapper.predict_with_uncertainty(
                    test_video_path,
                    start_time=current_time,
                    end_time=end_time
                )
                
                # Append results
                all_timestamps.extend(timestamps)
                all_ppg_signals.extend(ppg_signal)
                all_uncertainties.extend(uncertainties)
                
                # Force garbage collection and clear memory
                import gc
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Clear variables to free memory
                del timestamps
                del ppg_signal
                del uncertainties
                
            except Exception as e:
                print(f"Error processing chunk {current_time:.1f}s-{end_time:.1f}s: {str(e)}")
                # Continue with next chunk
                pass
            
            # Move to next chunk
            current_time = end_time
            
            # Additional memory cleanup
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print("Test video processing completed successfully")
        
        # Convert lists to numpy arrays
        timestamps = np.array(all_timestamps)
        ppg_signal = np.array(all_ppg_signals)
        uncertainties = np.array(all_uncertainties)
        
        # Validate the results
        validate_uncertainty(timestamps, ppg_signal, uncertainties)
        
        return timestamps, ppg_signal, uncertainties
    
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        raise


def validate_uncertainty(timestamps, ppg_signal, uncertainties, gt_signal=None):
    """
    Validate the uncertainty predictions using various metrics.
    
    Args:
        timestamps: Time points
        ppg_signal: Extracted PPG signal
        uncertainties: Predicted uncertainties
        gt_signal: Ground truth signal (if available)
    """
    print("\n===Validating Uncertainty Predictions===")
    
    # Create output directory
    output_dir = 'model_outputs/uncertainty_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Basic statistics
    print("\n1. Uncertainty Statistics:")
    print(f"Mean uncertainty: {np.mean(uncertainties):.4f}")
    print(f"Std uncertainty: {np.std(uncertainties):.4f}")
    print(f"Max uncertainty: {np.max(uncertainties):.4f}")
    print(f"Min uncertainty: {np.min(uncertainties):.4f}")
    
    # 2. Signal quality metrics
    print("\n2. Signal Quality Metrics:")
    try:
        from evaluation.post_process import _calculate_SNR, _calculate_fft_hr, _calculate_peak_hr, _compute_macc, power2db
        from scipy.signal import welch
        
        hr_fft = _calculate_fft_hr(ppg_signal, fs=30)
        hr_peak = _calculate_peak_hr(ppg_signal, fs=30)
        snr = _calculate_SNR(ppg_signal, hr_fft, fs=30)
        
        # Calculate Power Spectral Density
        f, pxx = welch(ppg_signal, fs=30, nperseg=min(len(ppg_signal), 256))
        
        # Calculate signal power in different bands
        # Heart rate band: 0.75-2.5 Hz (45-150 bpm)
        hr_band_mask = (f >= 0.75) & (f <= 2.5)
        hr_band_power = np.sum(pxx[hr_band_mask])
        
        # Total power in the signal
        total_power = np.sum(pxx)
        
        # Ratio of HR band power to total power
        hr_power_ratio = hr_band_power / (total_power + 1e-10)
        
        print(f"Signal SNR: {snr:.2f} dB")
        print(f"HR (FFT): {hr_fft:.2f} bpm")
        print(f"HR (Peak): {hr_peak:.2f} bpm")
        print(f"HR band power ratio: {hr_power_ratio:.4f}")
        
        # Save power spectrum plot
        plt.figure(figsize=(10, 5))
        plt.semilogy(f, pxx)
        plt.axvspan(0.75, 2.5, alpha=0.3, color='green', label='HR band (45-150 bpm)')
        plt.grid(True)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.title('Power Spectral Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'power_spectrum.png'))
        plt.close()
        
    except Exception as e:
        print(f"Error calculating signal quality metrics: {str(e)}")
    
    # 3. Uncertainty calibration (if ground truth is available)
    if gt_signal is not None:
        print("\n3. Uncertainty Calibration:")
        # Normalize signals for error calculation
        ppg_norm = (ppg_signal - np.mean(ppg_signal)) / np.std(ppg_signal)
        gt_norm = (gt_signal - np.mean(gt_signal)) / np.std(gt_signal)
        
        # Calculate actual errors
        errors = np.abs(ppg_norm - gt_norm)
        
        # Calculate calibration metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        calibration_error = mean_squared_error(errors, uncertainties)
        mae = mean_absolute_error(errors, uncertainties)
        print(f"Calibration Error (MSE): {calibration_error:.4f}")
        print(f"Calibration Error (MAE): {mae:.4f}")
        
        # Calculate correlation between errors and uncertainties
        correlation = np.corrcoef(errors, uncertainties)[0, 1]
        print(f"Error-Uncertainty Correlation: {correlation:.4f}")
        
        # Calculate percentage of points where uncertainty bounds contain ground truth
        within_bounds = np.sum(np.abs(ppg_norm - gt_norm) <= uncertainties) / len(gt_norm)
        print(f"Percentage within uncertainty bounds: {within_bounds*100:.2f}%")
        
        # Scatter plot of errors vs uncertainties
        plt.figure(figsize=(8, 8))
        plt.scatter(errors, uncertainties, alpha=0.5)
        plt.plot([0, max(np.max(errors), np.max(uncertainties))], 
                [0, max(np.max(errors), np.max(uncertainties))], 
                'r--', label='Perfect calibration')
        plt.xlabel('Actual Errors')
        plt.ylabel('Predicted Uncertainties')
        plt.title('Uncertainty Calibration Plot')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'uncertainty_calibration.png'))
        plt.close()
    
    # 4. Uncertainty distribution analysis
    print("\n4. Uncertainty Distribution Analysis:")
    
    # Histogram of uncertainties
    plt.figure(figsize=(10, 5))
    plt.hist(uncertainties, bins=30, alpha=0.7)
    plt.axvline(np.mean(uncertainties), color='r', linestyle='--', label=f'Mean: {np.mean(uncertainties):.4f}')
    plt.axvline(np.median(uncertainties), color='g', linestyle='--', label=f'Median: {np.median(uncertainties):.4f}')
    plt.grid(True)
    plt.xlabel('Uncertainty Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Uncertainty Values')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'uncertainty_histogram.png'))
    plt.close()
    
    # 5. Visual validation
    print("\n5. Visual Validation:")
    print("Generating validation plots...")
    
    # Create figure with subplots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Signal with uncertainty bands
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, ppg_signal, label='PPG Signal', color='blue')
    plt.fill_between(timestamps, 
                    ppg_signal - uncertainties, 
                    ppg_signal + uncertainties, 
                    color='lightblue', alpha=0.5, label='±1σ Uncertainty')
    if gt_signal is not None:
        plt.plot(timestamps, gt_signal, label='Ground Truth', color='red', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('PPG Signal with Uncertainty Bands')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Uncertainty over time
    plt.subplot(2, 1, 2)
    plt.plot(timestamps, uncertainties, label='Uncertainty', color='green')
    if gt_signal is not None:
        plt.plot(timestamps, np.abs(ppg_signal - gt_signal), 
                label='Actual Error', color='red', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Uncertainty/Error')
    plt.title('Uncertainty vs Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'uncertainty_validation.png'))
    plt.close()
    
    # 6. Chunk-based validation plots
    print("\n6. Generating Chunk-Based Validation Plots:")
    chunk_size = int(5 * 30)  # 5 seconds at 30 fps
    
    for i in range(0, len(timestamps), chunk_size):
        end_idx = min(i + chunk_size, len(timestamps))
        if end_idx - i < chunk_size // 2:  # Skip chunks that are too small
            continue
            
        chunk_timestamps = timestamps[i:end_idx]
        chunk_ppg = ppg_signal[i:end_idx]
        chunk_uncertainties = uncertainties[i:end_idx]
        
        # Normalize signal for better visualization
        chunk_ppg_norm = (chunk_ppg - np.mean(chunk_ppg)) / (np.std(chunk_ppg) + 1e-10)
        
        # Create figure for this chunk
        plt.figure(figsize=(10, 6))
        plt.plot(chunk_timestamps, chunk_ppg_norm, label='PPG Signal (normalized)', color='blue')
        plt.fill_between(chunk_timestamps, 
                        chunk_ppg_norm - chunk_uncertainties, 
                        chunk_ppg_norm + chunk_uncertainties, 
                        color='lightblue', alpha=0.5, label='±1σ Uncertainty')
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized Amplitude')
        plt.title(f'Chunk {i//chunk_size + 1}: Time {chunk_timestamps[0]:.1f}s to {chunk_timestamps[-1]:.1f}s')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'chunk_{i//chunk_size + 1}.png'))
        plt.close()
    
    print(f"All validation plots saved to {output_dir}/")
    
    return None


def visualize_results(timestamps, ppg_signal, uncertainties, output_path=None):
    """
    Visualize the PPG signal with uncertainty.
    
    Args:
        timestamps: Time points
        ppg_signal: Extracted PPG signal
        uncertainties: Uncertainty estimates at each time point
        output_path: Path to save the visualization
    """
    print("===Visualizing Results===")
    
    # Create confidence bands from uncertainties
    # Using 1x and 2x uncertainties for the bands
    lower_band_1x = ppg_signal - uncertainties
    upper_band_1x = ppg_signal + uncertainties
    lower_band_2x = ppg_signal - 2 * uncertainties
    upper_band_2x = ppg_signal + 2 * uncertainties
    
    confidence_bands_1x = np.vstack((lower_band_1x, upper_band_1x))
    confidence_bands_2x = np.vstack((lower_band_2x, upper_band_2x))
    
    # Create figure with two subplots
    plt.figure(figsize=(12, 8))
    
    # Plot 1: PPG signal with 1x uncertainty bands
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, ppg_signal, label='PPG Signal', color='blue')
    plt.fill_between(timestamps, lower_band_1x, upper_band_1x, color='lightblue', alpha=0.5, label='±1σ Uncertainty')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('PPG Signal with 1σ Uncertainty')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: PPG signal with 2x uncertainty bands
    plt.subplot(2, 1, 2)
    plt.plot(timestamps, ppg_signal, label='PPG Signal', color='blue')
    plt.fill_between(timestamps, lower_band_2x, upper_band_2x, color='lightcoral', alpha=0.5, label='±2σ Uncertainty')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('PPG Signal with 2σ Uncertainty')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # Also use the provided visualization function
    output_path2 = output_path.replace('.png', '_conf_bands.png') if output_path else None
    plot_bvp_with_confidence(
        ppg_signal, 
        confidence_bands_1x, 
        fs=30, 
        title="PPG Signal with Learned Uncertainty Bands",
        save_path=output_path2
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run SupervisedUncertaintyWrapper for rPPG signal uncertainty estimation')
    parser.add_argument('--config', type=str, default='configs/infer_configs/UBFC-rPPG_CHROM_UNCERTAINTY.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'train_and_test'], default='train_and_test',
                        help='Mode to run: train, test, or both')
    parser.add_argument('--model_path', type=str, default='model_outputs/uncertainty_model/supervised_chrom_uncertainty_model.joblib',
                        help='Path to save/load the trained model')
    parser.add_argument('--test_video', type=str, default=None,
                        help='Path to a test video file (required for test mode)')
    parser.add_argument('--output_dir', type=str, default='model_outputs/uncertainty_results',
                        help='Directory to save results')
    
    return parser.parse_args()


def main():
    """Main function to run the script."""
    args = parse_args()
    
    # Load configuration
    config_args = argparse.Namespace(config_file=args.config)
    config = get_config(config_args)
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create data loaders
    if args.mode == 'train' or args.mode == 'train_and_test':
        print("Creating data loader for training...")
        data_loaders = dict()
        
        # Select the right data loader based on dataset
        if config.UNSUPERVISED.DATA.DATASET == 'UBFC-rPPG':
            unsupervised_loader = data_loader.UBFCrPPGLoader.UBFCrPPGLoader
        elif config.UNSUPERVISED.DATA.DATASET == 'PURE':
            unsupervised_loader = data_loader.PURELoader.PURELoader
        elif config.UNSUPERVISED.DATA.DATASET == 'SCAMPS':
            unsupervised_loader = data_loader.SCAMPSLoader.SCAMPSLoader
        elif config.UNSUPERVISED.DATA.DATASET == 'MMPD':
            unsupervised_loader = data_loader.MMPDLoader.MMPDLoader
        elif config.UNSUPERVISED.DATA.DATASET == 'BP4D+':
            unsupervised_loader = data_loader.BP4DPlusLoader.BP4DPlusLoader
        elif config.UNSUPERVISED.DATA.DATASET == 'UBFC-PHYS':
            unsupervised_loader = data_loader.UBFCPHYSLoader.UBFCPHYSLoader
        else:
            print(f"Unknown dataset: {config.UNSUPERVISED.DATA.DATASET}")
            unsupervised_loader = None
        
        if unsupervised_loader is not None:
            # Create dataset and dataloader for supervised method
            unsupervised_data = unsupervised_loader(
                name="unsupervised",
                data_path=config.UNSUPERVISED.DATA.DATA_PATH,
                config_data=config.UNSUPERVISED.DATA,
                device=config.DEVICE
            )
            
            # Use a larger batch size since we're not using perturbations anymore
            batch_size = 4  # Adjust based on available memory
            if hasattr(config.UNSUPERVISED.DATA, 'BATCH_SIZE'):
                batch_size = config.UNSUPERVISED.DATA.BATCH_SIZE
            
            data_loaders["unsupervised"] = DataLoader(
                dataset=unsupervised_data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4
            )
        else:
            data_loaders["unsupervised"] = None
    
    # Run in selected mode
    if args.mode == 'train' or args.mode == 'train_and_test':
        # Train the model
        wrapper = train_uncertainty_model(config, data_loaders, args.model_path)
    
    if args.mode == 'test' or args.mode == 'train_and_test':
        # Test video path validation
        if args.test_video is None:
            # If no specific test video provided, use a video from the dataset
            data_path = config.UNSUPERVISED.DATA.DATA_PATH
            test_videos = glob.glob(os.path.join(data_path, '**/*.avi'), recursive=True)
            test_videos.extend(glob.glob(os.path.join(data_path, '**/*.mp4'), recursive=True))
            
            if not test_videos:
                raise ValueError(f"No video files found in {data_path}")
            
            test_video_path = test_videos[0]
            print(f"Using first video found in dataset: {test_video_path}")
        else:
            test_video_path = args.test_video
        
        # Test the model
        timestamps, ppg_signal, uncertainties = test_uncertainty_model(config, args.model_path, test_video_path)
        
        # Visualize the results
        output_file = os.path.join(args.output_dir, 'uncertainty_visualization.png')
        visualize_results(timestamps, ppg_signal, uncertainties, output_file)
        
        print(f"Results saved to {output_file}")


if __name__ == '__main__':
    main() 
