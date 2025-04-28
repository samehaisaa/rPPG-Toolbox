#!/usr/bin/env python
"""
Script to train and test the UncertaintyWrapper for rPPG signal uncertainty estimation.

This script:
1. Trains the UncertaintyWrapper on a dataset with ground truth
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
from unsupervised_methods.uncertainty_wrapper import UncertaintyWrapper
import glob
from visualization import plot_bvp_with_confidence


def train_uncertainty_model(config, data_loader, output_model_path):
    """
    Train the UncertaintyWrapper model using the provided data loader.
    
    Args:
        config: Configuration object
        data_loader: Data loader containing video data and ground truth
        output_model_path: Path to save the trained model
        
    Returns:
        Trained UncertaintyWrapper instance
    """
    print("===Training UncertaintyWrapper Model===")
    
    # Initialize UncertaintyWrapper with sampling frequency from config
    wrapper = UncertaintyWrapper(fs=config.UNSUPERVISED.DATA.FS)
    
    # Extract video paths and ground truth signals from data loader
    video_paths = []
    gt_signals = []
    
    for batch_idx, batch in enumerate(data_loader["unsupervised"]):
        # Extract video data and ground truth from batch
        video_data = batch[0].cpu().numpy()
        gt_bvp = batch[1].cpu().numpy()
        
        # For each item in the batch
        for idx in range(video_data.shape[0]):
            # Create a temporary video file
            video_frames = video_data[idx]
            video_name = f"temp_video_{batch_idx}_{idx}.mp4"
            video_path = os.path.join("perturbed_data", "temp_videos", video_name)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            
            # Save frames as a video file
            _save_video(video_frames, video_path)
            
            # Add to lists
            video_paths.append(video_path)
            gt_signals.append(gt_bvp[idx])
    
    # Train the model
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
    Test the trained UncertaintyWrapper model on a test video.
    
    Args:
        config: Configuration object
        model_path: Path to the trained model
        test_video_path: Path to the test video file
        
    Returns:
        Tuple of (timestamps, ppg_signal, uncertainties)
    """
    print(f"===Testing UncertaintyWrapper Model on {test_video_path}===")
    
    # Load trained model
    wrapper = UncertaintyWrapper(fs=config.UNSUPERVISED.DATA.FS, model_path=model_path)
    
    # Process test video
    timestamps, ppg_signal, uncertainties = wrapper.predict_with_uncertainty(test_video_path)
    
    return timestamps, ppg_signal, uncertainties


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
    parser = argparse.ArgumentParser(description='Run UncertaintyWrapper for rPPG signal uncertainty estimation')
    parser.add_argument('--config', type=str, default='configs/infer_configs/UBFC-rPPG_CHROM_UNCERTAINTY.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'train_and_test'], default='train_and_test',
                        help='Mode to run: train, test, or both')
    parser.add_argument('--model_path', type=str, default='model_outputs/uncertainty_model/chrom_uncertainty_model.joblib',
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
            # Create dataset and dataloader for unsupervised method
            unsupervised_data = unsupervised_loader(
                name="unsupervised",
                data_path=config.UNSUPERVISED.DATA.DATA_PATH,
                config_data=config.UNSUPERVISED.DATA,
                device=config.DEVICE
            )
            
            data_loaders["unsupervised"] = DataLoader(
                dataset=unsupervised_data,
                batch_size=1,
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
