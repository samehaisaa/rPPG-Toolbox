"""
Script to run uncertainty quantification experiments with different perturbation types.
This script runs the CHROM algorithm with various video perturbation techniques
to quantify uncertainty in rPPG predictions.
"""

import os
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from shutil import copyfile
from yacs.config import CfgNode as CN
import torch
from torch.utils.data import DataLoader
from config import get_config
from dataset import data_loader
from neural_methods import trainer
from unsupervised_methods.unsupervised_predictor import unsupervised_predict

from visualization import (
    plot_bvp_with_confidence, 
    plot_hr_distribution, 
    plot_perturbation_comparison,
    plot_uncertainty_vs_error
)

# For reproducibility
def seed_worker(worker_id):
    worker_seed = 42
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def update_config_perturbation(config, perturbation_type, params=None):
    """Update config with perturbation parameters."""
    config.defrost()
    
    # Set perturbation type
    if not hasattr(config.UNSUPERVISED, 'CHROM_PERTURBATIONS'):
        config.UNSUPERVISED.CHROM_PERTURBATIONS = CN()
    
    config.UNSUPERVISED.CHROM_PERTURBATIONS.TYPE = perturbation_type
    
    # Get default parameters from config if available
    if hasattr(config.UNSUPERVISED.CHROM_PERTURBATIONS, 'PARAMS') and \
       hasattr(config.UNSUPERVISED.CHROM_PERTURBATIONS.PARAMS, perturbation_type.upper()):
        default_params = getattr(config.UNSUPERVISED.CHROM_PERTURBATIONS.PARAMS, perturbation_type.upper())
    else:
        default_params = CN()
    
    # Update with provided parameters
    if params:
        for key, value in params.items():
            setattr(default_params, key, value)
    
    # Set parameters in config
    if not hasattr(config.UNSUPERVISED.CHROM_PERTURBATIONS, 'PARAMS'):
        config.UNSUPERVISED.CHROM_PERTURBATIONS.PARAMS = CN()
    setattr(config.UNSUPERVISED.CHROM_PERTURBATIONS.PARAMS, perturbation_type.upper(), default_params)
    
    config.freeze()
    return config

def run_experiment(config_file, perturbation_configs, output_dir="./model_outputs/uncertainty_experiments"):
    """
    Run experiments with different perturbation types and generate comparison visualizations.
    
    Args:
        config_file: Path to base configuration file
        perturbation_configs: Dictionary of perturbation types and their parameters
        output_dir: Base directory to save results
    """
    # Load base configuration
    args = argparse.Namespace(config_file=config_file)
    config = get_config(args)
    
    # Set common output directory
    config.defrost()
    config.LOG.PATH = output_dir
    config.freeze()
    
    # Dictionary to store results for each perturbation type
    all_results = {}
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save a copy of the base config
    base_config_path = os.path.join(output_dir, "base_config.yaml")
    copyfile(config_file, base_config_path)
    
    # Run each perturbation type
    for pert_type, params in perturbation_configs.items():
        print(f"\n{'='*50}")
        print(f"Running experiment with perturbation type: {pert_type}")
        print(f"Parameters: {params}")
        print(f"{'='*50}\n")
        
        # Update config for this perturbation
        experiment_config = update_config_perturbation(config.clone(), pert_type, params)
        
        # Create output directory for this perturbation type
        pert_output_dir = os.path.join(output_dir, pert_type)
        os.makedirs(pert_output_dir, exist_ok=True)
        
        # Create directory for perturbed videos
        if experiment_config.UNSUPERVISED.CHROM_PERTURBATIONS.SAVE_PERTURBED_VIDEOS:
            videos_dir = os.path.join(pert_output_dir, experiment_config.UNSUPERVISED.CHROM_PERTURBATIONS.SAVE_PATH)
            os.makedirs(videos_dir, exist_ok=True)
        else:
            videos_dir = None
        
        # Save the configuration for this experiment
        with open(os.path.join(pert_output_dir, "config.yaml"), 'w') as f:
            yaml.dump(experiment_config.dump(), f, default_flow_style=False)
        
        # Create data loader
        print("Creating data loader...")
        data_loaders = create_data_loader(experiment_config)
        
        if data_loaders["unsupervised"] is None:
            print(f"Error: Could not create data loader for {pert_type}. Skipping.")
            continue
        
        # Run the experiment
        print(f"Running CHROM with {pert_type} perturbation...")
        
        # Get perturbation parameters
        pert_params = {}
        if hasattr(experiment_config.UNSUPERVISED.CHROM_PERTURBATIONS.PARAMS, pert_type.upper()):
            pert_params = getattr(experiment_config.UNSUPERVISED.CHROM_PERTURBATIONS.PARAMS, pert_type.upper())
        
        # Process each batch
        all_batch_results = []
        for batch_idx, (frames, labels) in enumerate(data_loaders["unsupervised"]):
            frames = frames.cpu().numpy()
            labels = labels.cpu().numpy()
            
            # Create video save directory for this batch
            if videos_dir:
                batch_video_dir = os.path.join(videos_dir, f'batch_{batch_idx}')
            else:
                batch_video_dir = None
            
            # Process each item in batch
            for idx in range(frames.shape[0]):
                result = CHROME_DEHAAN(
                    frames[idx],
                    experiment_config.UNSUPERVISED.DATA.FS,
                    n_perturbations=experiment_config.UNSUPERVISED.CHROM_PERTURBATIONS.N_PERTURBATIONS,
                    perturbation_type=pert_type,
                    perturbation_params=pert_params,
                    save_path=os.path.join(batch_video_dir, f'item_{idx}') if batch_video_dir else None
                )
                
                # Add metadata to result
                result['id'] = f'batch_{batch_idx}_item_{idx}'
                result['gt_bvp'] = labels[idx]
                all_batch_results.append(result)
        
        # Store results
        all_results[pert_type] = all_batch_results
        
        # Generate visualizations
        generate_visualizations(all_batch_results, pert_type, pert_output_dir, experiment_config)
    
    # Generate comparative visualizations
    generate_comparative_visualizations(all_results, output_dir)
    
    print("\nAll experiments completed!")
    return all_results

def create_data_loader(config):
    """Create data loader based on configuration."""
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
        return {"unsupervised": None}
    
    # Create dataset and dataloader
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
        num_workers=4,
        worker_init_fn=seed_worker
    )
    
    return data_loaders

def generate_visualizations(results, pert_type, output_dir, config):
    """Generate visualizations for a single perturbation type."""
    print(f"Generating visualizations for {pert_type}...")
    for result in results:
        item_id = result['id']
        
        # Plot BVP with confidence bands
        if result.get('mean_bvp') is not None and result.get('confidence_bands') is not None:
            plot_bvp_with_confidence(
                result['mean_bvp'],
                result['confidence_bands'],
                config.UNSUPERVISED.DATA.FS,
                gt_bvp_signal=result['gt_bvp'],
                title=f"CHROM BVP with {pert_type} Perturbation - {item_id}",
                save_path=os.path.join(output_dir, f"{item_id}_bvp_confidence.png")
            )
        
        # Plot HR distributions for each window
        for j, window in enumerate(result.get('windows', [])):
            hr_keys = [k for k in window.keys() if k.startswith('perturbed_hr_')]
            if hr_keys:
                hr_method_key = hr_keys[0]
                if hr_method_key in window and window[hr_method_key]:
                    plot_hr_distribution(
                        window[hr_method_key],
                        mean_hr=window.get('hr_pred'),
                        gt_hr=window.get('hr_label'),
                        title=f"HR Distribution with {pert_type} Perturbation - {item_id}, Window {j}",
                        save_path=os.path.join(output_dir, f"{item_id}_window_{j}_hr_dist.png")
                    )

def generate_comparative_visualizations(all_results, output_dir):
    """Generate comparative visualizations across perturbation types."""
    print("\nGenerating comparative visualizations...")
    
    # Create directory for comparative visualizations
    compare_dir = os.path.join(output_dir, "comparison")
    os.makedirs(compare_dir, exist_ok=True)
    
    # Find all unique item IDs
    all_item_ids = {result['id'] for results in all_results.values() for result in results}
    
    # Generate comparison plots for each item
    for item_id in all_item_ids:
        # Get maximum number of windows
        max_windows = max(
            len(result.get('windows', []))
            for results in all_results.values()
            for result in results
            if result['id'] == item_id
        )
        
        # Generate plots for each window
        for window_idx in range(max_windows):
            try:
                plot_perturbation_comparison(
                    all_results,
                    item_id,
                    window_idx=window_idx,
                    save_path=os.path.join(compare_dir, f"{item_id}_window_{window_idx}_comparison.png")
                )
            except Exception as e:
                print(f"Error generating comparison plot for {item_id}, window {window_idx}: {e}")
    
    # Generate uncertainty vs error plots
    try:
        all_items = []
        for pert_type, results in all_results.items():
            for result in results:
                result['perturbation_type'] = pert_type
                all_items.append(result)
        
        for window_idx in range(3):  # First few windows
            try:
                plot_uncertainty_vs_error(
                    all_items,
                    window_idx=window_idx,
                    save_path=os.path.join(compare_dir, f"uncertainty_vs_error_window_{window_idx}.png")
                )
            except Exception as e:
                print(f"Error generating uncertainty vs error plot for window {window_idx}: {e}")
    except Exception as e:
        print(f"Error in uncertainty vs error plotting: {e}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run uncertainty quantification experiments")
    parser.add_argument("--config", type=str, default="configs/infer_configs/UBFC-rPPG_CHROM_UNCERTAINTY.yaml",
                        help="Path to base configuration file")
    parser.add_argument("--output", type=str, default="./model_outputs/uncertainty_experiments",
                        help="Output directory for experiments")
    args = parser.parse_args()
    
    # Define perturbation configurations
    perturbation_configs = {
        "gaussian_noise": {"NOISE_STD_FRACTION": 0.02},
        "blur": {"KERNEL_SIZE": 3},
        "brightness": {"BRIGHTNESS_FACTOR": 0.1},
        "crop": {"CROP_FRACTION": 0.1},
        "rotation": {"ANGLE_RANGE": 5},
        "color_jitter": {"FACTOR": 0.1},
        "compression": {"QUALITY": 80}
    }
    
    # Run experiments
    run_experiment(args.config, perturbation_configs, args.output) 
