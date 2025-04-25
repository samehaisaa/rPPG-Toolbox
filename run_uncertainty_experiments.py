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
import numpy as np
import torch
from config import get_config
from dataset import data_loader
from neural_methods import trainer
from unsupervised_methods.unsupervised_predictor import unsupervised_predict
from torch.utils.data import DataLoader

import dataset.data_loader as data_loader
from config import get_config
from unsupervised_methods.unsupervised_predictor import unsupervised_predict
from visualization import (
    plot_bvp_with_confidence, 
    plot_hr_distribution, 
    plot_perturbation_comparison,
    plot_uncertainty_vs_error
)

def update_config_perturbation(config, perturbation_type, params=None):
    """
    Update the configuration with the specified perturbation type and parameters.
    
    Args:
        config: Configuration object
        perturbation_type: Type of perturbation to use
        params: Dictionary of perturbation parameters
        
    Returns:
        Updated configuration
    """
    # Create CHROM_PERTURBATIONS if it doesn't exist
    if not hasattr(config.UNSUPERVISED, 'CHROM_PERTURBATIONS'):
        config.UNSUPERVISED.CHROM_PERTURBATIONS = CN()

    # Set perturbation type
    config.UNSUPERVISED.CHROM_PERTURBATIONS.TYPE = perturbation_type
    
    # Set number of perturbations
    if not hasattr(config.UNSUPERVISED.CHROM_PERTURBATIONS, 'N_PERTURBATIONS'):
        config.UNSUPERVISED.CHROM_PERTURBATIONS.N_PERTURBATIONS = 30
        
    # Set noise standard deviation fraction
    if not hasattr(config.UNSUPERVISED.CHROM_PERTURBATIONS, 'NOISE_STD_FRACTION'):
        config.UNSUPERVISED.CHROM_PERTURBATIONS.NOISE_STD_FRACTION = 0.01
    
    # Set perturbation parameters
    if params:
        if not hasattr(config.UNSUPERVISED.CHROM_PERTURBATIONS, 'PARAMS'):
            config.UNSUPERVISED.CHROM_PERTURBATIONS.PARAMS = CN()
            
        for key, value in params.items():
            setattr(config.UNSUPERVISED.CHROM_PERTURBATIONS.PARAMS, key.upper(), value)
    
    # Update output directory to include perturbation type
    config.LOG.PATH = os.path.join(config.LOG.PATH, perturbation_type)
    
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
        
        # Save the configuration for this experiment
        with open(os.path.join(pert_output_dir, "config.yaml"), 'w') as f:
            # Convert CfgNode to dict for easier writing
            yaml.dump(experiment_config.dump(), f, default_flow_style=False)
        
        # Create data loader
        print("Creating data loader...")
        # Create dictionary of data loaders, similar to how main.py does it
        data_loaders = dict()
        
        # Select the right data loader based on dataset
        if experiment_config.UNSUPERVISED.DATA.DATASET == 'UBFC-rPPG':
            unsupervised_loader = data_loader.UBFCrPPGLoader.UBFCrPPGLoader
        elif experiment_config.UNSUPERVISED.DATA.DATASET == 'PURE':
            unsupervised_loader = data_loader.PURELoader.PURELoader
        elif experiment_config.UNSUPERVISED.DATA.DATASET == 'SCAMPS':
            unsupervised_loader = data_loader.SCAMPSLoader.SCAMPSLoader
        elif experiment_config.UNSUPERVISED.DATA.DATASET == 'MMPD':
            unsupervised_loader = data_loader.MMPDLoader.MMPDLoader
        elif experiment_config.UNSUPERVISED.DATA.DATASET == 'BP4D+':
            unsupervised_loader = data_loader.BP4DPlusLoader.BP4DPlusLoader
        elif experiment_config.UNSUPERVISED.DATA.DATASET == 'UBFC-PHYS':
            unsupervised_loader = data_loader.UBFCPHYSLoader.UBFCPHYSLoader
        else:
            print(f"Unknown dataset: {experiment_config.UNSUPERVISED.DATA.DATASET}")
            unsupervised_loader = None
        
        if unsupervised_loader is not None:
            # Create dataset and dataloader for unsupervised method
            unsupervised_data = unsupervised_loader(
                name="unsupervised",
                config_data=experiment_config,
                training=False,
                unsupervised=True
            )
            
            data_loaders["unsupervised"] = DataLoader(
                dataset=unsupervised_data,
                batch_size=1,
                shuffle=False,
                num_workers=8,
                pin_memory=True
            )
        else:
            data_loaders["unsupervised"] = None
        
        # Run the experiment
        print(f"Running CHROM with {pert_type} perturbation...")
        results = unsupervised_predict(experiment_config, data_loaders, "CHROM")
        
        # Store results
        all_results[pert_type] = results
        
        # Generate individual visualizations
        print(f"Generating visualizations for {pert_type}...")
        for i, result in enumerate(results):
            item_id = result['item_id']
            
            # Plot BVP with confidence bands
            if result['bvp_mean'] is not None and result['confidence_bands'] is not None:
                plot_bvp_with_confidence(
                    result['bvp_mean'],
                    result['confidence_bands'],
                    experiment_config.UNSUPERVISED.DATA.FS,
                    gt_bvp_signal=result['gt_bvp'],
                    title=f"CHROM BVP with {pert_type} Perturbation - {item_id}",
                    save_path=os.path.join(pert_output_dir, f"{item_id}_bvp_confidence.png")
                )
            
            # Plot HR distributions for each window
            for j, window in enumerate(result['windows']):
                hr_method_key = [k for k in window.keys() if k.startswith('perturbed_hr_')][0]
                if hr_method_key in window and window[hr_method_key]:
                    plot_hr_distribution(
                        window[hr_method_key],
                        mean_hr=window.get('hr_pred'),
                        gt_hr=window.get('hr_label'),
                        title=f"HR Distribution with {pert_type} Perturbation - {item_id}, Window {j}",
                        save_path=os.path.join(pert_output_dir, f"{item_id}_window_{j}_hr_dist.png")
                    )
    
    # Generate comparative visualizations
    print("\nGenerating comparative visualizations...")
    
    # Create directory for comparative visualizations
    compare_dir = os.path.join(output_dir, "comparison")
    os.makedirs(compare_dir, exist_ok=True)
    
    # Find all unique item IDs across all results
    all_item_ids = set()
    for results in all_results.values():
        for result in results:
            all_item_ids.add(result['item_id'])
    
    # Generate comparison plots for each item
    for item_id in all_item_ids:
        # Determine max number of windows for this item across all perturbation types
        max_windows = 0
        for results in all_results.values():
            for result in results:
                if result['item_id'] == item_id:
                    max_windows = max(max_windows, len(result['windows']))
                    break
        
        # Generate comparison plots for each window
        for window_idx in range(max_windows):
            try:
                # Plot perturbation comparison
                plot_perturbation_comparison(
                    all_results,
                    item_id,
                    window_idx=window_idx,
                    save_path=os.path.join(compare_dir, f"{item_id}_window_{window_idx}_comparison.png")
                )
            except Exception as e:
                print(f"Error generating comparison plot for {item_id}, window {window_idx}: {e}")
    
    # Generate uncertainty vs error plot across all results
    try:
        # Flatten results list
        all_items = []
        for pert_type, results in all_results.items():
            for result in results:
                result['perturbation_type'] = pert_type  # Ensure perturbation type is included
                all_items.append(result)
        
        # Generate plot for different windows
        for window_idx in range(3):  # First few windows
            plot_uncertainty_vs_error(
                all_items,
                window_idx=window_idx,
                save_path=os.path.join(compare_dir, f"uncertainty_vs_error_window_{window_idx}.png")
            )
    except Exception as e:
        print(f"Error generating uncertainty vs error plot: {e}")
    
    print("\nAll experiments completed!")
    return all_results

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
        "gaussian_noise": {"noise_std_fraction": 0.02},
        "blur": {"kernel_size": 3},
        "brightness": {"brightness_factor": 0.1},
        "crop": {"crop_fraction": 0.1},
        "rotation": {"angle_range": 5},
        "color_jitter": {"factor": 0.1},
        "compression": {"quality": 80}
    }
    
    # Run experiments
    run_experiment(args.config, perturbation_configs, args.output) 
