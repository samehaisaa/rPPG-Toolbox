""" The main function of rPPG deep learning pipeline."""

import argparse
import random
import time
import os

import numpy as np
import torch
from config import get_config
from dataset import data_loader
from neural_methods import trainer
from unsupervised_methods.unsupervised_predictor import unsupervised_predict
from torch.utils.data import DataLoader
import neural_methods.trainer.PhysFormerTrainer
from visualization import plot_bvp_with_confidence, plot_hr_distribution
RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Create a general generator for use with the validation dataloader,
# the test dataloader, and the unsupervised dataloader
general_generator = torch.Generator()
general_generator.manual_seed(RANDOM_SEED)
# Create a training generator to isolate the train dataloader from
# other dataloaders and better control non-deterministic behavior
train_generator = torch.Generator()
train_generator.manual_seed(RANDOM_SEED)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def add_args(parser):
    """Adds arguments for parser."""
    parser.add_argument('--config_file', required=False,
                        default="configs/train_configs/PURE_PURE_UBFC-rPPG_TSCAN_BASIC.yaml", type=str, help="The name of the model.")
    '''Neural Method Sample YAML LIST:
      SCAMPS_SCAMPS_UBFC-rPPG_TSCAN_BASIC.yaml
      SCAMPS_SCAMPS_UBFC-rPPG_DEEPPHYS_BASIC.yaml
      SCAMPS_SCAMPS_UBFC-rPPG_PHYSNET_BASIC.yaml
      SCAMPS_SCAMPS_PURE_DEEPPHYS_BASIC.yaml
      SCAMPS_SCAMPS_PURE_TSCAN_BASIC.yaml
      SCAMPS_SCAMPS_PURE_PHYSNET_BASIC.yaml
      PURE_PURE_UBFC-rPPG_TSCAN_BASIC.yaml
      PURE_PURE_UBFC-rPPG_DEEPPHYS_BASIC.yaml
      PURE_PURE_UBFC-rPPG_PHYSNET_BASIC.yaml
      PURE_PURE_MMPD_TSCAN_BASIC.yaml
      UBFC-rPPG_UBFC-rPPG_PURE_TSCAN_BASIC.yaml
      UBFC-rPPG_UBFC-rPPG_PURE_DEEPPHYS_BASIC.yaml
      UBFC-rPPG_UBFC-rPPG_PURE_PHYSNET_BASIC.yaml
      MMPD_MMPD_UBFC-rPPG_TSCAN_BASIC.yaml
    Unsupervised Method Sample YAML LIST:
      PURE_UNSUPERVISED.yaml
      UBFC-rPPG_UNSUPERVISED.yaml
    '''
    return parser


def train_and_test(config, data_loader_dict):
    """Trains the model."""
    if config.MODEL.NAME == "Physnet":
        model_trainer = trainer.PhysnetTrainer.PhysnetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "iBVPNet":
        model_trainer = trainer.iBVPNetTrainer.iBVPNetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "FactorizePhys":
        model_trainer = trainer.FactorizePhysTrainer.FactorizePhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "Tscan":
        model_trainer = trainer.TscanTrainer.TscanTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "EfficientPhys":
        model_trainer = trainer.EfficientPhysTrainer.EfficientPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'DeepPhys':
        model_trainer = trainer.DeepPhysTrainer.DeepPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'BigSmall':
        model_trainer = trainer.BigSmallTrainer.BigSmallTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'PhysFormer':
        model_trainer = trainer.PhysFormerTrainer.PhysFormerTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'PhysMamba':
        model_trainer = trainer.PhysMambaTrainer.PhysMambaTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'RhythmFormer':
        model_trainer = trainer.RhythmFormerTrainer.RhythmFormerTrainer(config, data_loader_dict)
    else:
        raise ValueError('Your Model is Not Supported  Yet!')
    model_trainer.train(data_loader_dict)
    model_trainer.test(data_loader_dict)


def test(config, data_loader_dict):
    """Tests the model."""
    if config.MODEL.NAME == "Physnet":
        model_trainer = trainer.PhysnetTrainer.PhysnetTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "iBVPNet":
        model_trainer = trainer.iBVPNetTrainer.iBVPNetTrainer(config, data_loader_dict)    
    elif config.MODEL.NAME == "FactorizePhys":
        model_trainer = trainer.FactorizePhysTrainer.FactorizePhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "Tscan":
        model_trainer = trainer.TscanTrainer.TscanTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "EfficientPhys":
        model_trainer = trainer.EfficientPhysTrainer.EfficientPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'DeepPhys':
        model_trainer = trainer.DeepPhysTrainer.DeepPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'BigSmall':
        model_trainer = trainer.BigSmallTrainer.BigSmallTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'PhysFormer':
        model_trainer = trainer.PhysFormerTrainer.PhysFormerTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'PhysMamba':
        model_trainer = trainer.PhysMambaTrainer.PhysMambaTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == 'RhythmFormer':
        model_trainer = trainer.RhythmFormerTrainer.RhythmFormerTrainer(config, data_loader_dict)
    else:
        raise ValueError('Your Model is Not Supported  Yet!')
    model_trainer.test(data_loader_dict)


def unsupervised_method_inference(config, data_loader):
    if not config.UNSUPERVISED.METHOD:
        raise ValueError("Please set unsupervised method in yaml!")
    for unsupervised_method in config.UNSUPERVISED.METHOD:
        if unsupervised_method == "POS":
            unsupervised_predict(config, data_loader, "POS")
        elif unsupervised_method == "CHROM":
            unsupervised_predict(config, data_loader, "CHROM")
        elif unsupervised_method == "ICA":
            unsupervised_predict(config, data_loader, "ICA")
        elif unsupervised_method == "GREEN":
            unsupervised_predict(config, data_loader, "GREEN")
        elif unsupervised_method == "LGI":
            unsupervised_predict(config, data_loader, "LGI")
        elif unsupervised_method == "PBV":
            unsupervised_predict(config, data_loader, "PBV")
        elif unsupervised_method == "OMIT":
            unsupervised_predict(config, data_loader, "OMIT")
        else:
            raise ValueError("Not supported unsupervised method!")


def main(config, data_loader_dict):
    # ... setup code (logging, device, etc.) ...

    # Set random seeds for reproducibility
    if config.TOOLBOX_MODE == "train_and_test":
        # ... training logic ...
        pass # Placeholder
    elif config.TOOLBOX_MODE == "only_test":
        # ... testing logic for supervised models ...
        pass # Placeholder
    elif config.TOOLBOX_MODE == "unsupervised_method":
        # === Load data ===
        # data_loader_dict is now passed as an argument, no need to recreate it here
        # data_loader_dict = data_loader.unsupervised_data_loader(config=config)
        
        # === Run prediction === 
        # Assuming unsupervised_predict is called for each method in config.UNSUPERVISED.METHOD
        # We focus on the CHROM case here
        method_name = "CHROM" # Or get from loop/config
        if method_name in config.UNSUPERVISED.METHOD:
            print(f"Running prediction for: {method_name}")
            results = unsupervised_predict(config, data_loader_dict, method_name)
            print(f"Prediction finished for: {method_name}")
            
            # --- Visualization --- 
            # Ensure results is a list (as expected from refactored predictor)
            if isinstance(results, list) and method_name == "CHROM":
                print("Generating visualizations for CHROM perturbation results...")
                # Example: Visualize results for the first video item (index 0)
                item_index_to_plot = 0
                fs = float(config.UNSUPERVISED.DATA.FS)
                output_dir = os.path.join(config.LOG.PATH, "visualizations", method_name)
                os.makedirs(output_dir, exist_ok=True)
                
                # Check if results exist for the selected item index
                if item_index_to_plot < len(results):
                    item_data = results[item_index_to_plot]
                    item_id = item_data.get('id', f'item_{item_index_to_plot}') # Get item ID if available
                    
                    # Plot BVP with confidence bands if available
                    mean_bvp_item = item_data.get('mean_bvp')
                    conf_bands_item = item_data.get('confidence_bands')
                    if mean_bvp_item is not None and conf_bands_item is not None:
                        plot_bvp_with_confidence(mean_bvp_item, conf_bands_item, fs,
                                                 title=f"CHROM BVP - {item_id}",
                                                 save_path=os.path.join(output_dir, f"{item_id}_bvp_confidence.png"))
                    else:
                         print(f"Missing mean BVP or confidence bands for {item_id}. Cannot plot BVP confidence.")
                
                    # Plot HR distribution for the first few windows 
                    hr_method_key = 'perturbed_hr_fft' if config.INFERENCE.EVALUATION_METHOD == "FFT" else ('perturbed_hr_peak' if config.INFERENCE.EVALUATION_METHOD == "peak detection" else None)
                    mean_hr_key = 'hr_pred' # Key used within window_result
                    gt_hr_key = 'hr_label'  # Key used within window_result
                    
                    windows_data = item_data.get('windows', [])
                    if not windows_data:
                         print(f"No window data found for {item_id}. Cannot plot HR distributions.")
                    else:
                         num_windows_to_plot = min(3, len(windows_data)) # Plot first 3 windows
                         print(f"Plotting HR distributions for first {num_windows_to_plot} windows of {item_id}...")
                         for window_idx in range(num_windows_to_plot):
                            window_result = windows_data[window_idx]
                            perturbed_hrs_window = window_result.get(hr_method_key)
                            mean_hr_window = window_result.get(mean_hr_key)
                            gt_hr_window = window_result.get(gt_hr_key)
                            
                            if perturbed_hrs_window:
                                plot_hr_distribution(perturbed_hrs_window, 
                                                     mean_hr=mean_hr_window, 
                                                     gt_hr=gt_hr_window, 
                                                     title=f"CHROM HR Distribution - {item_id}, Window {window_idx}",
                                                     save_path=os.path.join(output_dir, f"{item_id}_window_{window_idx}_hr_dist.png"))
                            else:
                                 print(f"No perturbed HR data ('{hr_method_key}') found for {item_id}, Window {window_idx}.")
                else:
                     print(f"Item index {item_index_to_plot} out of range for visualization (results length: {len(results)}).")
            elif not isinstance(results, list):
                 print(f"Warning: Expected list results from unsupervised_predict for {method_name}, but got {type(results)}. Skipping visualization.")
                 
            # --- Aggregate and Print Overall Metrics (Optional) --- 
            # You might want to add logic here to loop through the structured 'results'
            # list, extract all window-level metrics (hr_pred, hr_label, snr, macc),
            # and calculate overall MAE, RMSE, Pearson, etc., similar to how it 
            # might have been done before the refactoring. 
            # Example placeholder:
            # all_pred_hrs = [w['hr_pred'] for item in results for w in item['windows']] 
            # all_gt_hrs = [w['hr_label'] for item in results for w in item['windows']]
            # calculate_overall_metrics(all_pred_hrs, all_gt_hrs, config) # Fictional function
            print("Placeholder: Aggregated metric calculation would go here.")
            
        else:
            print(f"Skipping prediction for {method_name} as it's not in config.UNSUPERVISED.METHOD")
            
    else:
        raise ValueError("Unsupported TOOLBOX_MODE!")


if __name__ == "__main__":
    # parse arguments.
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = trainer.BaseTrainer.BaseTrainer.add_trainer_args(parser)
    parser = data_loader.BaseLoader.BaseLoader.add_data_loader_args(parser)
    args = parser.parse_args()

    # configurations.
    config = get_config(args)
    print('Configuration:')
    # Print the configuration content (optional)
    # print(config)
    # print("-----------------------------------------------------")

    data_loader_dict = dict() # dictionary of data loaders 
    if config.TOOLBOX_MODE == "train_and_test":
        # train_loader
        if config.TRAIN.DATA.DATASET == "UBFC-rPPG":
            train_loader = data_loader.UBFCrPPGLoader.UBFCrPPGLoader
        elif config.TRAIN.DATA.DATASET == "PURE":
            train_loader = data_loader.PURELoader.PURELoader
        elif config.TRAIN.DATA.DATASET == "SCAMPS":
            train_loader = data_loader.SCAMPSLoader.SCAMPSLoader
        elif config.TRAIN.DATA.DATASET == "MMPD":
            train_loader = data_loader.MMPDLoader.MMPDLoader
        elif config.TRAIN.DATA.DATASET == "BP4DPlus":
            train_loader = data_loader.BP4DPlusLoader.BP4DPlusLoader
        elif config.TRAIN.DATA.DATASET == "BP4DPlusBigSmall":
            train_loader = data_loader.BP4DPlusBigSmallLoader.BP4DPlusBigSmallLoader
        elif config.TRAIN.DATA.DATASET == "UBFC-PHYS":
            train_loader = data_loader.UBFCPHYSLoader.UBFCPHYSLoader
        elif config.TRAIN.DATA.DATASET == "iBVP":
            train_loader = data_loader.iBVPLoader.iBVPLoader
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC-rPPG, PURE, MMPD, SCAMPS, BP4D+ (Normal and BigSmall preprocessing), UBFC-PHYS and iBVP.")

        # Create and initialize the train dataloader given the correct toolbox mode,
        # a supported dataset name, and a valid dataset paths
        if (config.TRAIN.DATA.DATASET and config.TRAIN.DATA.DATA_PATH):

            train_data_loader = train_loader(
                name="train",
                data_path=config.TRAIN.DATA.DATA_PATH,
                config_data=config.TRAIN.DATA,
                device=config.DEVICE)
            data_loader_dict['train'] = DataLoader(
                dataset=train_data_loader,
                num_workers=4,
                batch_size=config.TRAIN.BATCH_SIZE,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=train_generator
            )
        else:
            data_loader_dict['train'] = None

        # valid_loader
        if config.VALID.DATA.DATASET == "UBFC-rPPG":
            valid_loader = data_loader.UBFCrPPGLoader.UBFCrPPGLoader
        elif config.VALID.DATA.DATASET == "PURE":
            valid_loader = data_loader.PURELoader.PURELoader
        elif config.VALID.DATA.DATASET == "SCAMPS":
            valid_loader = data_loader.SCAMPSLoader.SCAMPSLoader
        elif config.VALID.DATA.DATASET == "MMPD":
            valid_loader = data_loader.MMPDLoader.MMPDLoader
        elif config.VALID.DATA.DATASET == "BP4DPlus":
            valid_loader = data_loader.BP4DPlusLoader.BP4DPlusLoader
        elif config.VALID.DATA.DATASET == "BP4DPlusBigSmall":
            valid_loader = data_loader.BP4DPlusBigSmallLoader.BP4DPlusBigSmallLoader
        elif config.VALID.DATA.DATASET == "UBFC-PHYS":
            valid_loader = data_loader.UBFCPHYSLoader.UBFCPHYSLoader
        elif config.VALID.DATA.DATASET == "iBVP":
            valid_loader = data_loader.iBVPLoader.iBVPLoader
        elif config.VALID.DATA.DATASET is None and not config.TEST.USE_LAST_EPOCH:
            raise ValueError("Validation dataset not specified despite USE_LAST_EPOCH set to False!")
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC-rPPG, PURE, MMPD, SCAMPS, BP4D+ (Normal and BigSmall preprocessing), UBFC-PHYS and iBVP")
        
        # Create and initialize the valid dataloader given the correct toolbox mode,
        # a supported dataset name, and a valid dataset path
        if (config.VALID.DATA.DATASET and config.VALID.DATA.DATA_PATH and not config.TEST.USE_LAST_EPOCH):
            valid_data = valid_loader(
                name="valid",
                data_path=config.VALID.DATA.DATA_PATH,
                config_data=config.VALID.DATA,
                device=config.DEVICE)
            data_loader_dict["valid"] = DataLoader(
                dataset=valid_data,
                num_workers=4,
                batch_size=config.TRAIN.BATCH_SIZE,  # batch size for val is the same as train
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=general_generator
            )
        else:
            data_loader_dict['valid'] = None

    if config.TOOLBOX_MODE == "train_and_test" or config.TOOLBOX_MODE == "only_test":
        # test_loader
        if config.TEST.DATA.DATASET == "UBFC-rPPG":
            test_loader = data_loader.UBFCrPPGLoader.UBFCrPPGLoader
        elif config.TEST.DATA.DATASET == "PURE":
            test_loader = data_loader.PURELoader.PURELoader
        elif config.TEST.DATA.DATASET == "SCAMPS":
            test_loader = data_loader.SCAMPSLoader.SCAMPSLoader
        elif config.TEST.DATA.DATASET == "MMPD":
            test_loader = data_loader.MMPDLoader.MMPDLoader
        elif config.TEST.DATA.DATASET == "BP4DPlus":
            test_loader = data_loader.BP4DPlusLoader.BP4DPlusLoader
        elif config.TEST.DATA.DATASET == "BP4DPlusBigSmall":
            test_loader = data_loader.BP4DPlusBigSmallLoader.BP4DPlusBigSmallLoader
        elif config.TEST.DATA.DATASET == "UBFC-PHYS":
            test_loader = data_loader.UBFCPHYSLoader.UBFCPHYSLoader
        elif config.TEST.DATA.DATASET == "iBVP":
            test_loader = data_loader.iBVPLoader.iBVPLoader
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC-rPPG, PURE, MMPD, SCAMPS, BP4D+ (Normal and BigSmall preprocessing), UBFC-PHYS and iBVP.")
        
        if config.TOOLBOX_MODE == "train_and_test" and config.TEST.USE_LAST_EPOCH:
            print("Testing uses last epoch, validation dataset is not required.", end='\n\n')   

        # Create and initialize the test dataloader given the correct toolbox mode,
        # a supported dataset name, and a valid dataset path
        if config.TEST.DATA.DATASET and config.TEST.DATA.DATA_PATH:
            test_data = test_loader(
                name="test",
                data_path=config.TEST.DATA.DATA_PATH,
                config_data=config.TEST.DATA,
                device=config.DEVICE)
            data_loader_dict["test"] = DataLoader(
                dataset=test_data,
                num_workers=4,
                batch_size=config.INFERENCE.BATCH_SIZE,
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=general_generator
            )
        else:
            data_loader_dict['test'] = None

    elif config.TOOLBOX_MODE == "unsupervised_method":
        # unsupervised method dataloader
        if config.UNSUPERVISED.DATA.DATASET == "UBFC-rPPG":
            unsupervised_loader = data_loader.UBFCrPPGLoader.UBFCrPPGLoader
        elif config.UNSUPERVISED.DATA.DATASET == "PURE":
            unsupervised_loader = data_loader.PURELoader.PURELoader
        elif config.UNSUPERVISED.DATA.DATASET == "SCAMPS":
            unsupervised_loader = data_loader.SCAMPSLoader.SCAMPSLoader
        elif config.UNSUPERVISED.DATA.DATASET == "MMPD":
            unsupervised_loader = data_loader.MMPDLoader.MMPDLoader
        elif config.UNSUPERVISED.DATA.DATASET == "BP4DPlus":
            unsupervised_loader = data_loader.BP4DPlusLoader.BP4DPlusLoader
        elif config.UNSUPERVISED.DATA.DATASET == "UBFC-PHYS":
            unsupervised_loader = data_loader.UBFCPHYSLoader.UBFCPHYSLoader
        elif config.UNSUPERVISED.DATA.DATASET == "iBVP":
            unsupervised_loader = data_loader.iBVPLoader.iBVPLoader
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC-rPPG, PURE, MMPD, SCAMPS, BP4D+, UBFC-PHYS and iBVP.")
        
        # Check if data path exists for unsupervised
        if not config.UNSUPERVISED.DATA.DATA_PATH or not os.path.isdir(config.UNSUPERVISED.DATA.DATA_PATH):
             raise ValueError(f"Unsupervised data path not found or not specified: {config.UNSUPERVISED.DATA.DATA_PATH}")
             
        unsupervised_data = unsupervised_loader(
            name="unsupervised",
            data_path=config.UNSUPERVISED.DATA.DATA_PATH,
            config_data=config.UNSUPERVISED.DATA,
            device=config.DEVICE)
        data_loader_dict["unsupervised"] = DataLoader(
            dataset=unsupervised_data,
            num_workers=4,
            batch_size=1, # Typically 1 for unsupervised per-video processing
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=general_generator
        )

    else:
        raise ValueError("Unsupported toolbox_mode! Currently support train_and_test or only_test or unsupervised_method.")

    # Call the main logic after setting up data loaders
    main(config, data_loader_dict)

    # Removed recursive call: main(config)
