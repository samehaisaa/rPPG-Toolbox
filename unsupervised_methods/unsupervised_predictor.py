"""Unsupervised learning methods including POS, GREEN, CHROME, ICA, LGI and PBV."""
import numpy as np
# from evaluation.post_process import *  # Change from wildcard import
from evaluation.post_process import _calculate_fft_hr, _calculate_peak_hr, _calculate_SNR, _compute_macc # Explicit imports
from unsupervised_methods.methods.CHROME_DEHAAN import *
from unsupervised_methods.methods.GREEN import *
from unsupervised_methods.methods.ICA_POH import *
from unsupervised_methods.methods.LGI import *
from unsupervised_methods.methods.PBV import *
from unsupervised_methods.methods.POS_WANG import *
from unsupervised_methods.methods.OMIT import *
from tqdm import tqdm
# from evaluation.BlandAltmanPy import BlandAltman # This seems unused now

def unsupervised_predict(config, data_loader, method_name):
    """ Model evaluation on the testing dataset.
        Returns a list of dictionaries, one per processed video item.
    """
    if data_loader["unsupervised"] is None:
        raise ValueError("No data for unsupervised method predicting")
    print("===Unsupervised Method ( " + method_name + " ) Predicting ===")
    
    all_item_results = [] # List to store results for each item
    
    sbar = tqdm(data_loader["unsupervised"], ncols=80)
    for batch_idx, test_batch in enumerate(sbar):
        batch_size = test_batch[0].shape[0]
        # Assuming batch_size is always 1 for unsupervised for simplicity now
        if batch_size > 1:
             print("Warning: Batch size > 1 detected in unsupervised predictor. Processing first item only for now.")
             # Or add logic here to loop through items in the batch properly

        for idx in range(batch_size): # Loop through items in batch (often just 1)
            data_input, labels_input = test_batch[0][idx].cpu().numpy(), test_batch[1][idx].cpu().numpy()
            data_input = data_input[..., :3]
            BVP = None # Initialize BVP
            all_perturbed_bvps = None # Initialize storage for perturbed signals for this item
            confidence_bands_item = None # For CHROM
            
            # --- Run selected rPPG method --- 
            try:
                if method_name == "POS":
                    BVP = POS_WANG(data_input, config.UNSUPERVISED.DATA.FS)
                elif method_name == "CHROM":
                    n_perturbations = config.UNSUPERVISED.CHROM_PERTURBATIONS.N_PERTURBATIONS if hasattr(config.UNSUPERVISED, 'CHROM_PERTURBATIONS') else 10
                    noise_std_fraction = config.UNSUPERVISED.CHROM_PERTURBATIONS.NOISE_STD_FRACTION if hasattr(config.UNSUPERVISED, 'CHROM_PERTURBATIONS') else 0.01
                    result = CHROME_DEHAAN(data_input, config.UNSUPERVISED.DATA.FS,
                                           n_perturbations=n_perturbations, 
                                           noise_std_fraction=noise_std_fraction)
                    BVP = result['BVP']
                    all_perturbed_bvps = result.get('perturbed_signals') 
                    confidence_bands_item = result.get('confidence_bands')
                elif method_name == "ICA":
                    BVP = ICA_POH(data_input, config.UNSUPERVISED.DATA.FS)
                elif method_name == "GREEN":
                    BVP = GREEN(data_input)
                elif method_name == "LGI":
                    BVP = LGI(data_input)
                elif method_name == "PBV":
                    BVP = PBV(data_input)
                elif method_name == "OMIT":
                    BVP = OMIT(data_input)
                else:
                    raise ValueError(f"unsupervised method name '{method_name}' wrong!")
            except Exception as e:
                 print(f"Error running {method_name} for batch {batch_idx} item {idx}: {e}")
                 BVP = None # Ensure BVP is None on error
                 
            # Check if BVP calculation was successful
            if BVP is None or BVP.size == 0:
                 print(f"Warning: BVP signal for batch {batch_idx} item {idx} is empty or failed. Skipping metrics calculation.")
                 # Add an empty result entry or skip entirely? Let's skip for now.
                 continue 

            # --- Prepare result structure for this item --- 
            item_result = {
                'id': f'batch_{batch_idx}_item_{idx}', 
                'method': method_name,
                'mean_bvp': BVP, # Store the calculated (mean) BVP
                'windows': [] # List to store results per window
            }
            if confidence_bands_item is not None:
                 item_result['confidence_bands'] = confidence_bands_item
                 
            # --- Calculate window size --- 
            video_frame_size = test_batch[0].shape[1]
            if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
                fs = float(config.UNSUPERVISED.DATA.FS)
                window_frame_size = int(config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * fs)
                max_possible_window = min(len(BVP), len(labels_input), video_frame_size)
                if window_frame_size <= 0: 
                     window_frame_size = max_possible_window
                elif window_frame_size > max_possible_window:
                     window_frame_size = max_possible_window
            else:
                window_frame_size = min(len(BVP), len(labels_input), video_frame_size) 

            if window_frame_size <= 0:
                print(f"Warning: Final window_frame_size is non-positive ({window_frame_size}) for item {item_result['id']}. Skipping metrics.")
                # Add item_result with empty windows list, or skip?
                all_item_results.append(item_result) # Add with empty windows
                continue

            # --- Process windows --- 
            num_windows = 0
            for i in range(0, len(BVP), window_frame_size):
                window_end = i + window_frame_size
                if window_end > len(BVP): break 
                if window_end > len(labels_input):
                     print(f"Warning: Window end {window_end} exceeds label length {len(labels_input)} for item {item_result['id']}. Skipping window.")
                     continue 

                BVP_window = BVP[i:window_end]
                gt_window = labels_input[i:window_end]
                
                if BVP_window.size == 0 or gt_window.size == 0:
                    print(f"Warning: Empty window encountered at index {i} for item {item_result['id']}. Skipping window.")
                    continue
                
                window_result = { 'window_index': num_windows } # Store results for this window
                num_windows += 1
                hr_label = None 
                hr_pred_mean = None 
                perturbed_hrs_fft_window = []
                perturbed_hrs_peak_window = []
                
                eval_method = config.INFERENCE.EVALUATION_METHOD
                hr_func = _calculate_fft_hr if eval_method == "FFT" else (_calculate_peak_hr if eval_method == "peak detection" else None)
                hr_list_pred = predict_hr_fft_all if eval_method == "FFT" else (predict_hr_peak_all if eval_method == "peak detection" else None)
                hr_list_gt = gt_hr_fft_all if eval_method == "FFT" else (gt_hr_peak_all if eval_method == "peak detection" else None)
                pert_hr_list_key = 'perturbed_hr_fft' if eval_method == "FFT" else ('perturbed_hr_peak' if eval_method == "peak detection" else None)
                pert_hr_window_list = perturbed_hrs_fft_window if eval_method == "FFT" else (perturbed_hrs_peak_window if eval_method == "peak detection" else None)

                # Calculate HR for Mean BVP
                if hr_func:
                    try:
                        hr_pred_mean = hr_func(BVP_window, fs=config.UNSUPERVISED.DATA.FS)
                        hr_label = hr_func(gt_window, fs=config.UNSUPERVISED.DATA.FS)
                        window_result['hr_pred'] = hr_pred_mean
                        window_result['hr_label'] = hr_label
                    except Exception as e:
                        # print(f"Error calculating Mean HR ({eval_method}) for window {i}: {e}")
                        window_result['hr_pred'] = np.nan
                        window_result['hr_label'] = np.nan
                        hr_pred_mean = None; hr_label = None # Ensure they are None for later checks
                else:
                    print(f"Warning: Unknown evaluation method '{eval_method}'. Skipping HR calculation.")
                    window_result['hr_pred'] = np.nan
                    window_result['hr_label'] = np.nan

                # Calculate HR for Perturbed BVPs (if CHROM)
                if method_name == "CHROM" and all_perturbed_bvps is not None and hr_func:
                    current_pert_hrs = []
                    for perturbed_signal in all_perturbed_bvps:
                        if window_end <= len(perturbed_signal):
                            perturbed_bvp_window = perturbed_signal[i:window_end]
                            if perturbed_bvp_window.size > 0:
                                try:
                                    hr_perturbed = hr_func(perturbed_bvp_window, fs=config.UNSUPERVISED.DATA.FS)
                                    current_pert_hrs.append(hr_perturbed)
                                except Exception as e:
                                    # print(f"Error calculating Perturbed HR ({eval_method}) for window {i}: {e}")
                                    current_pert_hrs.append(np.nan) 
                            else:
                                current_pert_hrs.append(np.nan) # Append NaN for empty perturbed window
                        else:
                            current_pert_hrs.append(np.nan) # Append NaN if window exceeds perturbed signal
                    if pert_hr_list_key:
                         window_result[pert_hr_list_key] = current_pert_hrs

                # Calculate SNR and MACC using Mean BVP
                if hr_label is not None: # Requires GT HR
                     try:
                         SNR = _calculate_SNR(BVP_window, hr_label, fs=config.UNSUPERVISED.DATA.FS)
                         window_result['snr'] = SNR
                     except Exception as e:
                         # print(f"Error calculating SNR for window {i}: {e}")
                         window_result['snr'] = np.nan
                         
                     try:
                         MACC = _compute_macc(BVP_window, gt_window)
                         window_result['macc'] = MACC
                     except Exception as e:
                         # print(f"Error calculating MACC for window {i}: {e}")
                         window_result['macc'] = np.nan
                else:
                     window_result['snr'] = np.nan
                     window_result['macc'] = np.nan
                     
                item_result['windows'].append(window_result)
            
            # Append the results for this item to the main list
            all_item_results.append(item_result)
            
            # End of loop for items in batch (idx)
        # End of loop for batches (batch_idx)
        
    return all_item_results # Return the list of dictionaries
