"""Unsupervised learning methods including POS, GREEN, CHROME, ICA, LGI and PBV."""
import numpy as np
from evaluation.post_process import *
from unsupervised_methods.methods.CHROME_DEHAAN import *
from unsupervised_methods.methods.GREEN import *
from unsupervised_methods.methods.ICA_POH import *
from unsupervised_methods.methods.LGI import *
from unsupervised_methods.methods.PBV import *
from unsupervised_methods.methods.POS_WANG import *
from unsupervised_methods.methods.OMIT import *
from tqdm import tqdm
from evaluation.BlandAltmanPy import BlandAltman

def unsupervised_predict(config, data_loader, method_name):
    """ Model evaluation on the testing dataset."""
    if data_loader["unsupervised"] is None:
        raise ValueError("No data for unsupervised method predicting")
    print("===Unsupervised Method ( " + method_name + " ) Predicting ===")
    predict_hr_peak_all = []
    gt_hr_peak_all = []
    predict_hr_fft_all = []
    gt_hr_fft_all = []
    SNR_all = []
    MACC_all = []
    # Store uncertainty specific data if applicable
    uncertainty_data = None
    if method_name == "CHROM":
        uncertainty_data = {
            'confidence_bands': [], 
            'perturbed_hr_fft': [],  # Store FFT HRs from perturbed signals per window
            'perturbed_hr_peak': [] # Store Peak HRs from perturbed signals per window
            }
    
    sbar = tqdm(data_loader["unsupervised"], ncols=80)
    for batch_idx, test_batch in enumerate(sbar):
        batch_size = test_batch[0].shape[0]
        for idx in range(batch_size):
            data_input, labels_input = test_batch[0][idx].cpu().numpy(), test_batch[1][idx].cpu().numpy()
            data_input = data_input[..., :3]
            BVP = None # Initialize BVP
            all_perturbed_bvps = None # Initialize storage for perturbed signals for this item
            
            if method_name == "POS":
                BVP = POS_WANG(data_input, config.UNSUPERVISED.DATA.FS)
            elif method_name == "CHROM":
                # Pass perturbation parameters from config if available, otherwise use defaults
                n_perturbations = config.UNSUPERVISED.CHROM_PERTURBATIONS.N_PERTURBATIONS if hasattr(config.UNSUPERVISED, 'CHROM_PERTURBATIONS') else 10
                noise_std_fraction = config.UNSUPERVISED.CHROM_PERTURBATIONS.NOISE_STD_FRACTION if hasattr(config.UNSUPERVISED, 'CHROM_PERTURBATIONS') else 0.01
                
                result = CHROME_DEHAAN(data_input, config.UNSUPERVISED.DATA.FS,
                                       n_perturbations=n_perturbations, 
                                       noise_std_fraction=noise_std_fraction)
                BVP = result['BVP']
                all_perturbed_bvps = result.get('perturbed_signals') # Get the perturbed signals
                if uncertainty_data is not None:
                    if 'confidence_bands' in result:
                         uncertainty_data['confidence_bands'].append(result['confidence_bands'])
                    # Initialize lists for perturbed HRs for this specific video item
                    uncertainty_data['perturbed_hr_fft'].append([]) 
                    uncertainty_data['perturbed_hr_peak'].append([])
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
                raise ValueError("unsupervised method name wrong!")

            # Check if BVP calculation was successful
            if BVP is None or BVP.size == 0:
                 print(f"Warning: BVP signal for batch index {idx} is empty. Skipping metrics calculation for this item.")
                 continue # Skip to the next item in the batch

            video_frame_size = test_batch[0].shape[1]
            # Determine window size for evaluation
            if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
                # Ensure FS is treated as float for calculation
                fs = float(config.UNSUPERVISED.DATA.FS)
                window_frame_size = int(config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * fs)
                # Ensure window_frame_size is not larger than the shortest relevant signal
                max_possible_window = min(len(BVP), len(labels_input), video_frame_size)
                if window_frame_size <= 0: # Handle zero or negative window size
                     print(f"Warning: Calculated window_frame_size is non-positive ({window_frame_size}). Using full video length.")
                     window_frame_size = max_possible_window
                elif window_frame_size > max_possible_window:
                     print(f"Warning: Configured window size ({window_frame_size}) exceeds available data length ({max_possible_window}). Using available length.")
                     window_frame_size = max_possible_window
            else:
                window_frame_size = min(len(BVP), len(labels_input), video_frame_size) # Use the length of the shortest signal

            # Check if window_frame_size is valid before proceeding
            if window_frame_size <= 0:
                print(f"Warning: Final window_frame_size is non-positive ({window_frame_size}). Skipping metrics for this item.")
                continue

            # Calculate metrics for each window
            current_item_perturbed_fft_hrs = []
            current_item_perturbed_peak_hrs = []
            for i in range(0, len(BVP), window_frame_size):
                # Define window end, ensuring it doesn't exceed BVP length
                window_end = i + window_frame_size
                if window_end > len(BVP):
                    # If the remaining part is too small, potentially skip or adjust
                    # For simplicity, we break if the last window would be full size or larger
                    # This avoids processing very small trailing segments if len(BVP) is not a multiple of window_frame_size
                     break 
                
                # Ensure indices are within bounds for all signals
                if window_end > len(labels_input):
                     print(f"Warning: Window end {window_end} exceeds label length {len(labels_input)}. Skipping window.")
                     continue # Or adjust window_end = len(labels_input) if partial window processing is desired

                BVP_window = BVP[i:window_end]
                gt_window = labels_input[i:window_end]
                
                # Ensure windows are not empty before calculating metrics
                if BVP_window.size == 0 or gt_window.size == 0:
                    print(f"Warning: Empty window encountered at index {i}. Skipping metrics calculation.")
                    continue

                hr_label = None # Initialize hr_label
                hr_pred_mean = None # HR from the mean BVP
                perturbed_hrs_fft_window = []
                perturbed_hrs_peak_window = []
                
                # --- Calculate HR for the MEAN BVP signal --- 
                if config.INFERENCE.EVALUATION_METHOD == "FFT":
                    try:
                        hr_pred_mean = _calculate_fft_hr(BVP_window, fs=config.UNSUPERVISED.DATA.FS)
                        hr_label = _calculate_fft_hr(gt_window, fs=config.UNSUPERVISED.DATA.FS)
                        predict_hr_fft_all.append(hr_pred_mean)
                        gt_hr_fft_all.append(hr_label)
                    except Exception as e:
                        print(f"Error calculating Mean FFT HR for window {i}: {e}. Skipping.")
                        # Skip this window for mean metrics if calculation fails
                        hr_pred_mean = None 
                        hr_label = None
                        # We might still proceed to calculate perturbed HRs below if needed
                        # continue 
                elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
                    try:
                        hr_pred_mean = _calculate_peak_hr(BVP_window, fs=config.UNSUPERVISED.DATA.FS)
                        hr_label = _calculate_peak_hr(gt_window, fs=config.UNSUPERVISED.DATA.FS)
                        predict_hr_peak_all.append(hr_pred_mean)
                        gt_hr_peak_all.append(hr_label)
                    except Exception as e:
                        print(f"Error calculating Mean Peak HR for window {i}: {e}. Skipping.")
                        hr_pred_mean = None
                        hr_label = None
                        # continue
                else:
                     print(f"Warning: Unknown evaluation method '{config.INFERENCE.EVALUATION_METHOD}'. Skipping HR calculation.")
                     hr_pred_mean = None
                     hr_label = None
                     # continue
                     
                # --- Calculate HR for EACH PERTURBED BVP signal (if available) --- 
                if method_name == "CHROM" and all_perturbed_bvps is not None:
                    for perturbed_signal in all_perturbed_bvps:
                        if window_end > len(perturbed_signal):
                             # Should ideally not happen if length normalization worked, but check anyway
                             print(f"Warning: Window end {window_end} exceeds perturbed signal length {len(perturbed_signal)}. Skipping HR for this perturbation.")
                             continue
                             
                        perturbed_bvp_window = perturbed_signal[i:window_end]
                        if perturbed_bvp_window.size == 0:
                            continue # Skip empty perturbed windows

                        if config.INFERENCE.EVALUATION_METHOD == "FFT":
                            try:
                                hr_perturbed = _calculate_fft_hr(perturbed_bvp_window, fs=config.UNSUPERVISED.DATA.FS)
                                perturbed_hrs_fft_window.append(hr_perturbed)
                            except Exception as e:
                                # print(f"Error calculating Perturbed FFT HR for window {i}: {e}. Skipping perturbation.")
                                perturbed_hrs_fft_window.append(np.nan) # Append NaN on error
                        elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
                            try:
                                hr_perturbed = _calculate_peak_hr(perturbed_bvp_window, fs=config.UNSUPERVISED.DATA.FS)
                                perturbed_hrs_peak_window.append(hr_perturbed)
                            except Exception as e:
                                # print(f"Error calculating Perturbed Peak HR for window {i}: {e}. Skipping perturbation.")
                                perturbed_hrs_peak_window.append(np.nan) # Append NaN on error
                                
                # Append lists of perturbed HRs for this window to the item's list
                if uncertainty_data is not None:
                     # The index corresponds to the current video item (batch_idx * batch_size + idx might be safer if batching logic complex) 
                     # Let's assume idx is the correct index within the current uncertainty_data lists append
                     item_index = len(uncertainty_data['perturbed_hr_fft']) - 1 # Index of the list for the current item
                     if item_index >= 0:
                         if perturbed_hrs_fft_window:
                              uncertainty_data['perturbed_hr_fft'][item_index].append(perturbed_hrs_fft_window)
                         if perturbed_hrs_peak_window:
                              uncertainty_data['perturbed_hr_peak'][item_index].append(perturbed_hrs_peak_window)

                # --- Calculate SNR and MACC using the MEAN BVP --- 
                if hr_label is not None and hr_pred_mean is not None: # Use mean HR prediction for consistency? Or hr_label?
                     try:
                         # SNR typically calculated using the predicted signal and ground truth HR
                         SNR = _calculate_SNR(BVP_window, hr_label, fs=config.UNSUPERVISED.DATA.FS)
                         SNR_all.append(SNR)
                     except Exception as e:
                         print(f"Error calculating SNR for window {i}: {e}. Skipping SNR.")
                         SNR_all.append(np.nan) 
                         
                     try:
                         # MACC compares the predicted BVP waveform to the ground truth waveform
                         MACC = _compute_macc(BVP_window, gt_window)
                         MACC_all.append(MACC)
                     except Exception as e:
                         print(f"Error calculating MACC for window {i}: {e}. Skipping MACC.")
                         MACC_all.append(np.nan) 
                else:
                     SNR_all.append(np.nan)
                     MACC_all.append(np.nan)

    results = {
        'predict_hr_peak': np.array(predict_hr_peak_all),
        'gt_hr_peak': np.array(gt_hr_peak_all),
        'predict_hr_fft': np.array(predict_hr_fft_all),
        'gt_hr_fft': np.array(gt_hr_fft_all),
        'SNR': np.array(SNR_all),
        'MACC': np.array(MACC_all)
    }
    
    # Add uncertainty data if it was generated
    if uncertainty_data is not None:
        # Make sure keys exist before updating
        if 'confidence_bands' in uncertainty_data:
             results['confidence_bands'] = np.array(uncertainty_data['confidence_bands'])
        if 'perturbed_hr_fft' in uncertainty_data:
             # This will be a list of lists of lists (item -> window -> perturbation HRs)
             # Keeping it as a list might be more flexible than forcing into numpy array
             results['perturbed_hr_fft'] = uncertainty_data['perturbed_hr_fft'] 
        if 'perturbed_hr_peak' in uncertainty_data:
             results['perturbed_hr_peak'] = uncertainty_data['perturbed_hr_peak']
             
    return results
