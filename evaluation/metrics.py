import numpy as np
import pandas as pd
import torch
from evaluation.post_process import *
from tqdm import tqdm
from evaluation.BlandAltmanPy import BlandAltman
import os
def read_label(dataset):
    """Read manually corrected labels."""
    df = pd.read_csv("label/{0}_Comparison.csv".format(dataset))
    out_dict = df.to_dict(orient='index')
    out_dict = {str(value['VideoID']): value for key, value in out_dict.items()}
    return out_dict

def plot_hr_distribution(gt_hr_fft_all, save_path, file_name="HR_Distribution.pdf", bins=30):
    """
    Génère et sauvegarde un histogramme de la distribution des HR dans gt_hr_fft_all.

    :param gt_hr_fft_all: Liste ou tableau numpy des valeurs HR (ground truth).
    :param save_path: Chemin où sauvegarder la figure.
    :param file_name: Nom du fichier de sortie.
    :param bins: Nombre de bins pour l'histogramme.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.hist(gt_hr_fft_all, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('GT PPG HR [bpm]')
    plt.ylabel('Fréquence')
    plt.title('Distribution des HR (Ground Truth)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    save_file = os.path.join(save_path, file_name)
    plt.savefig(save_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Histogramme sauvegardé sous {save_file}")


def read_hr_label(feed_dict, index):
    """Read manually corrected UBFC labels."""
    # For UBFC only
    if index[:7] == 'subject':
        index = index[7:]
    video_dict = feed_dict[index]
    if video_dict['Preferred'] == 'Peak Detection':
        hr = video_dict['Peak Detection']
    elif video_dict['Preferred'] == 'FFT':
        hr = video_dict['FFT']
    else:
        hr = video_dict['Peak Detection']
    return index, hr


def _reform_data_from_dict(data, flatten=True):
    """Helper func for calculate metrics: reformat predictions and labels from dicts. """
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)

    if flatten:
        sort_data = np.reshape(sort_data.cpu(), (-1))
    else:
        sort_data = np.array(sort_data.cpu())

    return sort_data


def calculate_metrics(predictions, labels, config):
    """Calculate rPPG Metrics (MAE, RMSE, MAPE, Pearson Coef.)."""
    predict_hr_fft_all = list()
    gt_hr_fft_all = list()
    predict_hr_peak_all = list()
    gt_hr_peak_all = list()
    SNR_all = list()
    MACC_all = list()
    print("keys of predictions: ", predictions.keys())
    print("Calculating metrics!")
    
    # Initialize counters for detailed information
    total_subjects = len(predictions.keys())
    total_windows_processed = 0
    total_windows_skipped = 0
    windows_per_subject = {}
    
    print("\n=============== WINDOW CONFIGURATION INFO ===============")
    print(f"Total number of test subjects: {total_subjects}")
    print(f"USE_SMALLER_WINDOW set to: {config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW}")
    print(f"WINDOW_SIZE set to: {config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE} seconds")
    print(f"Sampling rate (FS): {config.TEST.DATA.FS} frames per second")
    
    if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
        print(f"Window size in frames: {config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.TEST.DATA.FS} frames")
        print("Expected data points will be multiple per subject")
    else:
        print("Processing each subject as a single window - one data point per subject")
        print(f"Expected number of data points: {total_subjects}")
    print("=========================================================\n")
    
    for index in tqdm(predictions.keys(), ncols=80):
        prediction = _reform_data_from_dict(predictions[index])
        label = _reform_data_from_dict(labels[index])

        video_frame_size = prediction.shape[0]
        subject_windows = 0
        subject_skipped = 0
        
        if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
            window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.TEST.DATA.FS
            if window_frame_size > video_frame_size:
                window_frame_size = video_frame_size
        else:
            window_frame_size = video_frame_size
        
        # Calculate expected number of windows for this subject
        expected_windows = int(np.ceil(video_frame_size / window_frame_size))

        for i in range(0, len(prediction), window_frame_size):
            pred_window = prediction[i:i+window_frame_size]
            label_window = label[i:i+window_frame_size]

            if len(pred_window) < 9:
                print(f"Subject {index}: Window frame size of {len(pred_window)} is smaller than minimum pad length of 9. Window ignored!")
                subject_skipped += 1
                total_windows_skipped += 1
                continue

            subject_windows += 1
            total_windows_processed += 1

            if config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Standardized" or \
                    config.TEST.DATA.PREPROCESS.LABEL_TYPE == "Raw":
                diff_flag_test = False
            elif config.TEST.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
                diff_flag_test = True
            else:
                raise ValueError("Unsupported label type in testing!")
            
            if config.INFERENCE.EVALUATION_METHOD == "peak detection":
                gt_hr_peak, pred_hr_peak, SNR, macc = calculate_metric_per_video(
                    pred_window, label_window, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='Peak')
                gt_hr_peak_all.append(gt_hr_peak)
                predict_hr_peak_all.append(pred_hr_peak)
                SNR_all.append(SNR)
                MACC_all.append(macc)
            elif config.INFERENCE.EVALUATION_METHOD == "FFT":
                gt_hr_fft, pred_hr_fft, SNR, macc = calculate_metric_per_video(
                    pred_window, label_window, diff_flag=diff_flag_test, fs=config.TEST.DATA.FS, hr_method='FFT')
                gt_hr_fft_all.append(gt_hr_fft)
                predict_hr_fft_all.append(pred_hr_fft)
                SNR_all.append(SNR)
                MACC_all.append(macc)
            else:
                raise ValueError("Inference evaluation method name wrong!")
        
        # Store processed window count for this subject
        windows_per_subject[index] = {
            'video_length_frames': video_frame_size,
            'video_length_seconds': video_frame_size / config.TEST.DATA.FS,
            'window_size_frames': window_frame_size,
            'window_size_seconds': window_frame_size / config.TEST.DATA.FS,
            'expected_windows': expected_windows,
            'processed_windows': subject_windows,
            'skipped_windows': subject_skipped
        }
    
    # Detailed window processing report
    print("\n=============== WINDOW PROCESSING SUMMARY ===============")
    print(f"Total subjects processed: {total_subjects}")
    print(f"Total windows processed: {total_windows_processed}")
    print(f"Total windows skipped (too small): {total_windows_skipped}")
    print(f"Average windows per subject: {total_windows_processed / total_subjects:.2f}")
    
    # Calculate theoretical maximum windows
    if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
        window_size_frames = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.TEST.DATA.FS
        print(f"\nWindow configuration effect on data points:")
        print(f"  - Current WINDOW_SIZE: {config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE} seconds")
        print(f"  - Current data points for Bland-Altman plot: {total_windows_processed}")
        
        # Suggest other window sizes
        alternative_windows = [1, 2, 5, 10, 15, 30, 60]
        print(f"\nEstimated data points with different WINDOW_SIZE values:")
        for w_size in alternative_windows:
            if w_size == config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE:
                continue
            # Roughly estimate based on current data
            est_points = min(total_subjects * int(np.ceil(
                (config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * total_windows_processed) / 
                (total_subjects * w_size)
            )), total_subjects * 1000)  # Cap at a reasonable number
            print(f"  - WINDOW_SIZE = {w_size} seconds: ~{est_points} data points")
    
    print("=========================================================\n")
    
    # Filename ID to be used in any results files (e.g., Bland-Altman plots) that get saved
    if config.TOOLBOX_MODE == 'train_and_test':
        filename_id = config.TRAIN.MODEL_FILE_NAME
    elif config.TOOLBOX_MODE == 'only_test':
        model_file_root = config.INFERENCE.MODEL_PATH.split("/")[-1].split(".pth")[0]
        filename_id = model_file_root + "_" + config.TEST.DATA.DATASET
    else:
        raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')

    # At the end of calculations, print number of data points for Bland-Altman plots
    if config.INFERENCE.EVALUATION_METHOD == "FFT":
        gt_hr_fft_all = np.array(gt_hr_fft_all)
        predict_hr_fft_all = np.array(predict_hr_fft_all)
        SNR_all = np.array(SNR_all)
        MACC_all = np.array(MACC_all)
        num_test_samples = len(predict_hr_fft_all)
        print(f"\nNumber of data points for FFT Bland-Altman plot: {num_test_samples}")
        
        for metric in config.TEST.METRICS:
            if metric == "MAE":
                MAE_FFT = np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all))
                standard_error = np.std(np.abs(predict_hr_fft_all - gt_hr_fft_all)) / np.sqrt(num_test_samples)
                print("FFT MAE (FFT Label): {0} +/- {1}".format(MAE_FFT, standard_error))
            elif metric == "RMSE":
                # Calculate the squared errors, then RMSE, in order to allow
                # for a more robust and intuitive standard error that won't
                # be influenced by abnormal distributions of errors.
                squared_errors = np.square(predict_hr_fft_all - gt_hr_fft_all)
                RMSE_FFT = np.sqrt(np.mean(squared_errors))
                standard_error = np.sqrt(np.std(squared_errors) / np.sqrt(num_test_samples))
                print("FFT RMSE (FFT Label): {0} +/- {1}".format(RMSE_FFT, standard_error))
            elif metric == "MAPE":
                MAPE_FFT = np.mean(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
                standard_error = np.std(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) / np.sqrt(num_test_samples) * 100
                print("FFT MAPE (FFT Label): {0} +/- {1}".format(MAPE_FFT, standard_error))
            elif metric == "Pearson":
                Pearson_FFT = np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)
                correlation_coefficient = Pearson_FFT[0][1]
                standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                print("FFT Pearson (FFT Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
            elif metric == "SNR":
                SNR_FFT = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_FFT, standard_error))
            elif metric == "MACC":
                MACC_avg = np.mean(MACC_all)
                standard_error = np.std(MACC_all) / np.sqrt(num_test_samples)
                print("FFT MACC (FFT Label): {0} +/- {1}".format(MACC_avg, standard_error))
            elif "AU" in metric:
                pass
            elif "BA" in metric:  
                compare = BlandAltman(gt_hr_fft_all, predict_hr_fft_all, config, averaged=True)
                print(f"\nCreating Bland-Altman plots with {num_test_samples} data points")
                print(f"Window configuration: USE_SMALLER_WINDOW={config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW}, WINDOW_SIZE={config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE}")
                compare.scatter_plot(
                    x_label='GT PPG HR [bpm]',
                    y_label='rPPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_FFT_BlandAltman_ScatterPlot',
                    file_name=f'{filename_id}_FFT_BlandAltman_ScatterPlot.pdf')
                compare.difference_plot(
                    x_label='Difference between rPPG HR and GT PPG HR [bpm]',
                    y_label='Average of rPPG HR and GT PPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_FFT_BlandAltman_DifferencePlot',
                    file_name=f'{filename_id}_FFT_BlandAltman_DifferencePlot.pdf')
                plot_hr_distribution(gt_hr_fft_all, save_path=config.LOG.PATH)
            else:
                raise ValueError("Wrong Test Metric Type")
    elif config.INFERENCE.EVALUATION_METHOD == "peak detection":
        gt_hr_peak_all = np.array(gt_hr_peak_all)
        predict_hr_peak_all = np.array(predict_hr_peak_all)
        SNR_all = np.array(SNR_all)
        MACC_all = np.array(MACC_all)
        num_test_samples = len(predict_hr_peak_all)
        for metric in config.TEST.METRICS:
            if metric == "MAE":
                MAE_PEAK = np.mean(np.abs(predict_hr_peak_all - gt_hr_peak_all))
                standard_error = np.std(np.abs(predict_hr_peak_all - gt_hr_peak_all)) / np.sqrt(num_test_samples)
                print("Peak MAE (Peak Label): {0} +/- {1}".format(MAE_PEAK, standard_error))
            elif metric == "RMSE":
                # Calculate the squared errors, then RMSE, in order to allow
                # for a more robust and intuitive standard error that won't
                # be influenced by abnormal distributions of errors.
                squared_errors = np.square(predict_hr_peak_all - gt_hr_peak_all)
                RMSE_PEAK = np.sqrt(np.mean(squared_errors))
                standard_error = np.sqrt(np.std(squared_errors) / np.sqrt(num_test_samples))
                print("PEAK RMSE (Peak Label): {0} +/- {1}".format(RMSE_PEAK, standard_error))
            elif metric == "MAPE":
                MAPE_PEAK = np.mean(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) * 100
                standard_error = np.std(np.abs((predict_hr_peak_all - gt_hr_peak_all) / gt_hr_peak_all)) / np.sqrt(num_test_samples) * 100
                print("PEAK MAPE (Peak Label): {0} +/- {1}".format(MAPE_PEAK, standard_error))
            elif metric == "Pearson":
                Pearson_PEAK = np.corrcoef(predict_hr_peak_all, gt_hr_peak_all)
                correlation_coefficient = Pearson_PEAK[0][1]
                standard_error = np.sqrt((1 - correlation_coefficient**2) / (num_test_samples - 2))
                print("PEAK Pearson (Peak Label): {0} +/- {1}".format(correlation_coefficient, standard_error))
            elif metric == "SNR":
                SNR_PEAK = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("PEAK SNR (PEAK Label): {0} +/- {1} (dB)".format(SNR_PEAK, standard_error))
            elif metric == "MACC":
                MACC_avg = np.mean(MACC_all)
                standard_error = np.std(MACC_all) / np.sqrt(num_test_samples)
                print("PEAK MACC (PEAK Label): {0} +/- {1}".format(MACC_avg, standard_error))
            elif "AU" in metric:
                pass
            elif "BA" in metric:
                compare = BlandAltman(gt_hr_peak_all, predict_hr_peak_all, config, averaged=True)
                compare.scatter_plot(
                    x_label='GT PPG HR [bpm]',
                    y_label='rPPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_Peak_BlandAltman_ScatterPlot',
                    file_name=f'{filename_id}_Peak_BlandAltman_ScatterPlot.pdf')
                compare.difference_plot(
                    x_label='Difference between rPPG HR and GT PPG HR [bpm]',
                    y_label='Average of rPPG HR and GT PPG HR [bpm]',
                    show_legend=True, figure_size=(5, 5),
                    the_title=f'{filename_id}_Peak_BlandAltman_DifferencePlot',
                    file_name=f'{filename_id}_Peak_BlandAltman_DifferencePlot.pdf')
            else:
                raise ValueError("Wrong Test Metric Type")
    else:
        raise ValueError("Inference evaluation method name wrong!")
