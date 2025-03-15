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
    sbar = tqdm(data_loader["unsupervised"], ncols=80)
    for _, test_batch in enumerate(sbar):
        batch_size = test_batch[0].shape[0]
        for idx in range(batch_size):
            data_input, labels_input = test_batch[0][idx].cpu().numpy(), test_batch[1][idx].cpu().numpy()
            data_input = data_input[..., :3]
            if method_name == "POS":
                BVP = POS_WANG(data_input, config.UNSUPERVISED.DATA.FS)
            elif method_name == "CHROM":
                BVP = CHROME_DEHAAN(data_input, config.UNSUPERVISED.DATA.FS)
            elif method_name == "ICA":
                BVP = ICA_POH(data_input, config.UNSUPERVISED.DATA.FS)
            elif method_name == "GREEN":
                BVP = GREEN(data_input)
            elif method_name == "LGI":
                BVP = LGI(data_input)
            elif method_name == "PBV":
                BVP = PBV(data_input)
            else:
                raise ValueError("unsupervised method name wrong!")

            video_frame_size = test_batch[0].shape[1]
            if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
                window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.UNSUPERVISED.DATA.FS
                if window_frame_size > video_frame_size:
                    window_frame_size = video_frame_size
            else:
                window_frame_size = video_frame_size

            for i in range(0, len(BVP), window_frame_size):
                BVP_window = BVP[i:i+window_frame_size]
                label_window = labels_input[i:i+window_frame_size]

                if len(BVP_window) < 9:
                    print(f"Window frame size of {len(BVP_window)} is smaller than minimum pad length of 9. Window ignored!")
                    continue

                if config.INFERENCE.EVALUATION_METHOD == "peak detection":
                    gt_hr, pre_hr, SNR, macc = calculate_metric_per_video(BVP_window, label_window, diff_flag=False,
                                                                    fs=config.UNSUPERVISED.DATA.FS, hr_method='Peak')
                    gt_hr_peak_all.append(gt_hr)
                    predict_hr_peak_all.append(pre_hr)
                    SNR_all.append(SNR)
                    MACC_all.append(macc)
                elif config.INFERENCE.EVALUATION_METHOD == "FFT":
                    gt_fft_hr, pre_fft_hr, SNR, macc = calculate_metric_per_video(BVP_window, label_window, diff_flag=False,
                                                                    fs=config.UNSUPERVISED.DATA.FS, hr_method='FFT')
                    gt_hr_fft_all.append(gt_fft_hr)
                    predict_hr_fft_all.append(pre_fft_hr)
                    SNR_all.append(SNR)
                    MACC_all.append(macc)
                else:
                    raise ValueError("Inference evaluation method name wrong!")
    print("Used Unsupervised Method: " + method_name)

    # Filename ID to be used in any results files (e.g., Bland-Altman plots) that get saved
    if config.TOOLBOX_MODE == 'unsupervised_method':
        filename_id = method_name + "_" + config.UNSUPERVISED.DATA.DATASET
    else:
        raise ValueError('unsupervised_predictor.py evaluation only supports unsupervised_method!')

    if config.INFERENCE.EVALUATION_METHOD == "peak detection":
        predict_hr_peak_all = np.array(predict_hr_peak_all)
        gt_hr_peak_all = np.array(gt_hr_peak_all)
        SNR_all = np.array(SNR_all)
        MACC_all = np.array(MACC_all)
        num_test_samples = len(predict_hr_peak_all)
        for metric in config.UNSUPERVISED.METRICS:
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
                SNR_FFT = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_FFT, standard_error))
            elif metric == "MACC":
                MACC_avg = np.mean(MACC_all)
                standard_error = np.std(MACC_all) / np.sqrt(num_test_samples)
                print("MACC (avg): {0} +/- {1}".format(MACC_avg, standard_error))
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
    elif config.INFERENCE.EVALUATION_METHOD == "FFT":
        predict_hr_fft_all = np.array(predict_hr_fft_all)
        gt_hr_fft_all = np.array(gt_hr_fft_all)
        SNR_all = np.array(SNR_all)
        MACC_all = np.array(MACC_all)
        num_test_samples = len(predict_hr_fft_all)
        for metric in config.UNSUPERVISED.METRICS:
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
                SNR_PEAK = np.mean(SNR_all)
                standard_error = np.std(SNR_all) / np.sqrt(num_test_samples)
                print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_PEAK, standard_error))
            elif metric == "MACC":
                MACC_avg = np.mean(MACC_all)
                standard_error = np.std(MACC_all) / np.sqrt(num_test_samples)
                print("MACC (avg): {0} +/- {1}".format(MACC_avg, standard_error))
            elif "BA" in metric:
                compare = BlandAltman(gt_hr_fft_all, predict_hr_fft_all, config, averaged=True)
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
            else:
                raise ValueError("Wrong Test Metric Type")
    else:
        raise ValueError("Inference evaluation method name wrong!")



import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import norm
import os

def CHROME_DEHAAN_with_uncertainty(frames, FS, n_bootstrap=100):
    LPF, HPF = 0.7, 2.5
    WinSec = 1.6
    FN = frames.shape[0]
    NyquistF = 1/2*FS
    B, A = signal.butter(3, [LPF/NyquistF, HPF/NyquistF], 'bandpass')
    WinL = int(np.ceil(WinSec*FS))
    if WinL % 2: WinL += 1
    
    # Create bootstrap sample indices
    bootstrap_indices = [np.random.choice(FN, size=FN, replace=True) for _ in range(n_bootstrap)]
    all_BVPs = []
    
    # Original estimation
    RGB = process_video(frames)
    BVP_original = extract_BVP(RGB, FS, WinL, B, A)
    all_BVPs.append(BVP_original)
    
    # Bootstrap estimations
    for indices in bootstrap_indices:
        bootstrapped_frames = frames[indices]
        RGB_boot = process_video(bootstrapped_frames)
        BVP_boot = extract_BVP(RGB_boot, FS, WinL, B, A)
        all_BVPs.append(BVP_boot)
    
    return all_BVPs

def process_video(frames):
    RGB = []
    for frame in frames:
        sum = np.sum(np.sum(frame, axis=0), axis=0)
        RGB.append(sum/(frame.shape[0]*frame.shape[1]))
    return np.asarray(RGB)

def extract_BVP(RGB, FS, WinL, B, A):
    FN = RGB.shape[0]
    NWin = int(np.floor((FN-WinL//2)/(WinL//2)))
    WinS, WinM = 0, int(WinL//2)
    WinE = WinS+WinL
    totallen = (WinL//2)*(NWin+1)
    S = np.zeros(totallen)
    
    for i in range(NWin):
        RGBBase = np.mean(RGB[WinS:WinE, :], axis=0)
        RGBNorm = np.zeros((WinE-WinS, 3))
        for temp in range(WinS, WinE):
            RGBNorm[temp-WinS] = np.true_divide(RGB[temp], RGBBase)
        Xs = np.squeeze(3*RGBNorm[:, 0]-2*RGBNorm[:, 1])
        Ys = np.squeeze(1.5*RGBNorm[:, 0]+RGBNorm[:, 1]-1.5*RGBNorm[:, 2])
        Xf = signal.filtfilt(B, A, Xs, axis=0)
        Yf = signal.filtfilt(B, A, Ys)
        
        Alpha = np.std(Xf) / np.std(Yf)
        SWin = Xf-Alpha*Yf
        SWin = np.multiply(SWin, signal.windows.hann(WinL))
        
        S[WinS:WinM] = S[WinS:WinM] + SWin[:int(WinL//2)]
        S[WinM:WinE] = SWin[int(WinL//2):]
        WinS = WinM
        WinM = WinS+WinL//2
        WinE = WinS+WinL
    return S

def calculate_hr_with_uncertainty(BVPs, fs, low_hz=0.7, high_hz=2.5):
    all_hrs = []
    
    for bvp in BVPs:
        # FFT-based heart rate estimation
        T = len(bvp)/fs
        freqs = np.linspace(0, fs/2, int(np.floor(len(bvp)/2+1)))
        freq_mask = (freqs >= low_hz) & (freqs <= high_hz)
        
        fft_data = np.abs(np.fft.rfft(bvp))
        fft_data = fft_data[:len(freqs)]
        
        # Find the frequency with maximum amplitude in the relevant range
        max_idx = np.argmax(fft_data[freq_mask])
        target_freq_idx = np.where(freq_mask)[0][max_idx]
        hr_freq = freqs[target_freq_idx]
        hr_bpm = hr_freq * 60
        all_hrs.append(hr_bpm)
    
    # Original estimate is the first element
    hr_est = all_hrs[0]
    
    # Bootstrap estimates are the remaining elements
    bootstrap_hrs = np.array(all_hrs[1:])
    
    # Calculate confidence intervals
    hr_std = np.std(bootstrap_hrs)
    hr_lower = np.percentile(bootstrap_hrs, 2.5)
    hr_upper = np.percentile(bootstrap_hrs, 97.5)
    
    return hr_est, hr_std, hr_lower, hr_upper, bootstrap_hrs

def plot_hr_with_uncertainty(time_windows, hr_estimates, hr_lower, hr_upper, save_path, filename="hr_uncertainty_plot.pdf"):
    plt.figure(figsize=(10, 6))
    plt.plot(time_windows, hr_estimates, 'r-', label='Estimated HR')
    plt.fill_between(time_windows, hr_lower, hr_upper, color='r', alpha=0.2, label='95% Confidence Interval')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Heart Rate (BPM)')
    plt.title('rPPG Heart Rate Estimation with Uncertainty')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, filename), bbox_inches='tight', dpi=300)
    plt.close()

def plot_rppg_signal_with_uncertainty(time, original_signal, bootstrap_signals, save_path, filename="rppg_uncertainty_plot.pdf"):
    plt.figure(figsize=(12, 6))
    
    # Plot original signal
    plt.plot(time, original_signal, 'b-', label='Original rPPG Signal', zorder=5)
    
    # Calculate percentiles from bootstrap signals for confidence bands
    lower_band = np.percentile(bootstrap_signals, 2.5, axis=0)
    upper_band = np.percentile(bootstrap_signals, 97.5, axis=0)
    
    # Plot confidence bands
    plt.fill_between(time, lower_band, upper_band, color='blue', alpha=0.2, label='95% Confidence Interval')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title('rPPG Signal with Uncertainty')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, filename), bbox_inches='tight', dpi=300)
    plt.close()

def calibrate_with_gp(pred_hrs, gt_hrs):
    X = pred_hrs.reshape(-1, 1)
    y = gt_hrs
    
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(X, y)
    
    return gp

def evaluate_with_uncertainty(frames, labels, config, window_size=10, n_bootstrap=100):
    fs = config.UNSUPERVISED.DATA.FS
    total_frames = frames.shape[0]
    window_frames = int(window_size * fs)
    
    # Result arrays
    time_windows = []
    hr_estimates = []
    hr_stds = []
    hr_lower_bands = []
    hr_upper_bands = []
    gt_hrs = []
    bootstrap_distributions = []
    
    # Process each window
    for start_frame in range(0, total_frames, window_frames):
        end_frame = min(start_frame + window_frames, total_frames)
        if end_frame - start_frame < 9:  # Skip if window too small
            continue
            
        window_frames_data = frames[start_frame:end_frame]
        window_labels = labels[start_frame:end_frame]
        
        # Apply CHROME method with bootstrap uncertainty
        all_BVPs = CHROME_DEHAAN_with_uncertainty(window_frames_data, fs, n_bootstrap)
        
        # Calculate heart rate with uncertainty
        hr_est, hr_std, hr_lower, hr_upper, bootstrap_hrs = calculate_hr_with_uncertainty(all_BVPs, fs)
        
        # Calculate ground truth HR using FFT (same method as the prediction)
        T = len(window_labels)/fs
        freqs = np.linspace(0, fs/2, int(np.floor(len(window_labels)/2+1)))
        freq_mask = (freqs >= 0.7) & (freqs <= 2.5)
        
        fft_data = np.abs(np.fft.rfft(window_labels))
        fft_data = fft_data[:len(freqs)]
        
        max_idx = np.argmax(fft_data[freq_mask])
        target_freq_idx = np.where(freq_mask)[0][max_idx]
        gt_hr = freqs[target_freq_idx] * 60
        
        # Store results
        time_windows.append(start_frame / fs)
        hr_estimates.append(hr_est)
        hr_stds.append(hr_std)
        hr_lower_bands.append(hr_lower)
        hr_upper_bands.append(hr_upper)
        gt_hrs.append(gt_hr)
        bootstrap_distributions.append(bootstrap_hrs)
        
        # Plot rPPG signal with uncertainty if it's the first window (example)
        if start_frame == 0:
            time = np.arange(len(all_BVPs[0])) / fs
            original_signal = all_BVPs[0]
            bootstrap_signals = np.array(all_BVPs[1:])
            plot_rppg_signal_with_uncertainty(time, original_signal, bootstrap_signals, config.LOG.PATH)
    
    # Convert to numpy arrays
    time_windows = np.array(time_windows)
    hr_estimates = np.array(hr_estimates)
    hr_stds = np.array(hr_stds)
    hr_lower_bands = np.array(hr_lower_bands)
    hr_upper_bands = np.array(hr_upper_bands)
    gt_hrs = np.array(gt_hrs)
    
    # Plot HR over time with uncertainty
    plot_hr_with_uncertainty(time_windows, hr_estimates, hr_lower_bands, hr_upper_bands, config.LOG.PATH)
    
    # Calibrate uncertainty using Gaussian Process if we have ground truth
    if len(gt_hrs) > 10:
        gp = calibrate_with_gp(hr_estimates, gt_hrs)
        
        # Plot calibrated predictions
        plt.figure(figsize=(8, 6))
        plt.scatter(hr_estimates, gt_hrs, color="blue", label="Original Data", alpha=0.6)
        
        # Sort for smooth line plot
        sort_idx = np.argsort(hr_estimates)
        hr_est_sorted = hr_estimates[sort_idx]
        
        # Predict with GP
        y_pred, sigma = gp.predict(hr_est_sorted.reshape(-1, 1), return_std=True)
        
        plt.plot(hr_est_sorted, y_pred, 'r-', label="GP Calibration")
        plt.fill_between(hr_est_sorted, y_pred - 2*sigma, y_pred + 2*sigma, 
                         color='r', alpha=0.2, label="GP Uncertainty (2σ)")
        
        plt.xlabel("Uncalibrated HR Estimate (BPM)")
        plt.ylabel("Ground Truth HR (BPM)")
        plt.title("GP Calibration of HR Estimates")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(config.LOG.PATH, "gp_calibration.pdf"), bbox_inches='tight', dpi=300)
        plt.close()
    
    return {
        'time_windows': time_windows,
        'hr_estimates': hr_estimates,
        'hr_stds': hr_stds,
        'hr_lower_bands': hr_lower_bands,
        'hr_upper_bands': hr_upper_bands,
        'gt_hrs': gt_hrs,
        'bootstrap_distributions': bootstrap_distributions
    }

# Function to extend the unsupervised predictor with uncertainty
def unsupervised_predict_with_uncertainty(config, data_loader):
    print("=== CHROM Method with Uncertainty Quantification ===")
    results_per_subject = {}
    
    for _, test_batch in enumerate(data_loader["unsupervised"]):
        batch_size = test_batch[0].shape[0]
        for idx in range(batch_size):
            frames = test_batch[0][idx].cpu().numpy()
            labels = test_batch[1][idx].cpu().numpy()
            frames = frames[..., :3]  # Keep only RGB channels
            
            # Process with uncertainty quantification
            results = evaluate_with_uncertainty(
                frames, 
                labels, 
                config, 
                window_size=config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE,
                n_bootstrap=100
            )
            
            subject_id = f"subject_{idx}"
            results_per_subject[subject_id] = results
    
    return results_per_subject
