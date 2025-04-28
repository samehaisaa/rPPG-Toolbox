"""
UncertaintyWrapper module for rPPG-Toolbox.

This module implements a heteroscedastic uncertainty model that:
1. During training, takes CHROM-extracted PPG signals and ground truth
2. Computes per-sample features (SNR, RGB variances, etc.)
3. Trains a model to predict aleatoric uncertainty
4. Provides a prediction function that returns signal with uncertainty estimates
"""

import os
import numpy as np
import pickle
from scipy import signal
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import cv2
from evaluation.post_process import _calculate_SNR
from unsupervised_methods.methods.CHROME_DEHAAN import CHROME_DEHAAN, process_video


class UncertaintyWrapper:
    """
    Wrapper for estimating aleatoric uncertainty in CHROM-extracted PPG signals.
    
    This class implements a heteroscedastic model to predict uncertainty
    for each time point in the PPG signal based on features extracted
    from the video and signal.
    """
    
    def __init__(self, fs=30, model_path=None):
        """
        Initialize the UncertaintyWrapper.
        
        Args:
            fs (float): Sampling frequency in Hz
            model_path (str, optional): Path to load a pre-trained model
        """
        self.fs = fs
        self.feature_scaler = StandardScaler()
        self.error_scaler = StandardScaler()
        self.model = None
        
        # Load pre-trained model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def extract_features(self, frames, ppg_signal):
        """
        Extract features from video frames and PPG signal.
        
        Args:
            frames (np.ndarray): Video frames of shape (T, H, W, C)
            ppg_signal (np.ndarray): Extracted PPG signal
            
        Returns:
            np.ndarray: Feature vector for each time point
        """
        # Get RGB signals
        rgb_signals = process_video(frames)
        
        # Initialize array to store features for each time point
        features = []
        
        # Compute global features
        
        # 1. RGB channel variances
        rgb_vars = np.var(rgb_signals, axis=0)
        
        # 2. RGB channel means
        rgb_means = np.mean(rgb_signals, axis=0)
        
        # 3. Overall signal SNR
        signal_snr = _calculate_SNR(ppg_signal, self.fs)
        
        # Sliding window analysis for local features
        window_size = int(1.0 * self.fs)  # 1-second window
        hop_size = int(0.25 * self.fs)    # 0.25-second hop
        
        for i in range(0, len(ppg_signal), hop_size):
            end_idx = min(i + window_size, len(ppg_signal))
            if end_idx - i < window_size // 2:  # Skip windows that are too small
                continue
                
            # PPG signal window
            ppg_window = ppg_signal[i:end_idx]
            
            # Corresponding RGB window
            rgb_window = rgb_signals[i:end_idx] if i < len(rgb_signals) else None
            
            # Local features for this window
            window_features = []
            
            # 4. Local PPG signal statistics
            window_features.extend([
                np.mean(ppg_window),
                np.std(ppg_window),
                np.max(ppg_window) - np.min(ppg_window),  # Range
            ])
            
            # 5. Local signal SNR
            local_snr = _calculate_SNR(ppg_window, self.fs)
            window_features.append(local_snr)
            
            # 6. Local RGB statistics
            if rgb_window is not None:
                local_rgb_vars = np.var(rgb_window, axis=0)
                local_rgb_means = np.mean(rgb_window, axis=0)
                window_features.extend(local_rgb_vars)
                window_features.extend(local_rgb_means)
            else:
                # Use global values if rgb_window is None
                window_features.extend(rgb_vars)
                window_features.extend(rgb_means)
            
            # 7. Global features for context
            window_features.extend(rgb_vars)
            window_features.extend(rgb_means)
            window_features.append(signal_snr)
            
            # Create feature vector for each time point in the window
            for _ in range(i, end_idx):
                features.append(window_features)
        
        # Ensure we have features for all time points
        if len(features) < len(ppg_signal):
            # Repeat the last feature vector for any remaining time points
            features.extend([features[-1]] * (len(ppg_signal) - len(features)))
        elif len(features) > len(ppg_signal):
            # Truncate if we have more feature vectors than time points
            features = features[:len(ppg_signal)]
            
        return np.array(features)
    
    def train(self, video_paths, gt_signals, output_model_path=None):
        """
        Train the uncertainty model using videos and ground truth signals.
        
        Args:
            video_paths (list): List of paths to video files
            gt_signals (list): List of ground truth PPG signals
            output_model_path (str, optional): Path to save the trained model
            
        Returns:
            self: The trained model instance
        """
        all_features = []
        all_errors = []
        
        for i, (video_path, gt_signal) in enumerate(zip(video_paths, gt_signals)):
            print(f"Processing video {i+1}/{len(video_paths)}: {video_path}")
            
            # Load video frames
            frames = self._load_video(video_path)
            if frames is None:
                print(f"Warning: Could not load video {video_path}. Skipping.")
                continue
            
            # Extract PPG signal using CHROM
            chrom_result = CHROME_DEHAAN(frames, self.fs)
            ppg_signal = chrom_result['BVP']
            
            # Ensure signals have the same length for error calculation
            min_length = min(len(ppg_signal), len(gt_signal))
            ppg_signal = ppg_signal[:min_length]
            gt_signal = gt_signal[:min_length]
            
            # Normalize signals for error calculation
            ppg_norm = (ppg_signal - np.mean(ppg_signal)) / np.std(ppg_signal)
            gt_norm = (gt_signal - np.mean(gt_signal)) / np.std(gt_signal)
            
            # Calculate absolute errors
            errors = np.abs(ppg_norm - gt_norm)
            
            # Extract features
            features = self.extract_features(frames, ppg_signal)
            
            # Ensure features and errors have the same length
            min_length = min(len(features), len(errors))
            features = features[:min_length]
            errors = errors[:min_length]
            
            all_features.append(features)
            all_errors.append(errors)
        
        # Combine all features and errors
        X = np.vstack(all_features)
        y = np.concatenate(all_errors)
        
        # Scale features and errors
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.error_scaler.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Train model (Random Forest regressor)
        print("Training uncertainty model...")
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y_scaled)
        
        print(f"Model trained on {len(X)} samples from {len(video_paths)} videos.")
        
        # Save model if output path is provided
        if output_model_path:
            self.save_model(output_model_path)
        
        return self
    
    def predict_with_uncertainty(self, video_path, start_time=None, end_time=None):
        """
        Process a video and return the PPG signal with uncertainty estimates.
        
        Args:
            video_path (str): Path to the video file
            start_time (float, optional): Start time in seconds
            end_time (float, optional): End time in seconds
            
        Returns:
            tuple: (timestamps, ppg_signal, uncertainty)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load_model() first.")
        
        # Load video frames with memory-efficient approach
        frames = self._load_video_chunk(video_path, start_time, end_time)
        if frames is None:
            raise ValueError(f"Could not load video {video_path}")
        
        # Extract PPG signal using CHROM
        chrom_result = CHROME_DEHAAN(frames, self.fs)
        ppg_signal = chrom_result['BVP']
        
        # Extract features
        features = self.extract_features(frames, ppg_signal)
        
        # Scale features
        features_scaled = self.feature_scaler.transform(features)
        
        # Predict uncertainties (scaled errors)
        uncertainties_scaled = self.model.predict(features_scaled)
        
        # Convert back to original scale
        uncertainties = self.error_scaler.inverse_transform(uncertainties_scaled.reshape(-1, 1)).ravel()
        
        # Create timestamps
        timestamps = np.arange(len(ppg_signal)) / self.fs
        if start_time is not None:
            timestamps += start_time
        
        return timestamps, ppg_signal, uncertainties
    
    def _load_video_chunk(self, video_path, start_time=None, end_time=None):
        """
        Load a chunk of video frames efficiently.
        
        Args:
            video_path (str): Path to the video file
            start_time (float, optional): Start time in seconds
            end_time (float, optional): End time in seconds
            
        Returns:
            np.ndarray: Video frames
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame range
        if start_time is None:
            start_frame = 0
        else:
            start_frame = int(start_time * fps)
        
        if end_time is None:
            end_frame = total_frames
        else:
            end_frame = int(end_time * fps)
        
        # Ensure valid frame range
        start_frame = max(0, min(start_frame, total_frames))
        end_frame = max(start_frame, min(end_frame, total_frames))
        
        # Set starting position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Read frames
        frames = []
        for _ in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        if not frames:
            return None
        
        return np.array(frames)
    
    def save_model(self, model_path):
        """
        Save the trained model to a file.
        
        Args:
            model_path (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No trained model to save.")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_scaler': self.feature_scaler,
            'error_scaler': self.error_scaler,
            'fs': self.fs
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """
        Load a trained model from a file.
        
        Args:
            model_path (str): Path to the saved model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.feature_scaler = model_data['feature_scaler']
        self.error_scaler = model_data['error_scaler']
        self.fs = model_data['fs']
        
        print(f"Model loaded from {model_path}")
    
    def _load_video(self, video_path):
        """
        Load video frames from a file.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            np.ndarray: Video frames of shape (T, H, W, C)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        if not frames:
            return None
        
        return np.array(frames) 
