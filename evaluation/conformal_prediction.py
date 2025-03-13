import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import norm
import matplotlib.pyplot as plt

class ConformalPredictor:
    """
    Implements conformal prediction for heart rate estimation.
    This assumes a black-box model where we only have access to input-output pairs.
    """
    def __init__(self, alpha=0.1):
        """
        Initialize the conformal predictor with a significance level.
        
        Args:
            alpha: The significance level (e.g., 0.1 for 90% confidence)
        """
        self.alpha = alpha
        self.calibration_errors = None
        
    def calibrate(self, predictions, ground_truth):
        """
        Calibrate the predictor using a held-out calibration set.
        
        Args:
            predictions: Array of model predictions (estimated heart rates)
            ground_truth: Array of ground truth values (actual heart rates)
        """
        # Calculate absolute errors for calibration
        self.calibration_errors = np.abs(predictions - ground_truth)
        # Sort errors for quantile calculation
        self.calibration_errors = np.sort(self.calibration_errors)
        
    def predict(self, point_prediction):
        """
        Generate prediction intervals for a given point prediction.
        
        Args:
            point_prediction: The model's point prediction for heart rate
            
        Returns:
            tuple: (lower_bound, upper_bound) defining the prediction interval
        """
        if self.calibration_errors is None:
            raise ValueError("Calibrate the predictor first using calibration data")
            
        # Calculate the quantile for the desired confidence level
        n = len(self.calibration_errors)
        q_index = int(np.ceil((n + 1) * (1 - self.alpha))) - 1
        q_index = min(q_index, n - 1)  # Ensure we don't exceed array bounds
        
        # Get the quantile value
        quantile = self.calibration_errors[q_index]
        
        # Create prediction interval
        lower_bound = point_prediction - quantile
        upper_bound = point_prediction + quantile
        
        # Ensure lower bound is non-negative for heart rate
        lower_bound = max(0, lower_bound)
        
        return lower_bound, upper_bound
    
    def evaluate_coverage(self, predictions, ground_truth):
        """
        Evaluate the empirical coverage of the prediction intervals.
        
        Args:
            predictions: Array of model predictions
            ground_truth: Array of ground truth values
            
        Returns:
            float: The empirical coverage (proportion of ground truth values within intervals)
        """
        in_interval = 0
        intervals = []
        
        for pred, gt in zip(predictions, ground_truth):
            lower, upper = self.predict(pred)
            intervals.append((lower, upper))
            if lower <= gt <= upper:
                in_interval += 1
                
        empirical_coverage = in_interval / len(predictions)
        avg_interval_width = np.mean([upper - lower for lower, upper in intervals])
        
        return empirical_coverage, avg_interval_width, intervals
    
    def plot_intervals(self, predictions, ground_truth, method_name, file_name=None):
        """
        Plot the prediction intervals alongside ground truth values.
        
        Args:
            predictions: Array of model predictions
            ground_truth: Array of ground truth values
            method_name: Name of the method used (for plot title)
            file_name: Optional filename to save the plot
        """
        # Generate intervals
        _, _, intervals = self.evaluate_coverage(predictions, ground_truth)
        lower_bounds = [interval[0] for interval in intervals]
        upper_bounds = [interval[1] for interval in intervals]
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Plot ground truth
        plt.plot(ground_truth, 'go', label='Ground Truth HR')
        
        # Plot predictions
        plt.plot(predictions, 'bo', label='Predicted HR')
        
        # Plot intervals
        for i in range(len(predictions)):
            plt.plot([i, i], [lower_bounds[i], upper_bounds[i]], 'r-', alpha=0.3)
            
        # Add shaded area for intervals
        plt.fill_between(
            range(len(predictions)), 
            lower_bounds, 
            upper_bounds, 
            color='red', 
            alpha=0.1, 
            label=f'{(1-self.alpha)*100:.0f}% Prediction Intervals'
        )
        
        plt.xlabel('Sample Index')
        plt.ylabel('Heart Rate (bpm)')
        plt.title(f'Conformal Prediction Intervals for {method_name}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if file_name:
            plt.savefig(file_name, dpi=300, bbox_inches='tight')
            
        plt.close()
        
        # Create a second plot showing errors and interval widths
        plt.figure(figsize=(10, 6))
        
        errors = np.abs(predictions - ground_truth)
        interval_widths = [upper - lower for lower, upper in intervals]
        
        plt.scatter(errors, interval_widths, alpha=0.5)
        plt.xlabel('Absolute Error')
        plt.ylabel('Interval Width')
        plt.title(f'Error vs Interval Width for {method_name}')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if file_name:
            error_width_filename = file_name.replace('.pdf', '_error_width.pdf')
            plt.savefig(error_width_filename, dpi=300, bbox_inches='tight')
            
        plt.close()

# Function to implement conformal prediction in the unsupervised_predict function
def add_conformal_prediction(config, predict_hr_all, gt_hr_all, method_name, evaluation_method):
    """
    Add conformal prediction to the heart rate predictions.
    
    Args:
        config: Configuration object
        predict_hr_all: Array of predicted heart rates
        gt_hr_all: Array of ground truth heart rates
        method_name: Name of the unsupervised method used
        evaluation_method: String indicating whether "peak detection" or "FFT" was used
        
    Returns:
        tuple: (empirical_coverage, avg_interval_width)
    """
    # Split data into calibration and test sets (70% calibration, 30% test)
    # In a real application, you might want to use a proper calibration set
    pred_calibration, pred_test, gt_calibration, gt_test = train_test_split(
        predict_hr_all, gt_hr_all, test_size=0.3, random_state=42
    )
    
    # Create and calibrate the conformal predictor
    cp = ConformalPredictor(alpha=0.1)  # 90% confidence level
    cp.calibrate(pred_calibration, gt_calibration)
    
    # Evaluate coverage on test set
    empirical_coverage, avg_interval_width, _ = cp.evaluate_coverage(pred_test, gt_test)
    
    print(f"\n=== Conformal Prediction Results ({evaluation_method}) ===")
    print(f"Target confidence level: {(1-cp.alpha)*100:.0f}%")
    print(f"Empirical coverage: {empirical_coverage*100:.2f}%")
    print(f"Average interval width: {avg_interval_width:.2f} bpm")
    
    # Generate and save plots
    if config.TOOLBOX_MODE == 'unsupervised_method':
        filename_id = method_name + "_" + config.UNSUPERVISED.DATA.DATASET
    else:
        raise ValueError('unsupervised_predictor.py evaluation only supports unsupervised_method!')
    
    plot_filename = f'{filename_id}_{evaluation_method}_ConformalPrediction.pdf'
    cp.plot_intervals(pred_test, gt_test, method_name, plot_filename)
    
    return empirical_coverage, avg_interval_width
