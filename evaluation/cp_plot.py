import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt 

class CPPlot:
    def __init__(self, ground_truth, predictions, config=None):
        """
        Initialize the CPPlot class with ground truth values and predictions.
        Optionally, a config dictionary can be provided for additional settings.
        """
        self.gt = np.array(ground_truth)
        self.pred = np.array(predictions)
        self.config = config  # Not used here but can be extended for more options

    def plot(self, 
             x_label='Nominal Confidence Level', 
             y_label='Empirical Coverage',
             show_legend=True, 
             figure_size=(6, 6), 
             the_title='Conformal Prediction (CP) Plot', 
             file_name='CP_Plot.pdf'):
        """
        Generates a CP plot:
        - Varies the nominal confidence level between 0.5 and 0.99.
        - For each nominal level, it computes the quantile 'q' of the absolute residuals,
          and defines the prediction interval as [prediction - q, prediction + q].
        - It then calculates the empirical coverage: the fraction of cases where the ground truth
          falls within that prediction interval.
        - Both the empirical coverage curve and the ideal coverage line are plotted.
        """
        # Calculate absolute residuals between predictions and ground truth.
        residuals = np.abs(self.pred - self.gt)
        
        # Define a range of nominal confidence levels, e.g., from 0.5 to 0.99.
        nominal_confidences = np.linspace(0.5, 0.99, 50)
        empirical_coverages = []

        # For each nominal confidence level, calculate the corresponding prediction interval 
        # using the quantile of the residuals, and compute the empirical coverage.
        for conf in nominal_confidences:
            # Get the quantile 'q' corresponding to the current confidence level.
            q = np.quantile(residuals, conf)
            lower_bounds = self.pred - q
            upper_bounds = self.pred + q

            # Determine whether each ground truth value is within the prediction interval.
            covered = np.logical_and(self.gt >= lower_bounds, self.gt <= upper_bounds)
            empirical_coverage = np.mean(covered)
            empirical_coverages.append(empirical_coverage)

        # Plot the empirical coverage against the nominal confidence levels.
        plt.figure(figsize=figure_size)
        plt.plot(nominal_confidences, empirical_coverages, label='Empirical Coverage', marker='o')
        # Also plot the ideal line where empirical coverage equals the nominal level.
        plt.plot(nominal_confidences, nominal_confidences, 'r--', label='Ideal Coverage')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(the_title)
        if show_legend:
            plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(file_name)
        plt.close()
