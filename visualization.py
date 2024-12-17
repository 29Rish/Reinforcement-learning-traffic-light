import matplotlib.pyplot as plt  # Import the plotting library
import os  # Import the os module for file and directory operations

class Visualization:
    """
    A class to handle the visualization and saving of agent performance data.

    Attributes:
        _path (str): Directory path where the plots and data files will be saved.
        _dpi (int): Resolution of the saved plots in dots per inch.
    """

    def __init__(self, path, dpi):
        """
        Initialize the Visualization class with a file save path and DPI.

        Args:
            path (str): Path to the directory where outputs will be saved.
            dpi (int): DPI (resolution) for the saved plot images.
        """
        self._path = path  # Store the output directory path
        self._dpi = dpi  # Store the resolution for the saved plots

    def save_data_and_plot(self, data, filename, xlabel, ylabel):
        """
        Produce a plot of the data (e.g., agent performance over time) and save both the plot and raw data.

        Args:
            data (list): List of numerical values to be plotted and saved.
            filename (str): Base name for the saved plot and data files.
            xlabel (str): Label for the x-axis of the plot.
            ylabel (str): Label for the y-axis of the plot.

        Functionality:
            - Saves the plot as a PNG file with high resolution.
            - Saves the numerical data to a text file for further analysis.
        """
        # Calculate the minimum and maximum values in the data for setting plot limits
        min_val = min(data)  # Find the minimum value in the data
        max_val = max(data)  # Find the maximum value in the data

        # Set global font size for better readability of the plot
        plt.rcParams.update({'font.size': 24})

        # Create the plot
        plt.plot(data)  # Plot the data as a line graph
        plt.ylabel(ylabel)  # Set the label for the y-axis
        plt.xlabel(xlabel)  # Set the label for the x-axis

        # Adjust margins and axis limits for a cleaner display
        plt.margins(0)  # Remove any unnecessary margins around the plot
        plt.ylim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))  # Set y-axis limits with padding

        # Configure the figure size
        fig = plt.gcf()  # Get the current figure
        fig.set_size_inches(20, 11.25)  # Set the figure size to 20x11.25 inches for better resolution

        # Save the plot to a file in the specified directory
        plot_path = os.path.join(self._path, 'plot_' + filename + '.png')  # Create the full file path for the plot
        fig.savefig(plot_path, dpi=self._dpi)  # Save the plot with specified DPI
        plt.close("all")  # Close the plot to free memory

        # Save the raw numerical data to a text file
        data_file_path = os.path.join(self._path, 'plot_' + filename + '_data.txt')  # Path for the data file
        with open(data_file_path, "w") as file:
            for value in data:  # Write each data point to the file
                file.write("%s\n" % value)  # Save the value followed by a newline
