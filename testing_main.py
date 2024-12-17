from __future__ import absolute_import  # Ensure compatibility with Python 2 (if needed) for absolute imports.
from __future__ import print_function    # Ensure compatibility with Python 2 for print function behavior.

import os  # Provides functions to interact with the operating system.
from shutil import copyfile  # Allows for copying files between directories.

# Import custom modules required for simulation and visualization.
from testing_simulation import Simulation  # Handles the simulation environment.
from generator import TrafficGenerator  # Generates traffic data for the simulation.
from model import TestModel  # Loads and uses the pre-trained model for decision-making.
from visualization import Visualization  # Handles data visualization and saving plots.
from utils import import_test_configuration, set_sumo, set_test_path  # Utility functions for configuration and setup.

# Main entry point of the script.
if __name__ == "__main__":
    # Load test configuration settings from a configuration file.
    config = import_test_configuration(config_file='testing_settings.ini')

    # Set up the SUMO simulation command with the GUI option, configuration file, and maximum simulation steps.
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])

    # Set paths for the model to test and the directory where test plots will be saved.
    model_path, plot_path = set_test_path(config['models_path_name'], config['model_to_test'])

    # Initialize the test model using the pre-trained model located at the specified path.
    Model = TestModel(
        input_dim=config['num_states'],  # Number of input features (state representation).
        model_path=model_path  # Path to the saved model file.
    )

    # Initialize the traffic generator to create vehicle data for the simulation.
    TrafficGen = TrafficGenerator(
        config['max_steps'],  # Maximum number of simulation steps.
        config['n_cars_generated']  # Number of cars to generate during the simulation.
    )

    # Initialize the visualization module to handle plotting and saving test results.
    Visualization = Visualization(
        plot_path,  # Path to save plots and data.
        dpi=96  # Resolution of the saved plots.
    )

    # Initialize the simulation with all required components and configurations.
    Simulation = Simulation(
        Model,  # The loaded pre-trained model to test.
        TrafficGen,  # The traffic generator for creating vehicle scenarios.
        sumo_cmd,  # The SUMO command for running the simulation.
        config['max_steps'],  # Maximum number of steps in the simulation.
        config['green_duration'],  # Duration of green traffic light phases.
        config['yellow_duration'],  # Duration of yellow traffic light phases.
        config['num_states'],  # Number of input states for the model.
        config['num_actions']  # Number of actions the model can take.
    )

    # Print a message indicating the start of the test episode.
    print('\n----- Test episode')

    # Run the simulation with the specified random seed and get the total simulation time.
    simulation_time = Simulation.run(config['episode_seed'])  # `episode_seed` ensures reproducibility.
    print('Simulation time:', simulation_time, 's')  # Output the total time taken for the simulation.

    # Print the path where the test results are saved.
    print("----- Testing info saved at:", plot_path)

    # Copy the configuration file to the results directory for documentation purposes.
    copyfile(src='testing_settings.ini', dst=os.path.join(plot_path, 'testing_settings.ini'))

    # Save and plot the rewards collected during the test episode.
    Visualization.save_data_and_plot(
        data=Simulation.reward_episode,  # List of rewards collected at each step.
        filename='reward',  # Name of the file to save.
        xlabel='Action step',  # Label for the x-axis of the plot.
        ylabel='Reward'  # Label for the y-axis of the plot.
    )

    # Save and plot the queue lengths recorded during the test episode.
    Visualization.save_data_and_plot(
        data=Simulation.queue_length_episode,  # List of queue lengths at each step.
        filename='queue',  # Name of the file to save.
        xlabel='Step',  # Label for the x-axis of the plot.
        ylabel='Queue length (vehicles)'  # Label for the y-axis of the plot.
    )
