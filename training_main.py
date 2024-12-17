# Enable compatibility with Python 2 and 3
from __future__ import absolute_import
from __future__ import print_function

# Standard libraries
import os  # Used for file and directory operations
import datetime  # For handling dates and times
from shutil import copyfile  # To copy files

# Custom modules
from training_simulation import Simulation  # Manages the simulation process
from generator import TrafficGenerator  # Generates traffic data for SUMO
from memory import Memory  # Implements experience replay memory for RL
from model import TrainModel  # Defines the neural network model for training
from visualization import Visualization  # Handles data visualization
from utils import import_train_configuration, set_sumo, set_train_path  # Utility functions

if __name__ == "__main__":

    # TensorFlow imports
    import tensorflow as tf
    from tensorflow.keras.losses import MeanSquaredError  # Loss function for regression
    from tensorflow.keras.optimizers import Adam  # Optimizer for neural network training

    # Print GPU availability
    print("###################\nNum GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    print(tf.config.list_physical_devices('GPU'))

    # Configure TensorFlow to use GPUs efficiently
    if len(tf.config.list_physical_devices('GPU')) > 0:
        print("TensorFlow is using GPU.")
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)  # Allow dynamic memory growth
    else:
        print("TensorFlow is NOT using GPU.")

    # Import training configuration from settings file
    config = import_train_configuration(config_file='training_settings.ini')

    # Set up SUMO simulation command and model saving path
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    path = set_train_path(config['models_path_name'])

    # Initialize the neural network model with specified parameters
    Model = TrainModel(
        config['num_layers'], 
        config['width_layers'], 
        config['batch_size'], 
        config['learning_rate'], 
        input_dim=config['num_states'], 
        output_dim=config['num_actions']
    )

    # Ensure the model is compiled with the appropriate loss function and optimizer
    # Inside TrainModel class: model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=self._learning_rate))

    # Initialize experience replay memory
    Memory = Memory(
        config['memory_size_max'], 
        config['memory_size_min']
    )

    # Initialize traffic generator for creating vehicle flows
    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )

    # Initialize visualization tools for plotting results
    Visualization = Visualization(
        path, 
        dpi=96  # Resolution for plots
    )
        
    # Initialize the simulation environment
    Simulation = Simulation(
        Model,
        Memory,
        TrafficGen,
        sumo_cmd,
        config['gamma'],  # Discount factor for future rewards
        config['max_steps'], 
        config['green_duration'],  # Duration of green light phases
        config['yellow_duration'],  # Duration of yellow light phases
        config['num_states'],  # Number of states in the environment
        config['num_actions'],  # Number of possible actions
        config['training_epochs']  # Number of training epochs
    )
    
    # Start training loop
    episode = 0
    timestamp_start = datetime.datetime.now()  # Record the start time
    
    while episode < config['total_episodes']:
        print('\n----- Episode', str(episode+1), 'of', str(config['total_episodes']))
        
        # Update epsilon for epsilon-greedy policy
        epsilon = 1.0 - (episode / config['total_episodes'])  
        
        # Run the simulation for the current episode
        simulation_time, training_time = Simulation.run(episode, epsilon)
        
        # Print timing details for the episode
        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:', round(simulation_time+training_time, 1), 's')
        
        episode += 1

    # Print the start and end times of the training process
    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)

    # Save the trained model to the specified path
    Model.save_model(path)

    # Copy the training configuration file to the saved model directory
    copyfile(src='training_settings.ini', dst=os.path.join(path, 'training_settings.ini'))

    # Save and plot training results
    Visualization.save_data_and_plot(data=Simulation.reward_store, filename='reward', xlabel='Episode', ylabel='Cumulative negative reward')
    Visualization.save_data_and_plot(data=Simulation.cumulative_wait_store, filename='delay', xlabel='Episode', ylabel='Cumulative delay (s)')
    Visualization.save_data_and_plot(data=Simulation.avg_queue_length_store, filename='queue', xlabel='Episode', ylabel='Average queue length (vehicles)')
