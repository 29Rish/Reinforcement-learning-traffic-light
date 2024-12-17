import configparser
from sumolib import checkBinary
import os
import sys


def import_train_configuration(config_file):
    """
    Read the configuration file related to training settings and import its content into a dictionary.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        dict: Dictionary containing training configuration parameters.
    """
    content = configparser.ConfigParser()  # Initialize configparser
    content.read(config_file)  # Read the configuration file
    config = {}

    # Reading parameters from the 'simulation' section
    config['gui'] = content['simulation'].getboolean('gui')  # Run in GUI mode or not
    config['total_episodes'] = content['simulation'].getint('total_episodes')  # Total training episodes
    config['max_steps'] = content['simulation'].getint('max_steps')  # Max simulation steps per episode
    config['n_cars_generated'] = content['simulation'].getint('n_cars_generated')  # Number of cars generated per episode
    config['green_duration'] = content['simulation'].getint('green_duration')  # Duration of green lights
    config['yellow_duration'] = content['simulation'].getint('yellow_duration')  # Duration of yellow lights

    # Reading parameters from the 'model' section
    config['num_layers'] = content['model'].getint('num_layers')  # Number of neural network layers
    config['width_layers'] = content['model'].getint('width_layers')  # Width of each neural network layer
    config['batch_size'] = content['model'].getint('batch_size')  # Batch size for training
    config['learning_rate'] = content['model'].getfloat('learning_rate')  # Learning rate for training
    config['training_epochs'] = content['model'].getint('training_epochs')  # Training epochs per session

    # Reading parameters from the 'memory' section
    config['memory_size_min'] = content['memory'].getint('memory_size_min')  # Minimum memory buffer size
    config['memory_size_max'] = content['memory'].getint('memory_size_max')  # Maximum memory buffer size

    # Reading parameters from the 'agent' section
    config['num_states'] = content['agent'].getint('num_states')  # Number of states for the RL agent
    config['num_actions'] = content['agent'].getint('num_actions')  # Number of possible actions
    config['gamma'] = content['agent'].getfloat('gamma')  # Discount factor for future rewards

    # Reading paths from the 'dir' section
    config['models_path_name'] = content['dir']['models_path_name']  # Directory path for saving models
    config['sumocfg_file_name'] = content['dir']['sumocfg_file_name']  # SUMO configuration file name

    return config


def import_test_configuration(config_file):
    """
    Read the configuration file related to testing settings and import its content into a dictionary.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        dict: Dictionary containing testing configuration parameters.
    """
    content = configparser.ConfigParser()  # Initialize configparser
    content.read(config_file)  # Read the configuration file
    config = {}

    # Reading parameters from the 'simulation' section
    config['gui'] = content['simulation'].getboolean('gui')  # Run in GUI mode or not
    config['max_steps'] = content['simulation'].getint('max_steps')  # Max simulation steps per episode
    config['n_cars_generated'] = content['simulation'].getint('n_cars_generated')  # Number of cars generated per episode
    config['episode_seed'] = content['simulation'].getint('episode_seed')  # Seed for reproducible test simulations
    config['green_duration'] = content['simulation'].getint('green_duration')  # Duration of green lights
    config['yellow_duration'] = content['simulation'].getint('yellow_duration')  # Duration of yellow lights

    # Reading parameters from the 'agent' section
    config['num_states'] = content['agent'].getint('num_states')  # Number of states for the RL agent
    config['num_actions'] = content['agent'].getint('num_actions')  # Number of possible actions

    # Reading paths from the 'dir' section
    config['sumocfg_file_name'] = content['dir']['sumocfg_file_name']  # SUMO configuration file name
    config['models_path_name'] = content['dir']['models_path_name']  # Directory path for saved models
    config['model_to_test'] = content['dir'].getint('model_to_test')  # Model version number to test

    return config


def set_sumo(gui, sumocfg_file_name, max_steps):
    """
    Configure the SUMO environment based on user input.

    Args:
        gui (bool): If True, SUMO runs in GUI mode, otherwise in command-line mode.
        sumocfg_file_name (str): SUMO configuration file name.
        max_steps (int): Maximum number of simulation steps.

    Returns:
        list: SUMO command with the appropriate parameters.
    """
    # Verify that the SUMO_HOME environment variable is set
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')  # Path to SUMO tools
        sys.path.append(tools)
    else:
        sys.exit("Please declare environment variable 'SUMO_HOME'")  # Exit if SUMO_HOME is not set

    # Determine whether to use SUMO or SUMO-GUI
    if gui is False:
        sumoBinary = checkBinary('sumo')  # Command-line mode
    else:
        sumoBinary = checkBinary('sumo-gui')  # GUI mode

    # Prepare the SUMO command
    sumo_cmd = [sumoBinary, "-c", os.path.join('intersection', sumocfg_file_name),
                "--no-step-log", "true", "--waiting-time-memory", str(max_steps)]
    return sumo_cmd


def set_train_path(models_path_name):
    """
    Create a new model directory path for training with an incremental version number.

    Args:
        models_path_name (str): Base directory name to save models.

    Returns:
        str: Path to the newly created model directory.
    """
    # Set the base directory path for models
    models_path = os.path.join(os.getcwd(), models_path_name, '')
    os.makedirs(os.path.dirname(models_path), exist_ok=True)  # Ensure the base directory exists

    # Get the list of existing model folders
    dir_content = os.listdir(models_path)
    if dir_content:  # Check if any model directories already exist
        previous_versions = [int(name.split("_")[1]) for name in dir_content]  # Extract version numbers
        new_version = str(max(previous_versions) + 1)  # Increment the version
    else:
        new_version = '1'  # If no versions exist, start from version 1

    # Create a path for the new version
    data_path = os.path.join(models_path, 'model_' + new_version, '')
    os.makedirs(os.path.dirname(data_path), exist_ok=True)  # Create the directory
    return data_path


def set_test_path(models_path_name, model_n):
    """
    Identify the path for the specified model version and create a 'test' directory inside it.

    Args:
        models_path_name (str): Base directory containing models.
        model_n (int): Model version number to test.

    Returns:
        tuple: Paths to the model folder and the 'test' sub-directory.

    Raises:
        SystemExit: If the specified model version does not exist.
    """
    # Path to the specific model folder
    model_folder_path = os.path.join(os.getcwd(), models_path_name, 'model_' + str(model_n), '')

    # Check if the model directory exists
    if os.path.isdir(model_folder_path):
        plot_path = os.path.join(model_folder_path, 'test', '')  # Create a 'test' directory inside the model folder
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)  # Ensure the directory exists
        return model_folder_path, plot_path
    else:
        sys.exit('The model number specified does not exist in the models folder')  # Exit if the model does not exist
