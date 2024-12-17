import os
# Suppress TensorFlow log messages below error level to reduce clutter in the output.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import sys
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

class TrainModel:
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim):
        """
        Initialize the training model with given parameters.
        - num_layers: Number of hidden layers in the neural network.
        - width: Number of neurons in each hidden layer.
        - batch_size: Size of the batches used for training.
        - learning_rate: Learning rate for the optimizer.
        - input_dim: Dimension of the input features.
        - output_dim: Dimension of the output (e.g., number of actions).
        """
        self._input_dim = input_dim  # Store the input feature dimensions.
        self._output_dim = output_dim  # Store the output dimensions.
        self._batch_size = batch_size  # Batch size for training.
        self._learning_rate = learning_rate  # Learning rate for the optimizer.
        self._model = self._build_model(num_layers, width)  # Build and compile the model.

    def _build_model(self, num_layers, width):
        """
        Build and compile a fully connected feedforward neural network.
        - num_layers: Number of hidden layers.
        - width: Number of neurons per hidden layer.
        """
        # Define the input layer with the specified input dimensions.
        inputs = keras.Input(shape=(self._input_dim,))
        
        # Create the first hidden layer with ReLU activation.
        x = layers.Dense(width, activation='relu')(inputs)
        
        # Add the remaining hidden layers with the specified width and activation function.
        for _ in range(num_layers):
            x = layers.Dense(width, activation='relu')(x)
        
        # Define the output layer with a linear activation function.
        outputs = layers.Dense(self._output_dim, activation='linear')(x)

        # Build the model by connecting inputs and outputs.
        model = keras.Model(inputs=inputs, outputs=outputs, name='my_model')
        
        # Compile the model with mean squared error loss and Adam optimizer.
        model.compile(
            loss=MeanSquaredError(),
            optimizer=Adam(learning_rate=self._learning_rate)
        )
        return model

    def predict_one(self, state):
        """
        Predict the output (e.g., action values) for a single input state.
        - state: A single input state.
        """
        # Reshape the input to match the model's expected input dimensions.
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state, verbose=0)

    def predict_batch(self, states):
        """
        Predict the output (e.g., action values) for a batch of input states.
        - states: A batch of input states.
        """
        return self._model.predict(states, verbose=0)

    def train_batch(self, states, q_sa):
        """
        Train the model on a batch of input states and target outputs.
        - states: Batch of input states.
        - q_sa: Corresponding target outputs (e.g., updated Q-values for reinforcement learning).
        """
        self._model.fit(states, q_sa, epochs=1, verbose=0)

    def save_model(self, path):
        """
        Save the trained model and its architecture as a PNG image.
        - path: Path to the directory where the model will be saved.
        """
        self._model.save(os.path.join(path, 'trained_model.h5'))  # Save the model weights and architecture.
        plot_model(
            self._model, 
            to_file=os.path.join(path, 'model_structure.png'), 
            show_shapes=True, 
            show_layer_names=True
        )  # Save a visual representation of the model.

    @property
    def input_dim(self):
        # Get the input dimensions of the model.
        return self._input_dim

    @property
    def output_dim(self):
        # Get the output dimensions of the model.
        return self._output_dim

    @property
    def batch_size(self):
        # Get the batch size used for training.
        return self._batch_size


class TestModel:
    def __init__(self, input_dim, model_path):
        """
        Initialize the testing model by loading a pre-trained model.
        - input_dim: Dimension of the input features.
        - model_path: Path to the directory where the model is stored.
        """
        self._input_dim = input_dim  # Store the input feature dimensions.
        self._model = self._load_my_model(model_path)  # Load the pre-trained model.

    def _load_my_model(self, model_folder_path):
        """
        Load a pre-trained model from the specified folder.
        - model_folder_path: Path to the directory containing the model.
        """
        # Define the full path to the model file.
        model_file_path = os.path.join(model_folder_path, 'trained_model.h5')

        # Check if the model file exists, and load it if it does.
        if os.path.isfile(model_file_path):
            loaded_model = load_model(model_file_path)
            return loaded_model
        else:
            # Exit the program with an error message if the model file is not found.
            sys.exit("Model not found in the specified path: " + model_folder_path)

    def predict_one(self, state):
        """
        Predict the output (e.g., action values) for a single input state using the loaded model.
        - state: A single input state.
        """
        # Reshape the input to match the model's expected input dimensions.
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state, verbose=0)

    @property
    def input_dim(self):
        # Get the input dimensions of the model.
        return self._input_dim
