<h1 align="center"> Traffic Light Optimization With Reinforcement Learning </h1>

This project replicates a framework for optimizing traffic flow at complex intersections using a Deep Q-Learning Reinforcement Learning agent. By intelligently selecting traffic light phases, the agent aims to enhance traffic efficiency.

## Deep Q-Learning Agent

- **Framework**: Q-Learning integrated with a deep neural network.
- **Context**: Focused on traffic signal control at a single intersection.
- **Environment**: Simulates a 4-way intersection with 4 incoming and outgoing lanes per arm, each 750 meters long. Traffic lights control dedicated lanes for specific movements such as straight, left, and right turns.
- **Traffic Generation**: Each episode introduces 1000 vehicles with dynamic patterns, increasing environmental complexity.
- **Agent (Traffic Signal Control System - TLCS)**:
  - **State**: Discretized representation of oncoming lanes, identifying vehicle positions in defined cells.
  - **Action**: Selection of traffic light phases from predefined options, each lasting 10 seconds.
  - **Reward**: Rewards are based on cumulative waiting time reduction, encouraging efficient traffic flow management.
  - **Learning Mechanism**: Employs the Q-learning algorithm and a deep neural network to update action values and learn state-action mappings.

## Getting Started

Follow these steps to set up the project on your system:

1. Download and install Anaconda from the [official website](https://www.anaconda.com/distribution/#download-section).
2. Download and install SUMO from the [official website](https://www.dlr.de/ts/en/desktopdefault.aspx/tabid-9883/16931_read-41000/).

Ensure your system meets the following software version requirements:

- Python 3.11
- SUMO traffic simulator 1.2.0
- tensorflow 2.16.1
- tensorflow-gpu 2.5.0
- matplotlib 3.6.0
- numpy 1.19.5

## Running the Algorithm

1. Clone or download the repository.
2. Open a terminal and navigate to the root folder.
3. Execute `python training_main.py` to initiate the training process.

## Code Structure

The project is organized into the following components, each handling a specific aspect of the training process:

- **Model**: Defines the architecture of the deep neural network.
- **Memory**: Implements experience replay to stabilize training.
- **Simulation**: Manages the simulation environment and its interaction with SUMO.
- **TrafficGenerator**: Creates vehicle routes for each episode.
- **Visualization**: Includes functions for plotting and visualizing data.
- **Utils**: Contains utility functions for managing files and directories.

## Settings Explained

Training and testing configurations are specified in the `training_settings.ini` and `testing_settings.ini` files. Key parameters include episode duration, traffic generation details, neural network settings, and simulation configurations.

## Contributors

This project was replicated and documented by:

- **Rishita Yadav**  
  Columbia University, New York, NY, USA  

For questions or additional information, please open an issue on the repository's issues page.

### References

- Amazon AWS Documentation
- Dr. Rahee Walambe - Symbiosis Institute of Technology, Symbiosis International (Deemed) University
- Dr. Smita Mahajan - Symbiosis Institute of Technology, Symbiosis International (Deemed) University
- Analytics Vidhya
- Towards Data Science (Medium)
- Andrea Vidali - University of Milano-Bicocca
- "Deep Reinforcement Learning for Traffic Light Control in Intelligent Transportation Systems" - Xiao-Yang Liu, Ming Zhu, Sem Borst, Anwar Walid
