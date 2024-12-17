import traci
import numpy as np
import random
import timeit
import os

# Phase codes based on environment.net.xml
# These phases correspond to the green and yellow light phases for the traffic lights.
PHASE_NS_GREEN = 0  # North-South Green phase (action 0, binary 00)
PHASE_NS_YELLOW = 1  # North-South Yellow phase
PHASE_NSL_GREEN = 2  # North-South Left-turn Green phase (action 1, binary 01)
PHASE_NSL_YELLOW = 3  # North-South Left-turn Yellow phase
PHASE_EW_GREEN = 4  # East-West Green phase (action 2, binary 10)
PHASE_EW_YELLOW = 5  # East-West Yellow phase
PHASE_EWL_GREEN = 6  # East-West Left-turn Green phase (action 3, binary 11)
PHASE_EWL_YELLOW = 7  # East-West Left-turn Yellow phase


class Simulation:
    def __init__(self, Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_actions, training_epochs):
        """
        Initialize the Simulation class.
        
        Parameters:
        - Model: The neural network model used for decision-making.
        - Memory: Replay memory to store experience tuples.
        - TrafficGen: Traffic generator class for creating routes.
        - sumo_cmd: Command to launch the SUMO simulator.
        - gamma: Discount factor for the Q-learning algorithm.
        - max_steps: Maximum number of steps per episode.
        - green_duration: Duration of green lights for traffic phases.
        - yellow_duration: Duration of yellow lights during phase changes.
        - num_states: Size of the state space.
        - num_actions: Number of possible actions (phases).
        - training_epochs: Number of training epochs per episode.
        """
        self._Model = Model
        self._Memory = Memory
        self._TrafficGen = TrafficGen
        self._gamma = gamma  # Discount factor for future rewards
        self._step = 0  # Simulation step counter
        self._sumo_cmd = sumo_cmd  # SUMO command to start the simulation
        self._max_steps = max_steps  # Total steps allowed in one episode
        self._green_duration = green_duration  # Time duration for green phases
        self._yellow_duration = yellow_duration  # Time duration for yellow phases
        self._num_states = num_states  # Dimension of state representation
        self._num_actions = num_actions  # Number of actions (traffic light phases)
        self._reward_store = []  # Storage for total rewards per episode
        self._cumulative_wait_store = []  # Storage for cumulative waiting times
        self._avg_queue_length_store = []  # Storage for average queue lengths
        self._training_epochs = training_epochs  # Training epochs after each episode

    def run(self, episode, epsilon):
        """
        Run one simulation episode and train the model at the end.

        Parameters:
        - episode: The current episode number (used for random seed).
        - epsilon: Exploration rate for the epsilon-greedy policy.

        Returns:
        - simulation_time: Total time taken for the simulation.
        - training_time: Total time taken for training.
        """
        start_time = timeit.default_timer()

        # Generate traffic route file for the episode
        self._TrafficGen.generate_routefile(seed=episode)

        # Start the SUMO simulation
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # Initialize variables for tracking
        self._step = 0
        self._waiting_times = {}  # Stores waiting times for cars
        self._sum_neg_reward = 0  # Total negative rewards for the episode
        self._sum_queue_length = 0  # Total queue length across steps
        self._sum_waiting_time = 0  # Cumulative waiting time for all cars
        old_total_wait = 0
        old_state = -1  # Previous state (used in Q-learning update)
        old_action = -1  # Previous action taken

        while self._step < self._max_steps:
            # Get the current state of the traffic intersection
            current_state = self._get_state()

            # Calculate reward for the previous action: reduction in cumulative waiting time
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            # Save experience tuple into memory (state, action, reward, next_state)
            if self._step != 0:  # Skip saving for the first step
                self._Memory.add_sample((old_state, old_action, reward, current_state))

            # Select an action based on the current state and exploration rate epsilon
            action = self._choose_action(current_state, epsilon)

            # Change to yellow phase if the action changes
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # Activate the chosen green phase
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # Update variables for the next step
            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait

            # Accumulate negative rewards for performance tracking
            if reward < 0:
                self._sum_neg_reward += reward

        # Save episode statistics
        self._save_episode_stats()
        print("Total reward:", self._sum_neg_reward, "- Epsilon:", round(epsilon, 2))
        traci.close()

        # Measure simulation time
        simulation_time = round(timeit.default_timer() - start_time, 1)

        print("Training...")
        start_time = timeit.default_timer()
        for _ in range(self._training_epochs):
            self._replay()  # Perform training using replay memory
        training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time

    def _simulate(self, steps_todo):
        """
        Advance the SUMO simulation by the specified number of steps.
        Also updates cumulative statistics like queue length.

        Parameters:
        - steps_todo: Number of simulation steps to execute.
        """
        if (self._step + steps_todo) >= self._max_steps:
            steps_todo = self._max_steps - self._step  # Limit steps to max_steps

        while steps_todo > 0:
            traci.simulationStep()  # Advance the simulation by one step
            self._step += 1
            steps_todo -= 1

            # Measure queue length and waiting time
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length  # Queue length directly translates to waited seconds

    def _collect_waiting_times(self):
        """
        Calculate the total waiting time of all vehicles on incoming lanes.

        Returns:
        - total_waiting_time: Sum of waiting times for all cars in incoming lanes.
        """
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]  # Incoming roads to the intersection
        car_list = traci.vehicle.getIDList()  # List of vehicles currently in simulation
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)
            if road_id in incoming_roads:
                self._waiting_times[car_id] = wait_time
            else:
                # Remove cars that have cleared the intersection
                if car_id in self._waiting_times:
                    del self._waiting_times[car_id]

        # Total waiting time for all vehicles
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time
