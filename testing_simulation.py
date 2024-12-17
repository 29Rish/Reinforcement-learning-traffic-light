import traci  # Python interface for the SUMO traffic simulator
import numpy as np  # Used for numerical operations and state representation
import random  # Used for random number generation (if needed)
import timeit  # For measuring the simulation runtime
import os  # For interacting with the operating system

# Traffic light phase codes from the SUMO environment configuration file
PHASE_NS_GREEN = 0  # North-South green light
PHASE_NS_YELLOW = 1  # North-South yellow light
PHASE_NSL_GREEN = 2  # North-South left-turn green light
PHASE_NSL_YELLOW = 3  # North-South left-turn yellow light
PHASE_EW_GREEN = 4  # East-West green light
PHASE_EW_YELLOW = 5  # East-West yellow light
PHASE_EWL_GREEN = 6  # East-West left-turn green light
PHASE_EWL_YELLOW = 7  # East-West left-turn yellow light


class Simulation:
    """
    This class handles the simulation of a traffic intersection using SUMO and a trained model.
    """

    def __init__(self, Model, TrafficGen, sumo_cmd, max_steps, green_duration, yellow_duration, num_states, num_actions):
        """
        Initialize the Simulation class with required parameters and objects.
        """
        self._Model = Model  # Trained model for decision-making
        self._TrafficGen = TrafficGen  # Traffic generator for creating vehicle scenarios
        self._step = 0  # Step counter for the simulation
        self._sumo_cmd = sumo_cmd  # Command to start the SUMO simulation
        self._max_steps = max_steps  # Maximum number of simulation steps
        self._green_duration = green_duration  # Duration for green light phases
        self._yellow_duration = yellow_duration  # Duration for yellow light phases
        self._num_states = num_states  # Number of states in the environment
        self._num_actions = num_actions  # Number of actions the model can choose from
        self._reward_episode = []  # List to store rewards during the episode
        self._queue_length_episode = []  # List to store queue lengths during the episode

    def run(self, episode):
        """
        Runs the simulation for a single episode.
        """
        start_time = timeit.default_timer()  # Start timing the simulation

        # Generate a route file for this episode and start the SUMO simulation
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # Initialize variables
        self._step = 0
        self._waiting_times = {}  # Dictionary to track waiting times of cars
        old_total_wait = 0  # Previous total waiting time
        old_action = -1  # Placeholder for the previous action

        # Simulation loop
        while self._step < self._max_steps:
            # Get the current state of the intersection
            current_state = self._get_state()

            # Calculate the reward for the previous action
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            # Choose the next action based on the current state
            action = self._choose_action(current_state)

            # If the action changes, activate the yellow phase before switching
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # Activate the green phase for the chosen action
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # Update variables for the next step
            old_action = action
            old_total_wait = current_total_wait

            # Save the reward for the current step
            self._reward_episode.append(reward)

        # End the simulation and record the total runtime
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time

    def _simulate(self, steps_todo):
        """
        Simulate the given number of steps in SUMO.
        """
        if (self._step + steps_todo) >= self._max_steps:
            # Ensure we don't exceed the maximum number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # Simulate one step in SUMO
            self._step += 1  # Increment the step counter
            steps_todo -= 1
            queue_length = self._get_queue_length()  # Get the current queue length
            self._queue_length_episode.append(queue_length)

    def _collect_waiting_times(self):
        """
        Retrieve the cumulative waiting time of all cars in incoming lanes.
        """
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # Get the road where the car is located
            if road_id in incoming_roads:
                self._waiting_times[car_id] = wait_time  # Track cars in incoming roads
            elif car_id in self._waiting_times:
                del self._waiting_times[car_id]  # Remove cars that have left the intersection

        # Calculate total waiting time for all tracked cars
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time

    def _choose_action(self, state):
        """
        Use the model to choose the best action based on the current state.
        """
        return np.argmax(self._Model.predict_one(state))

    def _set_yellow_phase(self, old_action):
        """
        Activate the yellow light phase for the given action.
        """
        yellow_phase_code = old_action * 2 + 1  # Calculate the yellow phase code
        traci.trafficlight.setPhase("TL", yellow_phase_code)

    def _set_green_phase(self, action_number):
        """
        Activate the green light phase for the given action.
        """
        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)

    def _get_queue_length(self):
        """
        Retrieve the number of stationary vehicles (speed = 0) in incoming lanes.
        """
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length

    def _get_state(self):
        """
        Retrieve the state of the intersection as an array representing cell occupancy.
        """
        state = np.zeros(self._num_states)
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_pos = 750 - lane_pos  # Reverse position to represent proximity to traffic light

            # Map lane positions to cells
            if lane_pos < 7:
                lane_cell = 0
            elif lane_pos < 14:
                lane_cell = 1
            elif lane_pos < 21:
                lane_cell = 2
            elif lane_pos < 28:
                lane_cell = 3
            elif lane_pos < 40:
                lane_cell = 4
            elif lane_pos < 60:
                lane_cell = 5
            elif lane_pos < 100:
                lane_cell = 6
            elif lane_pos < 160:
                lane_cell = 7
            elif lane_pos < 400:
                lane_cell = 8
            elif lane_pos <= 750:
                lane_cell = 9

            # Determine the lane group
            if lane_id in ["W2TL_0", "W2TL_1", "W2TL_2"]:
                lane_group = 0
            elif lane_id == "W2TL_3":
                lane_group = 1
            elif lane_id in ["N2TL_0", "N2TL_1", "N2TL_2"]:
                lane_group = 2
            elif lane_id == "N2TL_3":
                lane_group = 3
            elif lane_id in ["E2TL_0", "E2TL_1", "E2TL_2"]:
                lane_group = 4
            elif lane_id == "E2TL_3":
                lane_group = 5
            elif lane_id in ["S2TL_0", "S2TL_1", "S2TL_2"]:
                lane_group = 6
            elif lane_id == "S2TL_3":
                lane_group = 7
            else:
                lane_group = -1  # Invalid lane group

            if lane_group >= 0:
                car_position = int(str(lane_group) + str(lane_cell))  # Combine group and cell into a unique position
                state[car_position] = 1  # Mark the cell as occupied

        return state

    # Properties for accessing rewards and queue lengths
    @property
    def queue_length_episode(self):
        return self._queue_length_episode

    @property
    def reward_episode(self):
        return self._reward_episode
