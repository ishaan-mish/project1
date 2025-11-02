# backend/train.py
import gymnasium as gym
import numpy as np
import os
import torch
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from .models.video_data import VIDEOS # <-- CHANGED

# --- 1. Define Simulated User Preferences ---
# (Rest of the file is identical)
# ... (all the code you had before) ...

# --- 1. Define Simulated User Preferences ---
# This is the "ground truth" for our simulation.
# We create a simple preference matrix.
# Rows: States (Last video watched, 0-4 = videos, 5 = initial state)
# Cols: Actions (Next video to recommend, 0-4)
# Values: Simulated reward (-1, 0, +1)
N_VIDEOS = len(VIDEOS)
INITIAL_STATE = N_VIDEOS # State 5 is the "start" state

# Example preferences:
# - People who start (state 5) like "Funny Cats" (action 1)
# - People who watch "Funny Cats" (state 1) like "Nature Sounds" (action 4)
# - People who watch "Relaxing Music" (state 0) dislike "Motivational Speech" (action 3)
SIMULATED_REWARDS = np.array([
#   Action: 0(Music), 1(Cats), 2(Space), 3(Speech), 4(Nature)
    [ 0,  1,  1, -1,  1], # State 0 (Music)
    [ 1,  0,  0,  1,  1], # State 1 (Cats)
    [-1,  1,  0,  1,  0], # State 2 (Space)
    [ 1,  1,  1,  0, -1], # State 3 (Speech)
    [ 1,  1, -1, -1,  0], # State 4 (Nature)
    [ 0,  1,  1,  0,  0]  # State 5 (Initial)
])


# --- 2. Create a Custom Gym Environment ---
class OfflineVideoEnv(gym.Env):
    """
    A simulated Gym environment for pre-training the recommender.
    It uses the SIMULATED_REWARDS matrix to generate rewards.
    """
    metadata = {'render.modes': ['console']}

    def __init__(self, n_videos, reward_matrix):
        super(OfflineVideoEnv, self).__init__()
        
        self.n_videos = n_videos
        self.reward_matrix = reward_matrix
        
        # State: Last video watched (0 to n_videos-1) + initial state
        self.initial_state = self.n_videos
        
        # Define action and observation space
        # Action: Recommend one of the N videos
        self.action_space = spaces.Discrete(self.n_videos)
        # Observation: The current state (last video watched)
        self.observation_space = spaces.Discrete(self.n_videos + 1)
        
        self.state = self.initial_state

    def reset(self, seed=None, options=None):
        """Resets the environment to the initial state."""
        super().reset(seed=seed)
        self.state = self.initial_state
        return self.state, {} # Return observation and info dict

    def step(self, action):
        """Take an action and return the new state, reward, etc."""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        # Get reward from our simulated preference matrix
        reward = self.reward_matrix[self.state, action]
        
        # The new state is the video that was just "watched" (the action)
        self.state = action 

        # In this simulation, the episode never truly "ends"
        terminated = False
        truncated = False
        info = {}

        return self.state, reward, terminated, truncated, info

    def render(self):
        print(f"Current State (Last Video): {self.state}")


# --- 3. Training and Saving Function ---
def train_and_save():
    """
    Trains a DQN model and saves its learned Q-values to disk
    in the format our live server expects.
    """
    print("--- Starting Offline Pre-Training ---")
    
    # 1. Initialize the environment
    env = OfflineVideoEnv(N_VIDEOS, SIMULATED_REWARDS)
    # Wrap it for Stable-Baselines
    env = DummyVecEnv([lambda: env])

    # 2. Initialize the DQN Model
    # "MlpPolicy" is a standard multi-layer perceptron network
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.001,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.9, # Use the same gamma as our online model
        device="auto"
    )

    # 3. Train the model
    print("\nTraining DQN model...")
    model.learn(total_timesteps=20000, log_interval=10)
    print("Training complete.")

    # 4. Extract the Q-table from the trained model
    print("\nExtracting Q-values from the neural network...")
    
    # Get the Q-network from the model
    q_net = model.q_net
    q_table_shape = (N_VIDEOS + 1, N_VIDEOS)
    q_table = np.zeros(q_table_shape)

    # We must loop through each state and ask the network
    # what it thinks the Q-values are for all actions in that state.
    for s in range(N_VIDEOS + 1):
        # Create a "batch" of 1 state
        obs = torch.tensor([s], device=model.device)
        
        # Get Q-values from the network
        with torch.no_grad():
            q_values = q_net(obs)
        
        # Store them in our table
        q_table[s] = q_values.cpu().numpy()[0]

    # 5. Save the Q-table
    q_table_path = "q_table.npy"
    np.save(q_table_path, q_table)
    print(f"✅ Successfully saved Q-table to {q_table_path}")

    # 6. Initialize the action_counts file
    # Our online UCB model needs this file to exist.
    # We'll initialize it with '1' instead of '0'
    # to encourage exploration (prevents division by zero).
    counts_path = "action_counts.npy"
    counts_table = np.ones(q_table_shape, dtype=int)
    np.save(counts_path, counts_table)
    print(f"✅ Successfully saved action counts to {counts_path}")
    print("\n--- Offline Pre-Training Complete ---")


# --- 4. Run the script ---
if __name__ == "__main__":
    train_and_save()