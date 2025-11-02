# backend/rl_model.py
import numpy as np
import os

class VideoQModel:
    """
    Manages the shared Q-table and action counts for a stateful RL recommender.
    This model implements Q-learning with UCB for exploration.
    
    - State: The last video watched (index).
    - Action: The next video to recommend (index).
    """

    def __init__(self, n_videos, alpha=0.1, gamma=0.9, c=2):
        self.n_videos = n_videos
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor (for future rewards)
        self.c = c              # UCB exploration parameter

        # The initial state (e.g., "no video watched yet")
        # We use 'n_videos' as the index for this special state.
        self.initial_state = self.n_videos 
        
        # Q-table: (n_states, n_actions)
        # We have n_videos states + 1 initial state
        self.q_table_shape = (self.n_videos + 1, self.n_videos)
        
        # Action counts: (n_states, n_actions)
        # Tracks n(s, a) - how many times action 'a' was taken in state 's'
        self.counts_shape = (self.n_videos + 1, self.n_videos)

        # File paths for persistence
        self.q_table_path = "q_table.npy"
        self.counts_path = "action_counts.npy"

        # Load data from disk
        self.q_table, self.action_counts = self._load_tables()

# In backend/rl_model.py

# In backend/rl_model.py

    def _load_tables(self):
        """Loads Q-table and action counts from .npy files if they exist."""
        q_table = None
        action_counts = None
        
        # Check if files exist first
        q_file_exists = os.path.exists(self.q_table_path)
        c_file_exists = os.path.exists(self.counts_path)

        if q_file_exists and c_file_exists:
            print(f"🔁 Found existing tables. Loading {self.q_table_path} and {self.counts_path}...")
            q_table = np.load(self.q_table_path)
            action_counts = np.load(self.counts_path)
            
            # --- RESET LOGIC ---
            if q_table.shape != self.q_table_shape or action_counts.shape != self.counts_shape:
                print(f"🚨 WARNING: Model shape mismatch detected!")
                print(f"    Expected Q-Table shape: {self.q_table_shape}")
                print(f"    Found Q-Table shape:    {q_table.shape}")
                print(f"    This happens if you add or remove videos.")
                print(f"    Deleting old model files and re-initializing.")
                
                os.remove(self.q_table_path)
                os.remove(self.counts_path)
                
                q_table = None
                action_counts = None
            else:
                print(f"✅ Model shapes match. Loaded successfully.")

        if q_table is None or action_counts is None:
            if q_file_exists or c_file_exists:
                pass # The warning was already printed above
            else:
                print(f"🆕 No existing tables found. Starting fresh.")
                
            # Create new tables as local variables
            q_table = np.zeros(self.q_table_shape)
            action_counts = np.zeros(self.counts_shape)
            action_counts.fill(1) 
            
            # --- THE FIX IS HERE ---
            # Save the new tables directly instead of calling self._save_tables()
            print(f"💾 Saving new, initialized tables to disk...")
            np.save(self.q_table_path, q_table)
            np.save(self.counts_path, action_counts)
            # --- END OF FIX ---
            
        return q_table, action_counts

    def _save_tables(self):
        """Saves the Q-table and action counts to disk."""
        np.save(self.q_table_path, self.q_table)
        np.save(self.counts_path, self.action_counts)

    def get_initial_state(self):
        """Returns the identifier for the starting state."""
        return self.initial_state

    def select_action(self, state):
        """
        Selects an action using the Upper Confidence Bound (UCB1) algorithm.
        This provides a balance between exploration and exploitation.
        """
        
        # Get the total number of recommendations made *from this state*
        total_state_count = np.sum(self.action_counts[state])

        ucb_scores = np.zeros(self.n_videos)

        for action in range(self.n_videos):
            count = self.action_counts[state, action]

            if count == 0:
                # This action has never been tried in this state.
                # Prioritize it by giving it an infinite score.
                return action 

            # UCB formula: Q(s,a) + c * sqrt(log(N(s)) / n(s,a))
            q_value = self.q_table[state, action]
            
            # Add 1 to total_state_count to avoid log(0) if it's the very first pull
            bonus = self.c * np.sqrt(np.log(total_state_count + 1) / count)
            
            ucb_scores[action] = q_value + bonus

        # Choose the action with the highest UCB score
        return int(np.argmax(ucb_scores))

    def update_q_table(self, state, action, reward):
        """
        Updates the Q-table based on user feedback using the Q-learning rule.
        Returns the new state.
        """
        
        # --- 1. Update Counts ---
        self.action_counts[state, action] += 1

        # --- 2. Define Q-Learning Components ---
        s = state
        a = action
        r = reward
        
        # The new state 's_prime' is the action (video) we just took.
        s_prime = action 

        # --- 3. Q-Learning Update Rule ---
        # Q(s,a) = Q(s,a) + alpha * (r + gamma * max(Q(s',a')) - Q(s,a))
        
        old_q = self.q_table[s, a]
        
        # Find the best possible Q-value from the *next* state
        max_future_q = np.max(self.q_table[s_prime])
        
        # Calculate the new Q-value
        new_q = old_q + self.alpha * (r + self.gamma * max_future_q - old_q)
        
        self.q_table[s, a] = new_q

        # --- 4. Save and Return ---
        self._save_tables()

        # Return the new state for the session
        return s_prime