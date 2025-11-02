# backend/server.py

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .models.video_data import VIDEOS
from .rl_model import VideoQModel
import numpy as np
import pandas as pd  # <-- Make sure this is imported

app = FastAPI()

# --- Model and Session Initialization ---
model = VideoQModel(n_videos=len(VIDEOS))
sessions = {}  
recent_rewards = []
# ----------------------------------------

# Enable CORS...
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Feedback(BaseModel):
    session_id: str
    video_id: int
    feedback: int  # -1, 0, or +1


@app.get("/recommend")
def recommend(session_id: str = Query(...)):
    if session_id not in sessions:
        state = model.get_initial_state()
        sessions[session_id] = state
    else:
        state = sessions[session_id]

    action = model.select_action(state)
    video = VIDEOS[action]
    return video


@app.post("/feedback")
def feedback(data: Feedback):
    if data.session_id not in sessions:
        state = model.get_initial_state()
    else:
        state = sessions[data.session_id]

    action_index = data.video_id - 1
    reward = data.feedback
    
    recent_rewards.append(reward)
    if len(recent_rewards) > 100:
        recent_rewards.pop(0) 

    new_state = model.update_q_table(state, action_index, reward)
    sessions[data.session_id] = new_state
    
    return {"status": "ok", "reward": reward}


# --- NEW UPGRADED /stats ENDPOINT ---
@app.get("/stats")
def get_stats():
    """
    Provides simple, real-time stats on model performance,
    PLUS key insights from the Q-table.
    """
    
    # --- Part 1: Basic Stats (from recent rewards) ---
    stats_dict = {}
    if not recent_rewards:
        stats_dict["message"] = "No feedback received yet."
    else:
        total_feedback = len(recent_rewards)
        avg_reward = np.mean(recent_rewards)
        likes = recent_rewards.count(1)
        neutrals = recent_rewards.count(0)
        dislikes = recent_rewards.count(-1)
        ctr = (likes + neutrals) / total_feedback if total_feedback > 0 else 0
        
        stats_dict = {
            "average_reward_last_100": f"{avg_reward:.4f}",
            "click_through_rate_approx": f"{ctr:.2%}",
            "recent_feedback_count": total_feedback,
            "likes": likes,
            "neutrals": neutrals,
            "dislikes": dislikes
        }

    # --- Part 2: Advanced Insights (from the Q-table) ---
    try:
        # Create labels
        video_titles = [v['title'] for v in VIDEOS]
        state_labels = video_titles + ["Initial-State"]
        action_labels = video_titles
        
        # Access the model's tables
        q_table = model.q_table
        counts_table = model.action_counts
        
        # 1. Get the best action from the "Initial-State"
        # The initial state is the last row in the q_table
        initial_state_row = q_table[model.initial_state]
        best_initial_action_index = np.argmax(initial_state_row)
        best_initial_action_value = initial_state_row[best_initial_action_index]
        best_initial_action_title = action_labels[best_initial_action_index]
        
        stats_dict["best_initial_video"] = f"'{best_initial_action_title}' (Value: {best_initial_action_value:.4f})"
        
        # 2. Find the most-explored state-action pair
        max_count = int(counts_table.max()) # Use int() for JSON compatibility
        max_idx = np.unravel_index(np.argmax(counts_table), counts_table.shape)
        state_label = state_labels[max_idx[0]]
        action_label = action_labels[max_idx[1]]
        
        stats_dict["most_tried_recommendation"] = f"After '{state_label}', recommended '{action_label}' ({max_count} times)"

    except Exception as e:
        stats_dict["advanced_stats_error"] = str(e)

    return stats_dict
# --- END OF UPGRADED /stats ENDPOINT ---


@app.get("/q_table")
def get_q_table():
    """
    Returns the current Q-table and Action Counts as JSON.
    This is for live inspection of the model's "brain".
    """
    # Create labels just like in inspect_model.py
    video_titles = [v['title'] for v in VIDEOS]
    state_labels = video_titles + ["Initial-State"]
    action_labels = video_titles

    # Access the model's in-memory tables
    q_table = model.q_table
    counts_table = model.action_counts

    # Use pandas for easy formatting
    q_df = pd.DataFrame(q_table, index=state_labels, columns=action_labels)
    c_df = pd.DataFrame(counts_table, index=state_labels, columns=action_labels)
    
    # Return as JSON-compatible dictionaries
    return {
        "q_table": q_df.to_dict(),
        "counts_table": c_df.to_dict()
    }