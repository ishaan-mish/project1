# app.py
import streamlit as st
import serial
import time
import csv
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import pickle
import requests  

# --- Import from our new eeg_model folder ---
import sys
sys.path.append('eeg_model')
from eeg_feature_extractor import generate_feature_vectors_from_samples

# --- Configuration ---
COM_PORT = 'COM19' # <-- UPDATE THIS TO YOUR COM PORT
BAUD_RATE = 115200
VIDEOS_DIR = 'videos' # <-- Name of our new videos folder

# EEG Model Config
FS = 256
FEATURE_EXTRACTION_BUFFER_LENGTH_SECONDS = 2
RAW_DATA_WINDOW_SIZE_FOR_FE = int(FEATURE_EXTRACTION_BUFFER_LENGTH_SECONDS * FS) 
CSV_FILENAME = 'temp_eeg.csv'
MODEL_PATH = os.path.join('eeg_model', 'model.h5')
SCALER_PATH = os.path.join('eeg_model', 'scaler.pkl')

# RL Backend Config
RL_API_BASE = "http://127.0.0.1:8000"

# --- Custom Keras Layer ---
class ExpandDimsLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.expand_dims(inputs, axis=-1)

# --- Load Model & Scaler ---
@st.cache_resource
def load_eeg_model():
    try:
        return tf.keras.models.load_model(MODEL_PATH, custom_objects={'ExpandDimsLayer': ExpandDimsLayer}, compile=False)
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

@st.cache_resource
def load_eeg_scaler():
    try:
        return pickle.load(open(SCALER_PATH, 'rb'))
    except Exception as e:
        st.error(f"❌ Error loading scaler: {str(e)}")
        return None

# --- Label Mapping ---
EEG_LABEL_MAP = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
REWARD_MAP = {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}

# --- Streamlit Setup ---
st.set_page_config(page_title="EEG-RL Pipeline", layout="wide")
st.title("🧠 EEG-Driven Video Recommender 🎬")

model = load_eeg_model()
scaler = load_eeg_scaler()

# --- Session State ---
if 'app_state' not in st.session_state:
    st.session_state.app_state = "STOPPED" 
    st.session_state.serial_port = None
    st.session_state.raw_data_buffer = []
    st.session_state.session_id = f"session_{int(time.time())}" 
    # Video info
    st.session_state.current_video_id = None
    st.session_state.current_video_title = None
    st.session_state.current_video_filename = None
    # Stats
    st.session_state.last_emotion = None
    st.session_state.last_update_time = time.time()
    st.session_state.status_message = "Ready. Press Start."

# --- UI Placeholders ---
col1, col2 = st.columns([1, 1.5])

with col1:
    st.button("Start 🚀", key="start_button", on_click=lambda: set_state("CONNECTING_SERIAL"),
              disabled=st.session_state.app_state != "STOPPED" or model is None)
    
with col2:
    st.button("Stop 🛑", key="stop_button", on_click=lambda: set_state("STOPPED"),
              disabled=st.session_state.app_state == "STOPPED")

status_placeholder = st.empty()
st.divider()

# Main Dashboard Layout
ui_col1, ui_col2 = st.columns(2)
with ui_col1:
    video_placeholder = st.empty()
    eeg_placeholder = st.empty()
    emotion_placeholder = st.empty()

with ui_col2:
    stats_placeholder = st.empty()

st.divider()

# Advanced Stats Expander
with st.expander("Show Advanced RL Model Stats (Q-Table)"):
    st.write("This shows the 'brain' of the RL model. Higher values mean the model 'thinks' recommending that video is better.")
    q_table_placeholder = st.empty()
    counts_table_placeholder = st.empty()

    if st.button("Refresh Q-Tables", key="refresh_q_button"): # Added key here
        try:
            res = requests.get(f"{RL_API_BASE}/q_table")
            if res.status_code == 200:
                data = res.json()
                st.write("--- 📈 Q-Table (What the model 'thinks') ---")
                q_df = pd.DataFrame(data['q_table'])
                q_table_placeholder.dataframe(q_df.style.background_gradient(cmap='viridis').format("{:.4f}"))
                st.write("--- 📊 Action Counts (What the model 'did') ---")
                c_df = pd.DataFrame(data['counts_table'])
                counts_table_placeholder.dataframe(c_df.style.background_gradient(cmap='plasma').format("{:n}"))
            else:
                st.error(f"Error fetching Q-Table: {res.text}")
        except Exception as e:
            st.error(f"Failed to connect to /q_table: {e}")

# --- State Management Function ---
def set_state(new_state):
    st.session_state.app_state = new_state
    st.session_state.last_update_time = time.time() 

    if new_state == "STOPPED":
        if st.session_state.serial_port and st.session_state.serial_port.is_open:
            st.session_state.serial_port.close()
            st.session_state.serial_port = None
        st.session_state.status_message = "Stopped. Ready to start."
        st.session_state.raw_data_buffer = []
        video_placeholder.empty()
        eeg_placeholder.empty()
        emotion_placeholder.empty()
    
    elif new_state == "CONNECTING_SERIAL":
        st.session_state.status_message = "Attempting to connect to EEG..."
    
    elif new_state == "COLLECTING_EEG":
        st.session_state.raw_data_buffer = [] 
        st.session_state.status_message = "Collecting EEG data..."
        video_placeholder.empty()
        eeg_placeholder.empty()
    
    elif new_state == "PAUSE_BEFORE_NEXT":
        st.session_state.status_message = f"Emotion '{st.session_state.last_emotion}' sent as feedback. Next video in 2s..."

# --- Function to Fetch and Display Simple Stats ---
def display_live_stats():
    try:
        res = requests.get(f"{RL_API_BASE}/stats")
        if res.status_code == 200:
            stats_data = res.json()
            df = pd.DataFrame.from_dict(stats_data, orient='index', columns=['Value'])
            with stats_placeholder.container():
                st.subheader("📈 Live RL Stats")
                st.dataframe(df, use_container_width=True)
        else:
            stats_placeholder.error(f"Error fetching stats: {res.text}")
    except Exception as e:
        stats_placeholder.warning(f"Stats server offline.")


# --- Main App Logic (No While Loop) ---

# Update status text at the top of every run
status_placeholder.info(f"Status: {st.session_state.status_message}")

# We only run logic if the app is not in the "STOPPED" state.
if st.session_state.app_state == "STOPPED":
    display_live_stats() # Show stats even when stopped
    st.stop() # Stop the script run here

# State: Connecting to Serial
elif st.session_state.app_state == "CONNECTING_SERIAL":
    try:
        st.session_state.serial_port = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
        time.sleep(2) 
        set_state("GETTING_RECOMMENDATION")
        st.rerun() # Re-run script to move to next state
    except serial.SerialException as e:
        st.session_state.status_message = f"Connection Error: {e}. Please check COM port. Stopping."
        set_state("STOPPED")
        st.rerun() # Re-run script to show the stopped state

# State: Getting new video from RL backend
elif st.session_state.app_state == "GETTING_RECOMMENDATION":
    st.session_state.status_message = "Fetching video recommendation..."
    display_live_stats() 
    video_placeholder.empty()
    eeg_placeholder.empty()
    emotion_placeholder.empty()
    
    try:
        res = requests.get(f"{RL_API_BASE}/recommend", params={"session_id": st.session_state.session_id})
        if res.status_code == 200:
            video = res.json()
            st.session_state.current_video_id = video['video_id']
            st.session_state.current_video_title = video['title']
            st.session_state.current_video_filename = video['filename']
            set_state("SHOWING_VIDEO")
            st.rerun() # Re-run script to show the video
        else:
            st.session_state.status_message = f"Error from RL server: {res.text}. Stopping."
            set_state("STOPPED")
            st.rerun()
    except requests.ConnectionError:
        st.session_state.status_message = f"Cannot connect to RL server at {RL_API_BASE}. Is it running? Stopping."
        set_state("STOPPED")
        st.rerun()
    except Exception as e:
        st.session_state.status_message = f"Error: {e}. Stopping."
        set_state("STOPPED")
        st.rerun()

# State: Show video and wait for user button press
elif st.session_state.app_state == "SHOWING_VIDEO":
    st.session_state.status_message = f"Now playing: {st.session_state.current_video_title}"
    
    video_file_path = os.path.join(VIDEOS_DIR, st.session_state.current_video_filename)
    if os.path.exists(video_file_path):
        video_placeholder.video(video_file_path)
    else:
        video_placeholder.error(f"Video file not found: {video_file_path}")
        set_state("STOPPED")
        st.rerun()
    
    # Show the button. When clicked, on_click changes state, which triggers a re-run.
    # No st.rerun() is needed here; we wait for the user.
    eeg_placeholder.button(
        "I'm done watching. Start EEG.",
        on_click=lambda: set_state("COLLECTING_EEG"),
        key="eeg_button" # This key is now unique for this script run
    )

# State: Poll serial port until buffer is full
# State: Poll serial port until buffer is full
# State: Poll serial port until buffer is full
elif st.session_state.app_state == "COLLECTING_EEG":
    # This state will now run as a blocking operation
    # It will not re-run until all data is collected
    try:
        eeg_placeholder.write(f"Collecting EEG data... (0 / {RAW_DATA_WINDOW_SIZE_FOR_FE} samples)")
        
        # --- Use a fast, blocking while loop ---
        while len(st.session_state.raw_data_buffer) < RAW_DATA_WINDOW_SIZE_FOR_FE:
            
            # Allow user to stop mid-collection
            if st.session_state.app_state != "COLLECTING_EEG":
                st.rerun() # User must have clicked stop
                st.stop() # Stop this script run

            # Try to read from the serial port
            if st.session_state.serial_port.in_waiting > 0:
                try:
                    # Inner try-except to skip bad lines
                    line = st.session_state.serial_port.readline().decode('utf-8').strip()
                    if line: 
                        af7_value, tp9_value = map(float, line.split(","))
                        timestamp = datetime.now().timestamp()
                        st.session_state.raw_data_buffer.append([timestamp, af7_value, tp9_value])
                        
                        # --- Update UI periodically ---
                        if len(st.session_state.raw_data_buffer) % 50 == 0: # Update every 50 samples
                             eeg_placeholder.write(f"Collecting EEG data... ({len(st.session_state.raw_data_buffer)} / {RAW_DATA_WINDOW_SIZE_FOR_FE} samples)")

                except (ValueError, IndexError) as e:
                    print(f"WARNING: Skipped malformed serial line. Data: '{line}', Error: {e}")
                    pass # Ignore the bad line and continue
            else:
                # If buffer is empty, sleep for a tiny bit to avoid busy-waiting
                time.sleep(0.001) 
        
        # --- Collection is Done ---
        eeg_placeholder.write(f"Collection complete! ({len(st.session_state.raw_data_buffer)} / {RAW_DATA_WINDOW_SIZE_FOR_FE} samples)")
        set_state("PREDICTING_EMOTION")
        st.rerun() # Move to next state
        st.stop() # Stop this script run
        
    except Exception as e: 
        st.session_state.status_message = f"Critical serial port error: {e}. Stopping."
        set_state("STOPPED")
        st.rerun()
# State: Run the ML model
elif st.session_state.app_state == "PREDICTING_EMOTION":
    st.session_state.status_message = "Analyzing emotion..."
    eeg_placeholder.write("Analyzing...")
    
    with open(CSV_FILENAME, 'w', newline='') as f:
        csv.writer(f).writerows([['Timestamp', 'AF7 Value', 'TP9 Value']] + st.session_state.raw_data_buffer[-RAW_DATA_WINDOW_SIZE_FOR_FE:])
    
    features, _ = generate_feature_vectors_from_samples(
        file_path=CSV_FILENAME, nsamples=150, period=1.0, state=None,
        remove_redundant=False, cols_to_ignore=[0]
    )
    
    if features is not None and features.size > 0:
        features = features[-1:] if features.ndim == 2 else features.reshape(1, -1)
        
        if features.shape[1] == 486:
            X_scaled = scaler.transform(features)
            predictions = model.predict(X_scaled, verbose=0)
            predicted_class = np.argmax(predictions, axis=1)[0]
            predicted_label = EEG_LABEL_MAP[predicted_class]
            st.session_state.last_emotion = predicted_label
            emotion_placeholder.success(f"Predicted Emotion: **{predicted_label}**")
            set_state("SENDING_FEEDBACK")
        else:
            st.session_state.status_message = f"Feature count mismatch: expected 486, got {features.shape[1]}. Stopping."
            set_state("STOPPED")
    else:
        st.session_state.status_message = "Failed to extract features from EEG data. Stopping."
        set_state("STOPPED")
    
    st.rerun() # Re-run to move to next state

# State: Send feedback to RL backend
elif st.session_state.app_state == "SENDING_FEEDBACK":
    st.session_state.status_message = "Sending feedback to RL model..."
    
    reward = REWARD_MAP[st.session_state.last_emotion]
    feedback_data = {
        "session_id": st.session_state.session_id,
        "video_id": st.session_state.current_video_id,
        "feedback": reward
    }
    
    try:
        res = requests.post(f"{RL_API_BASE}/feedback", json=feedback_data)
        if res.status_code == 200:
            set_state("PAUSE_BEFORE_NEXT")
        else:
            st.session_state.status_message = f"Error sending feedback: {res.text}. Stopping."
            set_state("STOPPED")
    except Exception as e:
        st.session_state.status_message = f"Feedback request failed: {e}. Stopping."
        set_state("STOPPED")
    
    st.rerun() # Re-run to move to next state

# State: Wait 2 seconds before next loop
elif st.session_state.app_state == "PAUSE_BEFORE_NEXT":
    current_time = time.time()
    time_elapsed = current_time - st.session_state.last_update_time
    
    if time_elapsed > 2.0:
        set_state("GETTING_RECOMMENDATION") 
    else:
        eeg_placeholder.write("Next recommendation in: {:.1f}s".format(2.0 - time_elapsed))
    
    st.rerun() # Re-run to check timer again