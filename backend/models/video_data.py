# backend/models/video_data.py
import os
import glob

# --- Dynamic Video Loader ---

# 1. Define the path to the videos folder.
#    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'videos'))
#    - os.path.dirname(__file__)  -> /backend/models
#    - '..'                       -> /backend
#    - '..'                       -> /
#    - 'videos'                   -> /videos
VIDEOS_DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'videos'))

# 2. Find all .mp4 files in that directory
#    We sort them to ensure a consistent video_id (1, 2, 3...)
video_files = sorted(glob.glob(os.path.join(VIDEOS_DIR_PATH, "*.mp4")))

# 3. Programmatically build the VIDEOS list
VIDEOS = []
for i, file_path in enumerate(video_files):
    # Get the filename (e.g., "Funny Cats.mp4")
    filename = os.path.basename(file_path)
    # Get the title by removing ".mp4" (e.g., "Funny Cats")
    title = os.path.splitext(filename)[0]
    
    VIDEOS.append({
        "video_id": i + 1,  # Start IDs from 1
        "title": title,
        "filename": filename # e.g., "Funny Cats.mp4"
    })

# 4. Print a status message so we know it worked when the server starts
if not VIDEOS:
    print(f"--- WARNING ---")
    print(f"No .mp4 files found in {VIDEOS_DIR_PATH}")
    print(f"The application will not work until videos are added.")
    print(f"---------------")
else:
    print(f"--- Video Loader Initialized ---")
    print(f"Found {len(VIDEOS)} videos in {VIDEOS_DIR_PATH}:")
    for v in VIDEOS:
        print(f"  ID {v['video_id']}: {v['filename']}")
    print(f"--------------------------------")