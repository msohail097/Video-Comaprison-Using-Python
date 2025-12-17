# # ===============================
# # Google Colab - Pole Dance Comparison (Technique-Based)
# # ===============================

# import cv2
# import mediapipe as mp
# import numpy as np
# import matplotlib.pyplot as plt

# mp_pose = mp.solutions.pose

# # -------------------------------
# # Extract pose keypoints
# # -------------------------------
# def extract_keypoints(video_path):
#     cap = cv2.VideoCapture(video_path)
#     keypoints_list, frames_list = [], []

#     with mp_pose.Pose(
#         static_image_mode=False,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5
#     ) as pose:
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             frames_list.append(frame)
#             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             res = pose.process(rgb)

#             if res.pose_landmarks:
#                 kp = []
#                 for lm in res.pose_landmarks.landmark:
#                     kp.extend([lm.x, lm.y])
#                 keypoints_list.append(np.array(kp))
#             else:
#                 keypoints_list.append(np.zeros(66))

#     cap.release()
#     return keypoints_list, frames_list

# # -------------------------------
# # Detect active (pole-performance) frames
# # -------------------------------
# def get_active_frames(keypoints, threshold=0.015):
#     active = []
#     for i in range(1, len(keypoints)):
#         if np.linalg.norm(keypoints[i] - keypoints[i-1]) > threshold:
#             active.append(i)
#     return active

# # -------------------------------
# # Technique metrics (RELATIVE)
# # -------------------------------
# def hand_height_ratio(kp, side="right"):
#     kp = kp.reshape(-1, 2)
#     hand = kp[16] if side == "right" else kp[15]
#     shoulder = kp[12] if side == "right" else kp[11]
#     hip = kp[24] if side == "right" else kp[23]
#     return abs(hand[1] - shoulder[1]) / (abs(hip[1] - shoulder[1]) + 1e-6)

# def elbow_straightness(kp, side="right"):
#     kp = kp.reshape(-1, 2)
#     shoulder = kp[12] if side == "right" else kp[11]
#     elbow = kp[14] if side == "right" else kp[13]
#     wrist = kp[16] if side == "right" else kp[15]
#     upper = np.linalg.norm(shoulder - elbow)
#     lower = np.linalg.norm(elbow - wrist)
#     return lower / (upper + 1e-6)

# def bottom_leg_grip(kp, side="right"):
#     kp = kp.reshape(-1, 2)
#     hip = kp[24] if side == "right" else kp[23]
#     knee = kp[26] if side == "right" else kp[25]
#     ankle = kp[28] if side == "right" else kp[27]
#     thigh = np.linalg.norm(hip - knee)
#     calf = np.linalg.norm(knee - ankle)
#     return thigh / (calf + 1e-6)

# # -------------------------------
# # Compare technique (20% rule)
# # -------------------------------
# def compare_technique(trainer_kps, trainee_kps, diff_threshold=0.2):
#     active_frames = get_active_frames(trainee_kps)
#     feedback = set()

#     for i in active_frames:
#         if i >= len(trainer_kps):
#             continue

#         t = trainer_kps[i]
#         s = trainee_kps[i]

#         # Hand height
#         if abs(hand_height_ratio(s) - hand_height_ratio(t)) / hand_height_ratio(t) > diff_threshold:
#             feedback.add("Your hand is placed higher/lower on the pole compared to the trainer.")

#         # Arm bend
#         if abs(elbow_straightness(s) - elbow_straightness(t)) / elbow_straightness(t) > diff_threshold:
#             feedback.add("You bend your arm during the spin while the trainer keeps it straighter.")

#         # Bottom leg grip
#         if abs(bottom_leg_grip(s) - bottom_leg_grip(t)) / bottom_leg_grip(t) > diff_threshold:
#             feedback.add("Your bottom leg grip differs — trainer engages the thigh more.")

#     similarity_score = max(0, 1 - len(feedback) * 0.15)

#     if not feedback:
#         feedback.add("No major technical issues detected. Overall technique is similar.")

#     return similarity_score, list(feedback)

# # -------------------------------
# # Visualize skeleton
# # -------------------------------
# def visualize_skeleton(frame, keypoints):
#     kp = keypoints.reshape(-1, 2)
#     for x, y in kp:
#         cv2.circle(frame, (int(x*frame.shape[1]), int(y*frame.shape[0])), 4, (0,255,0), -1)
#     plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     plt.axis("off")
#     plt.show()

# # -------------------------------
# # MAIN
# # -------------------------------
# trainer_video = "trainer.mp4"
# trainee_video = "trainee.mp4"

# print("Extracting trainer pose...")
# trainer_kps, trainer_frames = extract_keypoints(trainer_video)

# print("Extracting trainee pose...")
# trainee_kps, trainee_frames = extract_keypoints(trainee_video)

# print("Analyzing technique...")
# score, feedback = compare_technique(trainer_kps, trainee_kps)

# print(f"\nSimilarity Score: {score:.2f}")
# print("---- FEEDBACK ----")
# for f in feedback:
#     print("•", f)

# print("\nVisualizing sample frame...")
# visualize_skeleton(trainee_frames[len(trainee_frames)//2], trainee_kps[len(trainee_kps)//2])




#
#  with voice 


# =========================================================
# POLE DANCE AI COACH — DYNAMIC VIDEO-BASED FEEDBACK + VOICE
# =========================================================

# import cv2
# import mediapipe as mp
# import numpy as np
# from collections import defaultdict
# from gtts import gTTS
# import os

# mp_pose = mp.solutions.pose

# # ---------------------------------------------------------
# # 1. Extract pose keypoints
# # ---------------------------------------------------------
# def extract_keypoints(video_path):
#     cap = cv2.VideoCapture(video_path)
#     keypoints_list = []

#     with mp_pose.Pose(
#         static_image_mode=False,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5
#     ) as pose:

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             res = pose.process(rgb)

#             if res.pose_landmarks:
#                 kp = []
#                 for lm in res.pose_landmarks.landmark:
#                     kp.extend([lm.x, lm.y])
#                 keypoints_list.append(np.array(kp))
#             else:
#                 keypoints_list.append(np.zeros(66))

#     cap.release()
#     return keypoints_list

# # ---------------------------------------------------------
# # 2. Technique metrics (relative & scale-invariant)
# # ---------------------------------------------------------
# def hand_height_ratio(kp):
#     kp = kp.reshape(-1, 2)
#     hand, shoulder, hip = kp[16], kp[12], kp[24]
#     return abs(hand[1] - shoulder[1]) / (abs(hip[1] - shoulder[1]) + 1e-6)

# def elbow_straightness(kp):
#     kp = kp.reshape(-1, 2)
#     shoulder, elbow, wrist = kp[12], kp[14], kp[16]
#     return np.linalg.norm(elbow - wrist) / (np.linalg.norm(shoulder - elbow) + 1e-6)

# def bottom_leg_grip(kp):
#     kp = kp.reshape(-1, 2)
#     hip, knee, ankle = kp[24], kp[26], kp[28]
#     return np.linalg.norm(hip - knee) / (np.linalg.norm(knee - ankle) + 1e-6)

# # ---------------------------------------------------------
# # 3. Detect frame-level mistake events
# # ---------------------------------------------------------
# def detect_mistake_events(trainer_kps, trainee_kps, threshold=0.2):
#     events = []

#     for i in range(min(len(trainer_kps), len(trainee_kps))):
#         t, s = trainer_kps[i], trainee_kps[i]

#         diffs = {
#             "hand placement": abs(hand_height_ratio(s) - hand_height_ratio(t)),
#             "arm bend": abs(elbow_straightness(s) - elbow_straightness(t)),
#             "leg grip": abs(bottom_leg_grip(s) - bottom_leg_grip(t))
#         }

#         for k, v in diffs.items():
#             if v > threshold:
#                 events.append({
#                     "frame": i,
#                     "type": k,
#                     "severity": v
#                 })

#     return events

# # ---------------------------------------------------------
# # 4. Analyze behavioral patterns
# # ---------------------------------------------------------
# def analyze_patterns(events, total_frames):
#     summary = defaultdict(list)

#     for e in events:
#         summary[e["type"]].append(e["severity"])

#     patterns = {}
#     for k, v in summary.items():
#         patterns[k] = {
#             "count": len(v),
#             "frequency": len(v) / total_frames,
#             "severity": np.mean(v)
#         }

#     return patterns

# # ---------------------------------------------------------
# # 5. Generate dynamic coaching feedback
# # ---------------------------------------------------------
# def generate_feedback(patterns):
#     feedback = []

#     for move, p in patterns.items():
#         freq = p["frequency"]
#         sev = p["severity"]

#         timing = (
#             "throughout most of the routine" if freq > 0.4 else
#             "during several transitions" if freq > 0.2 else
#             "occasionally"
#         )

#         intensity = (
#             "significantly" if sev > 0.6 else
#             "moderately" if sev > 0.35 else
#             "slightly"
#         )

#         feedback.append(
#             f"Your {move} {timing} differs {intensity} from the trainer."
#         )

#     if not feedback:
#         feedback.append(
#             "Your movement patterns closely match the trainer with no consistent technical issues."
#         )

#     feedback.append(
#         "Focus on controlled repetitions and slow practice to correct these deviations."
#     )

#     return feedback

# # ---------------------------------------------------------
# # 6. Convert feedback to female voice & save file
# # ---------------------------------------------------------
# def generate_voice(feedback_lines, output_file="pole_coach_feedback.mp3"):
#     text = " ".join(feedback_lines)
#     tts = gTTS(text=text, lang="en", slow=False)
#     tts.save(output_file)
#     return output_file

# # ---------------------------------------------------------
# # 7. MAIN PIPELINE
# # ---------------------------------------------------------
# if __name__ == "__main__":

#     trainer_video = "trainer.mp4"
#     trainee_video = "trainee.mp4"

#     print("🔍 Extracting trainer poses...")
#     trainer_kps = extract_keypoints(trainer_video)

#     print("🔍 Extracting trainee poses...")
#     trainee_kps = extract_keypoints(trainee_video)

#     print("📊 Analyzing differences...")
#     events = detect_mistake_events(trainer_kps, trainee_kps)
#     patterns = analyze_patterns(events, len(trainee_kps))

#     print("🧠 Generating dynamic coaching feedback...")
#     feedback = generate_feedback(patterns)

#     print("\n--- COACH FEEDBACK ---")
#     for f in feedback:
#         print("•", f)

#     print("\n🔊 Generating voice feedback...")
#     audio_file = generate_voice(feedback)

#     print(f"\n✅ Voice feedback saved as: {os.path.abspath(audio_file)}")

# =========================================================
# POLE DANCE AI COACH — HUMAN FEEDBACK + VOICE (STREAMLIT)
# =========================================================
# =========================================================
# POLE DANCE AI COACH — HUMAN FEEDBACK + VOICE (STREAMLIT)
# =========================================================
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
import uuid
from gtts import gTTS
import random

# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------
mp_pose = mp.solutions.pose
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Pose extraction
# ---------------------------------------------------------
def extract_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    prog_container = st.empty()
    progress_bar = prog_container.progress(0)

    with mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks:
                kp = []
                for lm in res.pose_landmarks.landmark:
                    kp.extend([lm.x, lm.y])
                keypoints.append(np.array(kp))
            else:
                keypoints.append(np.zeros(66))
            progress_bar.progress((i+1)/total_frames)
    
    cap.release()
    prog_container.empty()
    return keypoints

# ---------------------------------------------------------
# Technique metrics
# ---------------------------------------------------------
def hand_height_ratio(kp):
    kp = kp.reshape(-1, 2)
    return abs(kp[16][1] - kp[12][1]) / (abs(kp[24][1] - kp[12][1]) + 1e-6)

def elbow_straightness(kp):
    kp = kp.reshape(-1, 2)
    return np.linalg.norm(kp[14] - kp[16]) / (np.linalg.norm(kp[12] - kp[14]) + 1e-6)

def bottom_leg_grip(kp):
    kp = kp.reshape(-1, 2)
    return np.linalg.norm(kp[24] - kp[26]) / (np.linalg.norm(kp[26] - kp[28]) + 1e-6)

# ---------------------------------------------------------
# Detect mistake segments
# ---------------------------------------------------------
def detect_mistake_segments(trainer_kps, trainee_kps, threshold=0.2):
    segments = []
    current = None
    for i in range(min(len(trainer_kps), len(trainee_kps))):
        t, s = trainer_kps[i], trainee_kps[i]
        diffs = {
            "arm bend": abs(elbow_straightness(s) - elbow_straightness(t)),
            "hand placement": abs(hand_height_ratio(s) - hand_height_ratio(t)),
            "leg grip": abs(bottom_leg_grip(s) - bottom_leg_grip(t)),
        }
        for move, diff in diffs.items():
            if diff > threshold:
                if current is None or current["move"] != move:
                    current = {"move": move, "start": i, "values": [diff]}
                else:
                    current["values"].append(diff)
            else:
                if current and current["move"] == move:
                    current["end"] = i
                    segments.append(current)
                    current = None
    return segments

def describe_timing(start, end, total):
    pos = ((start + end) / 2) / total
    if pos < 0.33: return "early in the movement"
    elif pos < 0.66: return "during the middle phase"
    else: return "towards the end of the movement"

FEEDBACK_TEMPLATES = {
    "arm bend": ["{timing}, your supporting arm starts to bend compared to the trainer."],
    "hand placement": ["{timing}, your hand placement shifts on the pole."],
    "leg grip": ["{timing}, your bottom leg grip becomes weaker."]
}

# ---------------------------------------------------------
# Generate paragraph feedback
# ---------------------------------------------------------
def generate_paragraph_feedback(segments, total_frames):
    if not segments:
        return "Your movement closely matches the trainer. Excellent control and precision throughout the routine."
    
    feedback_sentences = []
    seen = set()  # keep track of unique (move, timing) pairs
    
    for seg in segments:
        timing = describe_timing(seg["start"], seg.get("end", seg["start"]), total_frames)
        key = (seg["move"], timing)
        if key not in seen:
            sentence = random.choice(FEEDBACK_TEMPLATES[seg["move"]]).format(timing=timing)
            feedback_sentences.append(sentence)
            seen.add(key)
    
    # Combine sentences into a single paragraph
    paragraph = " ".join(feedback_sentences)
    return paragraph

def generate_voice(feedback_paragraph, audio_path):
    tts = gTTS(text=feedback_paragraph, lang="en", slow=False)
    tts.save(audio_path)
    return audio_path

# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.set_page_config(page_title="Pole Power AI Coach", layout="centered")
st.title("💃 Pole Power AI Coach")

# Initialize session state
state_keys = ["analysis_done", "audio_path", "trainer_kps", "trainee_kps", "feedback_paragraph", "job_id"]
for key in state_keys:
    if key not in st.session_state:
        st.session_state[key] = None if key != "analysis_done" else False

trainer_file = st.file_uploader("Upload Trainer Video", type=["mp4"])
trainee_file = st.file_uploader("Upload Trainee Video", type=["mp4"])

if st.button("Analyze") or st.session_state.analysis_done:
    if not trainer_file or not trainee_file:
        st.error("Please upload both videos.")
    else:
        st.session_state.analysis_done = True
        
        if st.session_state.job_id is None:
            st.session_state.job_id = str(uuid.uuid4())
        
        jid = st.session_state.job_id
        t_path, s_path = f"{UPLOAD_DIR}/{jid}_t.mp4", f"{UPLOAD_DIR}/{jid}_s.mp4"
        a_path = f"{OUTPUT_DIR}/{jid}_fb.mp3"

        # Save uploaded files
        if not os.path.exists(t_path):
            with open(t_path, "wb") as f: f.write(trainer_file.getbuffer())
            with open(s_path, "wb") as f: f.write(trainee_file.getbuffer())

        # Extract keypoints
        if st.session_state.trainer_kps is None:
            st.session_state.trainer_kps = extract_keypoints(t_path)
        if st.session_state.trainee_kps is None:
            st.session_state.trainee_kps = extract_keypoints(s_path)

        # Generate paragraph feedback
        if st.session_state.feedback_paragraph is None:
            segs = detect_mistake_segments(st.session_state.trainer_kps, st.session_state.trainee_kps)
            st.session_state.feedback_paragraph = generate_paragraph_feedback(segs, len(st.session_state.trainee_kps))

        # Generate voice feedback
        if st.session_state.audio_path is None:
            with st.spinner("Generating voice feedback..."):
                st.session_state.audio_path = generate_voice(st.session_state.feedback_paragraph, a_path)

        # Display results
        st.success("Analysis Complete!")
        st.subheader("🧠 Coaching Feedback (Paragraph)")
        st.write(st.session_state.feedback_paragraph)

        st.subheader("🔊 Voice Coach")
        st.audio(st.session_state.audio_path)

        if st.button("Start New Analysis"):
            for key in state_keys: st.session_state[key] = None if key != "analysis_done" else False
            st.rerun()
