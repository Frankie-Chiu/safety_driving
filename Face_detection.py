# url = "https://translate.google.com/translate_tts?ie=UTF-8&tl=en&client=tw-ob&q=Please keep your eye on the road"

import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import requests
from pydub import AudioSegment
from pydub.playback import play
import io
import datetime
import threading
import io
import time

last_warning_time = 0
last_sleepy_time = 0
last_not_centered_time = 0

last_eye_closed_time = 0
last_sleepy_time = 0  # Note: This variable is already declared, so just included here for context
last_not_in_front_time = 0
MESSAGE_DISPLAY_TIME = 1  

EYE_AR_THRESH = 0.23
EYE_AR_CONSEC_SECONDS = 0.5
FACE_CONSEC_SECONDS = 1.0
frame_rate = 30  # Update this to match the actual frame rate of your webcam
EYE_AR_CONSEC_FRAMES = EYE_AR_CONSEC_SECONDS * frame_rate

#Face_dection_config
last_face_detected_time = time.time()
last_face_not_detected_time = time.time()
FACE_DETECTED_MESSAGE_DURATION = 2 
FACE_NOT_DETECTED_WARNING_DURATION = 8
NOT_ATTENTION_WARNING_DURATION = 1

EYES_CLOSED_WARNING_FILE_NAME = 'wakeup.mp3'
FACE_NOT_DETECTED_WARNING_FILE_NAME = "face_not_detected.mp3"
YOU_ARE_SLEEPY_WARNING_FILE_NAME = 'sleepy.mp3'
ROAD_ATTENTION_WARNING_FILE_NAME = 'road_attention.mp3'

def play_sound(file_name, waiting_time):
    global last_warning_time
    current_time = time.time()

    def run():
        file_path = file_name  # Update this path as necessary
        sound = AudioSegment.from_file(file_path, format="mp3")
        play(sound)

    if current_time - last_warning_time >= waiting_time:
        last_warning_time = current_time
        threading.Thread(target=run).start()

def play_eyes_close_warning():
    play_sound(EYES_CLOSED_WARNING_FILE_NAME, 2)

def play_sleepy_warning():
    play_sound(YOU_ARE_SLEEPY_WARNING_FILE_NAME, 2)

def play_road_attention_warning():
    play_sound(ROAD_ATTENTION_WARNING_FILE_NAME, 4)

def play_face_not_detected_warning():
    play_sound(FACE_NOT_DETECTED_WARNING_FILE_NAME, 3)


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])  
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6]) 
    mar = (A + B) / (2.0 * C)
    return mar

def visualize_facial_features(frame, shape_np):
    facial_features_indexes = {
        'left_eye': range(42, 48),
        'right_eye': range(36, 42),
        'mouth': range(48, 68)  # Adjusted to include the full mouth range
    }
    
    colors = {
        'left_eye': (0, 0, 255),  # Red for eyes
        'right_eye': (0, 0, 255),
        'mouth': (255, 0, 0)  # Blue for mouth
    }
    
    line_thickness = 2

    ear = 0
    for feature, indexes in facial_features_indexes.items():
        points = shape_np[list(indexes)]
        hull = cv2.convexHull(points)
        color = colors[feature]
        cv2.drawContours(frame, [hull], -1, color, line_thickness)
        
        if feature in ['left_eye', 'right_eye']:
            ear += eye_aspect_ratio(points)
        elif feature == 'mouth':
            mar = mouth_aspect_ratio(points)

    ear /= 2 

    # Display EAR and MAR on the frame
    cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def estimate_yaw_angle(shape_np):
    nose_tip = shape_np[33]
    chin = shape_np[8]

    left_side = shape_np[0]
    right_side = shape_np[16]

    nose_to_left_vector = nose_tip - left_side
    nose_to_right_vector = nose_tip - right_side

    dist_left = np.linalg.norm(nose_to_left_vector)
    dist_right = np.linalg.norm(nose_to_right_vector)

    yaw = (dist_left - dist_right) / (dist_left + dist_right) 
    return yaw

def visualize_yaw_angle(frame, shape_np):
    yaw = estimate_yaw_angle(shape_np)
    cv2.putText(frame, f"Yaw: {yaw:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


def check_eyes_closed(frame, ear, counter, consec_frames):
    global last_warning_time, last_eye_closed_time
    current_time = time.time()
    if current_time - last_not_centered_time >  EYE_AR_THRESH:
        counter += 1
        if counter >= consec_frames:
            last_eye_closed_time = current_time
            play_eyes_close_warning()
    else:
        counter = 0
    
    if current_time - last_eye_closed_time < MESSAGE_DISPLAY_TIME:
        cv2.putText(frame, "Eye Closed", (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return counter

def check_mouth_open(frame, mar):
    global last_sleepy_time
    current_time = time.time()
    if mar > 1:
        last_sleepy_time = current_time
        play_sleepy_warning()

    if current_time - last_sleepy_time < MESSAGE_DISPLAY_TIME:
        cv2.putText(frame, "You are sleepy", (frame.shape[1] - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def check_face_not_centered(vertical_direction, horizontal_direction, frame):
    global last_not_centered_time, last_not_in_front_time
    current_time = time.time()

    if vertical_direction != "Center" or horizontal_direction != "Center":
        if last_not_centered_time == 0: 
            last_not_centered_time = current_time
        else:
            if (current_time - last_not_centered_time > FACE_CONSEC_SECONDS) and (last_not_in_front_time == 0 or current_time - last_not_in_front_time > MESSAGE_DISPLAY_TIME):
                last_not_in_front_time = current_time
    else:
        last_not_centered_time = 0
        last_not_in_front_time = 0

    if 0 < current_time - last_not_in_front_time <= MESSAGE_DISPLAY_TIME:
        cv2.putText(frame, "Not in front", (frame.shape[1] - 200, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

def handle_face_detection_messages(face_detected, last_face_detected_time, last_face_not_detected_time, continuous_face_detected, frame):
    current_time = time.time()
    message_displayed = False  
    
    if face_detected:
        if last_face_detected_time == 0:
            last_face_detected_time = current_time
        elif current_time - last_face_detected_time > FACE_DETECTED_MESSAGE_DURATION and not continuous_face_detected:
            continuous_face_detected = True
        last_face_not_detected_time = 0
    else:
        if continuous_face_detected and last_face_not_detected_time == 0:
            last_face_not_detected_time = current_time
        elif continuous_face_detected and current_time - last_face_not_detected_time > NOT_ATTENTION_WARNING_DURATION:
            cv2.putText(frame, "Please turn your head to the front.", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            play_road_attention_warning()
            message_displayed = True
            continuous_face_detected = False  
            last_face_not_detected_time = current_time
        if not continuous_face_detected:
            last_face_detected_time = 0  
    
    if not face_detected and current_time - last_face_not_detected_time > FACE_NOT_DETECTED_WARNING_DURATION:
        cv2.putText(frame, "Warning: Face cannot be detected.", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        play_face_not_detected_warning()
        last_face_not_detected_time = current_time
        message_displayed = True

    return last_face_detected_time, last_face_not_detected_time, continuous_face_detected, message_displayed


def detect_face_direction(shape_np):
    # Calculate the center of the eyes
    left_eye_center = np.mean(shape_np[36:42], axis=0)
    right_eye_center = np.mean(shape_np[42:48], axis=0)
    eye_center = (left_eye_center + right_eye_center) / 2

    # Inter-eye distance
    inter_eye_distance = np.linalg.norm(right_eye_center - left_eye_center)

    # Nose tip for reference
    nose_tip = shape_np[33]

    # Normalize vertical and horizontal movements by inter-eye distance
    vertical_movement = (nose_tip[1] - eye_center[1]) / inter_eye_distance
    horizontal_movement = (nose_tip[0] - eye_center[0]) / inter_eye_distance

    # Calculate the eye-line angle
    dx = right_eye_center[0] - left_eye_center[0]
    dy = right_eye_center[1] - left_eye_center[1]
    eye_line_angle = np.degrees(np.arctan2(dy, dx))

    # Determine directions based on normalized movements
    if vertical_movement > 0.8:  
        vertical_direction = "Down"
    elif vertical_movement < 0.65:  
        vertical_direction = "Up"
    else:
        vertical_direction = "Center"

    if horizontal_movement > 0.15:  
        horizontal_direction = "Left"
    elif horizontal_movement < -0.15: 
        horizontal_direction = "Right"
    else:
        horizontal_direction = "Center"

    if eye_line_angle > 5:  
        roll_direction = "Tilt Right"
    elif eye_line_angle < 0: 
        roll_direction = "Tilt Left"
    else:
        roll_direction = "Straight"

    cv2.putText(frame, f"vertical_movement: {vertical_movement:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"horizontal_movement: {horizontal_movement:.2f}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"eye_line_angle: {eye_line_angle:.2f}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return vertical_direction, horizontal_direction, roll_direction, vertical_movement


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


COUNTER = 0
continuous_face_detected = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    face_detected = len(rects) > 0

    last_face_detected_time, last_face_not_detected_time, continuous_face_detected, message_displayed = handle_face_detection_messages(
        face_detected, last_face_detected_time, last_face_not_detected_time, continuous_face_detected, frame
    )

    for rect in rects:
        shape = predictor(gray, rect)
        shape_np = np.array([(p.x, p.y) for p in shape.parts()], dtype="int")

        leftEye = shape_np[42:48]
        rightEye = shape_np[36:42]
        mouth = shape_np[48:60]
        vertical_direction, horizontal_direction, roll_direction, vertical_movement = detect_face_direction(shape_np)

        cv2.putText(frame, f"Face Detected: {len(rects)}", (10 - 200, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        if vertical_direction == "Up":
            ear = ear + 0.05
        elif vertical_direction == "Down":
            ear = ear - 0.05
        mar = mouth_aspect_ratio(mouth)

        COUNTER = check_eyes_closed(frame, ear, COUNTER, EYE_AR_CONSEC_FRAMES)
        check_mouth_open(frame, mar)

        visualize_facial_features(frame, shape_np)
        visualize_yaw_angle(frame, shape_np)

        check_face_not_centered(vertical_direction, horizontal_direction, frame)

        cv2.putText(frame, f"Vertical: {vertical_direction}", (frame.shape[1] - 200, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Horizontal: {horizontal_direction}", (frame.shape[1] - 200, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Roll: {roll_direction}", (frame.shape[1] - 200, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

#add