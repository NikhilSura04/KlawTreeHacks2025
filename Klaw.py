# Nikhil Surapaneni
# Rishi Bengani
# TreeHacks 2025


import pygame
import cv2
import mediapipe as mp
import time
import numpy as np
import pickle
import argparse
import serial
import pyautogui
import time
import speech_recognition as sr
import threading

# User defined constants
WIDTH, HEIGHT = 640, 480
RECORD_TIME = 10

data = []

# Define the function to calculate the angle between three points
def calculate_angle(point1, point2, point3):
    """
    Calculate the angle between three points
    """
    point1 = np.array(point1)
    point2 = np.array(point2)  # This is the joint point
    point3 = np.array(point3)
    
    vector1 = point1 - point2
    vector2 = point3 - point2
    
    cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Ensure the cosine value is within the valid range
    
    angle = np.arccos(cosine_angle)
    angle = np.degrees(angle)  # Convert to degrees
    
    return angle

parser = argparse.ArgumentParser()
parser.add_argument("--gesture", help="Gesture to be recorded", type=str, default=None)
parser.add_argument("--data_dir", help="Directory to save data to", type=str, default=None)
args = parser.parse_args()

# Establish serial connection if a port is specified
arduino_serial = None
if args.port:
    arduino_serial = serial.Serial(args.port, 230400)
    print("The port %s is available" % arduino_serial)

if args.knn:
    knn_file = open(args.knn, 'rb')     
    knn = pickle.load(knn_file)

# Mediapipe and Pygame initializations
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh  # Add face mesh module
hands = mp_hands.Hands(model_complexity=0, max_num_hands=1)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)  # Initialize face mesh

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 36)  # Font for displaying text

start = time.time()
gesture = ""
face_info = ""

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    recognizer = sr.Recognizer()
    subtitle_text = ""  # Store recognized text

    def recognize_speech():
        """Continuously listens for speech and updates subtitle_text."""
        global subtitle_text
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)  # Reduce background noise
            while True:
                try:
                    audio = recognizer.listen(source, phrase_time_limit=8)  # Listen for speech
                    subtitle_text = recognizer.recognize_google(audio)  # Convert speech to text
                except sr.UnknownValueError:
                    subtitle_text = "..."
                except sr.RequestError:
                    subtitle_text = "Speech recognition unavailable"

    # Run speech recognition in a separate thread to avoid blocking Pygame
    speech_thread = threading.Thread(target=recognize_speech, daemon=True)
    speech_thread.start()
    while cap.isOpened():
        clock.tick(60)
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Process the image and detect hands
        results_hands = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        results_face = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                # Calculate and display angle for a specific joint as an example
                try:
                    wrist = [hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x,
                             hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y,
                             hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z]

                    index_mcp = [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
                                 hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
                                 hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z]

                    index_pip = [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x,
                                 hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
                                 hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].z]

                    # Calculate the angle
                    angle = calculate_angle(wrist, index_mcp, index_pip)
                    gesture = f"Hand Angle: {angle:.2f}"  # Display the calculated angle
                except Exception as e:
                    print(f"Error calculating angle: {e}")
                
                # Draw hand landmarks on the original image
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()  # Get screen resolution

        CLICK_DELAY = 0.5  # Time in seconds to prevent multiple clicks (adjust as needed)
        last_click_time = 0  # Initialize last click timestamp
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                # Get index finger tip coordinates
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                # Flip X-coordinates for natural movement
                screen_x = SCREEN_WIDTH - (index_tip.x * SCREEN_WIDTH)
                screen_y = index_tip.y * SCREEN_HEIGHT  # Keep Y-axis the same

                # Move the mouse cursor smoothly
                pyautogui.moveTo(screen_x, screen_y, duration=0.1)

                # Calculate distance between index tip and thumb tip for clickin
                distance = np.linalg.norm(
                    np.array([index_tip.x, index_tip.y]) - np.array([thumb_tip.x, thumb_tip.y])
                )
                
                if index_tip.y < 0.4:  # Move hand up
                    pyautogui.scroll(10)  # Scroll up
                elif index_tip.y > 0.6:  # Move hand down
                    pyautogui.scroll(-10)  # Scroll down

                current_time = time.time()
                # If fingers are close enough, trigger a mouse click
                if distance < 0.05:  # Adjust threshold as needed
                    pyautogui.click()
                    last_click_time = current_time
        

        HEAD_TILT_THRESHOLD = 0.02

        # Face tracking
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                FACE_LANDMARKS = face_landmarks.landmark

                # smile deteection, may have to change threshold values
                left_mouth = FACE_LANDMARKS[61]  # Left corner of the mouth
                right_mouth = FACE_LANDMARKS[291]  # Right corner of the mouth
                top_lip = FACE_LANDMARKS[13]  # Upper lip
                bottom_lip = FACE_LANDMARKS[14]  # Lower lip
                left_ear = FACE_LANDMARKS[234]  # Left ear
                right_ear = FACE_LANDMARKS[454]  # Right ear

                tilt_value = left_ear.y - right_ear.y

                if tilt_value > HEAD_TILT_THRESHOLD:
                    pyautogui.press("volumeup")
                elif tilt_value < -(HEAD_TILT_THRESHOLD):
                    pyautogui.press("volumedown")

                # Calculate distances
                mouth_width = abs(right_mouth.x - left_mouth.x)
                mouth_height = abs(top_lip.y - bottom_lip.y)

                left_eyebrow = FACE_LANDMARKS[70]  # Left eyebrow middle
                right_eyebrow = FACE_LANDMARKS[300]  # Right eyebrow middle
                left_eye_top = FACE_LANDMARKS[159]  # Top of left eye
                right_eye_top = FACE_LANDMARKS[386]  # Top of right eye

                # Calculate distances between eyebrows and eyes
                left_brow_eye_distance = abs(left_eyebrow.y - left_eye_top.y)
                right_brow_eye_distance = abs(right_eyebrow.y - right_eye_top.y)

                # If the eyebrows are raised significantly, trigger the effect
                if left_brow_eye_distance > 0.05 and right_brow_eye_distance > 0.05:  # Adjust threshold
                    face_info = "Eyebrow Raise"
                
                drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

                # Draw only face contours
                mp_drawing.draw_landmarks(
                    image,
                    face_landmarks,
                    mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    drawing_spec,  # Apply custom drawing spec
                    drawing_spec
                )

        # Convert the image from BGR to RGB, rotate and flip for correct orientation
        image = cv2.cvtColor(np.rot90(image), cv2.COLOR_BGR2RGB)
        image = pygame.surfarray.make_surface(image)
        image = pygame.transform.flip(image, True, False)
        image = pygame.transform.scale(image, (WIDTH, HEIGHT))
        
        # Render the image and the gesture text on the screen
        screen.blit(image, (0, 0))
        gesture_text = font.render(gesture, True, (255, 255, 255))
        face_text = font.render(face_info, True, (255, 255, 255))
        screen.blit(gesture_text, (10, 10))
        screen.blit(face_text, (10, 50))

        # Render subtitles at the bottom of the screen
        subtitle_render = font.render(subtitle_text, True, (255, 255, 255))
        screen.blit(subtitle_render, (10, HEIGHT - 50))  # Position subtitles at bottom

        
        pygame.display.update()

        # Handle quitting the loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                quit()

    cap.release()