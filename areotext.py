import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import time

# Initialize Mediapipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Load the trained model and label binarizer
model = load_model("model_save.h5")
with open("label_binarizer.pkl", "rb") as file:
    lb = pickle.load(file)

# Initialize OpenCV window and drawing canvases
cap = cv2.VideoCapture(0)
canvas = None
whiteboard = None
text_display = None
drawing = False
previous_position = None
current_stroke = []
strokes = []
last_prediction_time = 0
prediction_delay = 1.0
current_text = ""

# Button interaction tracking with timestamps
last_space_press_time = 0
last_undo_press_time = 0
button_cooldown = 0.3  # Cooldown period in seconds

# Colors and brush settings
pen_color = (0, 0, 255)  # Default pen color (red)
pen_thickness = 5

# Define button positions and dimensions
button_size = (50, 50)
clear_button_pos = (10, 10)
blue_button_pos = (70, 10)
green_button_pos = (130, 10)
red_button_pos = (190, 10)
undo_button_pos = (250, 10)

# Space bar dimensions and position at the top
space_bar_height = 40
space_bar_width = 180
space_bar_pos = (310, 10)

# Button state tracking
button_hover = {
    'clear': False,
    'blue': False,
    'green': False,
    'red': False,
    'undo': False,
    'space': False
}

def preprocess_stroke(stroke, target_size=(32, 32)):
    img = np.zeros((400, 400), dtype=np.uint8)
    
    for i in range(len(stroke) - 1):
        cv2.line(img, stroke[i], stroke[i + 1], 255, thickness=3)
    
    coords = np.column_stack(np.where(img > 0))
    if len(coords) == 0:
        return None
    
    min_y, min_x = coords.min(axis=0)
    max_y, max_x = coords.max(axis=0)
    
    padding = 10
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = min(img.shape[1], max_x + padding)
    max_y = min(img.shape[0], max_y + padding)
    
    cropped = img[min_y:max_y, min_x:max_x]
    if cropped.size == 0:
        return None
    
    resized = cv2.resize(cropped, target_size)
    return resized

def predict_text(stroke):
    processed_img = preprocess_stroke(stroke)
    if processed_img is None:
        return ""
    
    processed_img = processed_img.reshape(1, 32, 32, 1)
    processed_img = processed_img / 255.0
    
    prediction = model.predict(processed_img)
    predicted_char = lb.inverse_transform(prediction)[0]
    
    return predicted_char

def draw_laser_effect(frame, x, y, color=(0, 0, 255)):
    # Draw a glowing circle at fingertip
    cv2.circle(frame, (x, y), 5, color, -1)
    # Add glow effect with increasing radii and decreasing opacity
    for radius in range(6, 15):
        alpha = 0.2 - (radius - 6) * 0.02
        overlay = frame.copy()
        cv2.circle(overlay, (x, y), radius, color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def is_point_in_rect(point, rect_pos, rect_size):
    return (rect_pos[0] <= point[0] <= rect_pos[0] + rect_size[0] and 
            rect_pos[1] <= point[1] <= rect_pos[1] + rect_size[1])

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to capture frame from camera.")
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Initialize canvases if needed
    if canvas is None:
        canvas = np.zeros_like(frame)
        whiteboard = np.ones_like(frame) * 255
        text_display = np.ones((100, frame.shape[1], 3), dtype=np.uint8) * 255

    current_time = time.time()
    
    # Reset button hover states
    for key in button_hover:
        button_hover[key] = False

    # Get finger position if hand is detected
    fingertip_x, fingertip_y = None, None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            fingertip = hand_landmarks.landmark[8]  # Index fingertip
            fingertip_x = int(fingertip.x * frame.shape[1])
            fingertip_y = int(fingertip.y * frame.shape[0])
            
            # Update button hover states
            button_hover['clear'] = is_point_in_rect((fingertip_x, fingertip_y), clear_button_pos, button_size)
            button_hover['blue'] = is_point_in_rect((fingertip_x, fingertip_y), blue_button_pos, button_size)
            button_hover['green'] = is_point_in_rect((fingertip_x, fingertip_y), green_button_pos, button_size)
            button_hover['red'] = is_point_in_rect((fingertip_x, fingertip_y), red_button_pos, button_size)
            button_hover['undo'] = is_point_in_rect((fingertip_x, fingertip_y), undo_button_pos, button_size)
            button_hover['space'] = is_point_in_rect((fingertip_x, fingertip_y), space_bar_pos, (space_bar_width, space_bar_height))

    # Draw buttons with hover effect
    # Clear button
    button_color = (50, 50, 50) if button_hover['clear'] else (0, 0, 0)
    cv2.rectangle(frame, clear_button_pos, 
                 (clear_button_pos[0] + button_size[0], clear_button_pos[1] + button_size[1]), 
                 button_color, -1)
    cv2.putText(frame, 'Clear', (clear_button_pos[0] + 5, clear_button_pos[1] + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Color buttons
    button_blue = (255, 100, 100) if button_hover['blue'] else (255, 0, 0)
    cv2.rectangle(frame, blue_button_pos, 
                 (blue_button_pos[0] + button_size[0], blue_button_pos[1] + button_size[1]), 
                 button_blue, -1)
    
    button_green = (100, 255, 100) if button_hover['green'] else (0, 255, 0)
    cv2.rectangle(frame, green_button_pos, 
                 (green_button_pos[0] + button_size[0], green_button_pos[1] + button_size[1]), 
                 button_green, -1)
    
    button_red = (100, 100, 255) if button_hover['red'] else (0, 0, 255)
    cv2.rectangle(frame, red_button_pos, 
                 (red_button_pos[0] + button_size[0], red_button_pos[1] + button_size[1]), 
                 button_red, -1)
    
    # Undo button
    button_undo = (180, 180, 180) if button_hover['undo'] else (150, 150, 150)
    cv2.rectangle(frame, undo_button_pos, 
                 (undo_button_pos[0] + button_size[0], undo_button_pos[1] + button_size[1]), 
                 button_undo, -1)
    cv2.putText(frame, 'Undo', (undo_button_pos[0] + 5, undo_button_pos[1] + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Space button
    button_space = (220, 220, 220) if button_hover['space'] else (200, 200, 200)
    cv2.rectangle(frame, space_bar_pos, 
                 (space_bar_pos[0] + space_bar_width, space_bar_pos[1] + space_bar_height), 
                 button_space, -1)
    cv2.rectangle(frame, space_bar_pos, 
                 (space_bar_pos[0] + space_bar_width, space_bar_pos[1] + space_bar_height), 
                 (50, 50, 50), 2)
    cv2.putText(frame, 'SPACE', (space_bar_pos[0] + space_bar_width//2 - 30, space_bar_pos[1] + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Update text display
    text_display.fill(255)
    cv2.putText(text_display, f"Predicted Text: {current_text}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get index fingertip coordinates (landmark 8)
            fingertip = hand_landmarks.landmark[8]
            x = int(fingertip.x * frame.shape[1])
            y = int(fingertip.y * frame.shape[0])

            # Check button interactions
            if button_hover['clear']:
                canvas = np.zeros_like(frame)
                whiteboard = np.ones_like(frame) * 255
                current_stroke = []
                strokes = []
                current_text = ""
            
            elif button_hover['blue']:
                pen_color = (255, 0, 0)  # Blue in BGR
            
            elif button_hover['green']:
                pen_color = (0, 255, 0)  # Green in BGR
            
            elif button_hover['red']:
                pen_color = (0, 0, 255)  # Red in BGR
                
            # Check if undo button is pressed (with cooldown)
            elif button_hover['undo'] and current_time - last_undo_press_time > button_cooldown:
                if current_text:
                    current_text = current_text[:-1]
                    last_undo_press_time = current_time
            
            # Check if space button is pressed (with cooldown)
            elif button_hover['space'] and current_time - last_space_press_time > button_cooldown:
                current_text += " "
                last_space_press_time = current_time

            # Handle drawing
            middle_tip_y = hand_landmarks.landmark[12].y * frame.shape[0]
            drawing = y < middle_tip_y

            # Draw laser effect at index fingertip with current pen color
            draw_laser_effect(frame, x, y, pen_color)

            if drawing:
                if previous_position:
                    cv2.line(canvas, previous_position, (x, y), pen_color, pen_thickness)
                    cv2.line(whiteboard, previous_position, (x, y), pen_color, pen_thickness)
                    current_stroke.append((x, y))
                previous_position = (x, y)
            else:
                if current_stroke and current_time - last_prediction_time > prediction_delay:
                    predicted_char = predict_text(current_stroke)
                    if predicted_char:
                        current_text += predicted_char
                    current_stroke = []
                    last_prediction_time = current_time
                    canvas = np.zeros_like(frame)
                    whiteboard = np.ones_like(frame) * 255
                previous_position = None

    # Combine frame with canvas
    combined_frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    
    # Combine whiteboard with text display
    combined_whiteboard = np.vstack([whiteboard, text_display])
    
    # Display windows
    cv2.imshow("Camera Feed with Air Drawing", combined_frame)
    cv2.imshow("Whiteboard with Predictions", combined_whiteboard)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
