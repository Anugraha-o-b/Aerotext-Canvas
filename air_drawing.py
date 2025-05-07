import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize OpenCV window and drawing canvases
cap = cv2.VideoCapture(0)
canvas = None
whiteboard = None
drawing = False
previous_position = None

# Colors and brush settings
pen_color = (0, 0, 255)  # Default pen color (red)
pen_thickness = 5
is_eraser_mode = False  # Flag to track if eraser is active
eraser_thickness = 20   # Eraser is thicker than pen

# Define button positions and dimensions
button_size = (50, 50)
clear_button_pos = (10, 10)
blue_button_pos = (70, 10)
green_button_pos = (130, 10)
red_button_pos = (190, 10)
eraser_button_pos = (250, 10)  # New eraser button position

def draw_laser_effect(frame, x, y, color=(0, 0, 255)):
    # Draw a glowing circle at fingertip
    cv2.circle(frame, (x, y), 5, color, -1)
    # Add glow effect with increasing radii and decreasing opacity
    for radius in range(6, 15):
        alpha = 0.2 - (radius - 6) * 0.02  # Decreasing opacity for outer circles
        overlay = frame.copy()
        cv2.circle(overlay, (x, y), radius, color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

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

    # Draw buttons
    cv2.rectangle(frame, clear_button_pos, 
                 (clear_button_pos[0] + button_size[0], clear_button_pos[1] + button_size[1]), 
                 (0, 0, 0), -1)
    cv2.putText(frame, 'Clear', (clear_button_pos[0] + 5, clear_button_pos[1] + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.rectangle(frame, blue_button_pos, 
                 (blue_button_pos[0] + button_size[0], blue_button_pos[1] + button_size[1]), 
                 (255, 0, 0), -1)
    cv2.rectangle(frame, green_button_pos, 
                 (green_button_pos[0] + button_size[0], green_button_pos[1] + button_size[1]), 
                 (0, 255, 0), -1)
    cv2.rectangle(frame, red_button_pos, 
                 (red_button_pos[0] + button_size[0], red_button_pos[1] + button_size[1]), 
                 (0, 0, 255), -1)
    
    # Draw eraser button
    cv2.rectangle(frame, eraser_button_pos, 
                 (eraser_button_pos[0] + button_size[0], eraser_button_pos[1] + button_size[1]), 
                 (200, 200, 200), -1)  # Light gray color for eraser button
    cv2.putText(frame, 'Eraser', (eraser_button_pos[0], eraser_button_pos[1] + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Highlight the active tool (pen or eraser)
    active_button_pos = eraser_button_pos if is_eraser_mode else (blue_button_pos if pen_color == (255, 0, 0) else 
                         (green_button_pos if pen_color == (0, 255, 0) else red_button_pos))
    cv2.rectangle(frame, active_button_pos, 
                 (active_button_pos[0] + button_size[0], active_button_pos[1] + button_size[1]), 
                 (255, 255, 255), 2)  # White highlight around active tool

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get index fingertip coordinates (landmark 8)
            fingertip = hand_landmarks.landmark[8]
            x = int(fingertip.x * frame.shape[1])
            y = int(fingertip.y * frame.shape[0])

            # Check button interactions
            if clear_button_pos[0] < x < clear_button_pos[0] + button_size[0] and clear_button_pos[1] < y < clear_button_pos[1] + button_size[1]:
                canvas = np.zeros_like(frame)
                whiteboard = np.ones_like(frame) * 255
            
            elif blue_button_pos[0] < x < blue_button_pos[0] + button_size[0] and blue_button_pos[1] < y < blue_button_pos[1] + button_size[1]:
                pen_color = (255, 0, 0)
                is_eraser_mode = False
            
            elif green_button_pos[0] < x < green_button_pos[0] + button_size[0] and green_button_pos[1] < y < green_button_pos[1] + button_size[1]:
                pen_color = (0, 255, 0)
                is_eraser_mode = False
            
            elif red_button_pos[0] < x < red_button_pos[0] + button_size[0] and red_button_pos[1] < y < red_button_pos[1] + button_size[1]:
                pen_color = (0, 0, 255)
                is_eraser_mode = False
                
            elif eraser_button_pos[0] < x < eraser_button_pos[0] + button_size[0] and eraser_button_pos[1] < y < eraser_button_pos[1] + button_size[1]:
                is_eraser_mode = True

            # Handle drawing
            middle_tip_y = hand_landmarks.landmark[12].y * frame.shape[0]
            drawing = y < middle_tip_y

            # Choose the effect color based on mode
            effect_color = (200, 200, 200) if is_eraser_mode else pen_color
            
            # Draw laser effect at index fingertip
            draw_laser_effect(frame, x, y, effect_color)

            if drawing:
                if previous_position:
                    if is_eraser_mode:
                        # Draw white lines (erase) on both canvases with thicker stroke
                        cv2.line(canvas, previous_position, (x, y), (0, 0, 0), eraser_thickness*2)
                        cv2.line(whiteboard, previous_position, (x, y), (255, 255, 255), eraser_thickness*2)
                    else:
                        # Draw with selected color
                        cv2.line(canvas, previous_position, (x, y), pen_color, pen_thickness)
                        cv2.line(whiteboard, previous_position, (x, y), pen_color, pen_thickness)
                previous_position = (x, y)
            else:
                previous_position = None

    # Combine frame with canvas
    combined_frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    
    # Display windows
    cv2.imshow("Camera Feed with Air Drawing", combined_frame)
    cv2.imshow("Whiteboard", whiteboard)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()