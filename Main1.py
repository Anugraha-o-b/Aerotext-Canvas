import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hand Detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize OpenCV window and drawing canvases
cap = cv2.VideoCapture(0)
canvas = None  # Overlay canvas for the camera feed
whiteboard = None  # Separate whiteboard window
drawing = False  # Flag to toggle drawing mode
previous_position = None  # Stores previous fingertip position

# Colors and brush settings
pen_color = (0, 0, 255)  # Default pen color (red)
pen_thickness = 5

# Define button positions and dimensions
button_size = (50, 50)  # Width and height of buttons
clear_button_pos = (10, 10)  # Position of clear button
blue_button_pos = (70, 10)  # Position of blue button
green_button_pos = (130, 10)  # Position of green button
red_button_pos = (190, 10)  # Position of red button

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to capture frame from camera.")
        break

    # Flip the image horizontally and convert to RGB for Mediapipe processing
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks
    results = hands.process(image_rgb)

    # Initialize canvas and whiteboard if not done already (matches the frame size)
    if canvas is None:
        canvas = np.zeros_like(frame)  # Black canvas with same dimensions as the frame
        whiteboard = np.ones_like(frame) * 255  # Whiteboard is a white canvas

    # Draw the buttons on the frame
    cv2.rectangle(frame, clear_button_pos, (clear_button_pos[0] + button_size[0], clear_button_pos[1] + button_size[1]), (0, 0, 0), -1)  # Clear button (black)
    cv2.putText(frame, 'Clear', (clear_button_pos[0] + 5, clear_button_pos[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.rectangle(frame, blue_button_pos, (blue_button_pos[0] + button_size[0], blue_button_pos[1] + button_size[1]), (255, 0, 0), -1)  # Blue button
    cv2.rectangle(frame, green_button_pos, (green_button_pos[0] + button_size[0], green_button_pos[1] + button_size[1]), (0, 255, 0), -1)  # Green button
    cv2.rectangle(frame, red_button_pos, (red_button_pos[0] + button_size[0], red_button_pos[1] + button_size[1]), (0, 0, 255), -1)  # Red button

    # Check if any hand landmarks were detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get coordinates of the index finger tip (landmark 8)
            fingertip = hand_landmarks.landmark[8]
            x = int(fingertip.x * frame.shape[1])
            y = int(fingertip.y * frame.shape[0])

            # Check if finger is hovering over any button
            if clear_button_pos[0] < x < clear_button_pos[0] + button_size[0] and clear_button_pos[1] < y < clear_button_pos[1] + button_size[1]:
                # Clear both canvas and whiteboard
                canvas = np.zeros_like(frame)
                whiteboard = np.ones_like(frame) * 255
                #print("Canvas cleared")

            elif blue_button_pos[0] < x < blue_button_pos[0] + button_size[0] and blue_button_pos[1] < y < blue_button_pos[1] + button_size[1]:
                pen_color = (255, 0, 0)  # Change color to blue
                #print("Changed color to blue")

            elif green_button_pos[0] < x < green_button_pos[0] + button_size[0] and green_button_pos[1] < y < green_button_pos[1] + button_size[1]:
                pen_color = (0, 255, 0)  # Change color to green
                #print("Changed color to green")

            elif red_button_pos[0] < x < red_button_pos[0] + button_size[0] and red_button_pos[1] < y < red_button_pos[1] + button_size[1]:
                pen_color = (0, 0, 255)  # Change color to red
                #print("Changed color to red")

            # Toggle drawing mode based on the finger state (e.g., index higher than middle finger)
            middle_tip_y = hand_landmarks.landmark[12].y * frame.shape[0]
            drawing = y < middle_tip_y  # Set drawing mode if index is above middle finger

            # Draw only if drawing mode is active
            if drawing:
                # Start drawing if previous position exists
                if previous_position:
                    # Draw on both the camera canvas and the whiteboard
                    cv2.line(canvas, previous_position, (x, y), pen_color, pen_thickness)
                    cv2.line(whiteboard, previous_position, (x, y), pen_color, pen_thickness)

                # Update previous position
                previous_position = (x, y)
            else:
                # Reset previous position when drawing is inactive
                previous_position = None

            # Draw landmarks on the frame for visual feedback
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Combine the canvas with the original frame for the camera feed
    combined_frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Display both windows: camera feed with drawing and the whiteboard
    cv2.imshow("Camera Feed with Air Drawing", combined_frame)
    cv2.imshow("Whiteboard", whiteboard)

    # Exit when the Esc key is pressed
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
