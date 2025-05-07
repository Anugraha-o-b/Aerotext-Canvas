import tkinter as tk
from PIL import Image, ImageTk

# Create the rules window
rules_window = tk.Tk()
rules_window.title("How to Use Aerotext Canvas")
rules_window.geometry("800x600")
rules_window.configure(bg="#f0f0f0")

# Create a frame for the rules content
content_frame = tk.Frame(rules_window, bg="#f0f0f0", padx=40, pady=40)
content_frame.pack(expand=True, fill="both")

# Title
title_label = tk.Label(
    content_frame, 
    text="How to Use Aerotext Canvas", 
    font=("Arial", 24, "bold"), 
    bg="#f0f0f0",
    fg="#333333"
)
title_label.pack(pady=(0, 30))

# Create the 4 simple rule points
rules = [
    "1. Hold your hand in the air and move to draw shapes or write text",
    "2. Your index finger  should be higher than all other fingers to enable the pen",
    "3. Lower index finger will stop the pen activation",
    "4. Press ESC button to land back to home page",
    "5. SPACE and UNDO button time is 1sec.Move your fingers as fast as possible",
    "6. When finished, press EXIT to close the application"
]

# Add each rule as a separate label
for rule in rules:
    rule_label = tk.Label(
        content_frame,
        text=rule,
        font=("Arial", 14),
        bg="#f0f0f0",
        fg="#333333",
        anchor="w",
        justify="left",
        pady=10
    )
    rule_label.pack(fill="x")

# Close button
close_button = tk.Button(
    content_frame,
    text="Close",
    font=("Arial", 12, "bold"),
    bg="#2196F3",
    fg="white",
    padx=20,
    pady=10,
    relief="flat",
    command=rules_window.destroy
)
close_button.pack(pady=30)

# Center the window on the screen
rules_window.update_idletasks()
width = rules_window.winfo_width()
height = rules_window.winfo_height()
x = (rules_window.winfo_screenwidth() // 2) - (width // 2)
y = (rules_window.winfo_screenheight() // 2) - (height // 2)
rules_window.geometry('{}x{}+{}+{}'.format(width, height, x, y))

# Run the rules window
rules_window.mainloop()