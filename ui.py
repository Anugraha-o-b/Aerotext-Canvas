import tkinter as tk
from PIL import Image, ImageTk
import subprocess

# Function to start air-writing (without text conversion)
def start_writing():
    subprocess.Popen(["python", "air_drawing.py"])  # Replace with your actual script

# Function to start air-writing with text conversion
def convert_to_text():
    subprocess.Popen(["python", "areotext.py"]) # Replace with subprocess.call(["python", "air_drawing.py"])

# Function to exit the application
def exit_app():
    root.destroy()

# Function to open rules page
def open_rules():
    subprocess.Popen(["python", "rules.py"])

# Create the main window
root = tk.Tk()
root.title("Aerotext Canvas")
root.attributes('-fullscreen', True)  # Make window fullscreen

# Load and set the background image
bg_image = Image.open("1.png")  # Replace with actual background image
bg_image = bg_image.resize((1920, 1080), Image.Resampling.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)

bg_label = tk.Label(root, image=bg_photo)
bg_label.place(relwidth=1, relheight=1)

# Function to create a centered stylish button
def create_stylish_button(text, y_offset, command=None):
    width = 400  # Button width
    height = 80  # Button height
    x = (root.winfo_screenwidth() - width) // 2  # Center the button horizontally
    y = y_offset  # Position based on given y offset
    
    btn_canvas = tk.Canvas(root, width=width, height=height, highlightthickness=0, bg="black", bd=0)
    btn_canvas.place(x=x, y=y)
    
    # Create a rounded rectangle effect
    #btn_canvas.create_oval(5, 5, height, height, fill="#ffcc88", outline="")  # Left rounded corner
    #btn_canvas.create_oval(width - height, 5, width - 5, height, fill="#ffcc88", outline="")  # Right rounded corner
    #btn_canvas.create_rectangle(height // 2, 5, width - height // 2, height, fill="#ffcc88", outline="")  # Center part
    
    # Transparent text button
    btn = tk.Button(root, text=text, font=("Arial", 15, "bold"), fg="black",
                    bg="#ffffff", activebackground="#ffaa55", borderwidth=0,
                    relief="flat", command=command)
    btn.place(x=x, y=y, width=width, height=height)

# Function to create a hyperlink at the bottom
def create_hyperlink():
    # Create a frame with dark background for better visibility
    link_frame = tk.Frame(root, bg="#ffffff", padx=10, pady=5)
    link_frame.place(relx=0.5, rely=0.95, anchor="center")
    
    # Create the hyperlink label
    link_label = tk.Label(
        link_frame, 
        text="How to use Aerotext Canvas", 
        font=("Arial", 12, "underline"), 
        fg="#000000",
        bg="#ffffff",
        cursor="hand2"
    )
    link_label.pack()
    
    # Bind click event to open rules.py
    link_label.bind("<Button-1>", lambda e: open_rules())

# Dynamically positioned buttons
screen_height = root.winfo_screenheight()
create_stylish_button("START WRITING", screen_height // 3, start_writing)
create_stylish_button("CONVERT TO TEXT", screen_height // 2, convert_to_text)
create_stylish_button("EXIT", int(screen_height * 0.7), exit_app)

# Add hyperlink at the bottom
create_hyperlink()

# Exit fullscreen with Escape key
root.bind("<Escape>", lambda event: root.attributes("-fullscreen", False))

# Run the Tkinter event loop
root.mainloop()
