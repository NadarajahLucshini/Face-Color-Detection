import cv2
import numpy as np
import pyttsx3
import time
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog

# Initialize the camera
cap = None

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Predefined skin colors with BGR values and descriptions
skin_colors = {
    "Lightest Beige": ((165, 204, 232), "Very light beige, almost porcelain"),
    "Light Beige 2": ((186, 207, 239), "Slightly warmer light beige"),
    "Light Beige 3": ((193, 221, 245), "Light beige with pinkish undertones"),
    "Light Yellowish": ((168, 227, 246), "Light skin with yellow undertones"),
    "Light Tan": ((143, 198, 243), "Light tan, golden hue"),
    "Tan": ((139, 193, 240), "Classic tan skin tone"),
    "Medium Tan": ((146, 189, 228), "Medium tan, slightly deeper"),
    "Medium Brown": ((125, 158, 208), "Medium brown skin"),
    "Brown": ((99, 151, 204), "A standard brown skin tone"),
    "Darker Brown": ((100, 139, 171), "Darker shade of brown"),
    "Dark Brown 2": ((44, 81, 133), "Rich dark brown"),
    "Dark Reddish Brown": ((43, 74, 125), "Dark brown with strong red tones"),
    "Deep Black": ((16, 30, 48), "Deepest black skin tone"),
}

def get_skin_color(avg_bgr):
    """Finds the closest skin tone name with advanced handling for light colors."""
    min_diff = float('inf')
    closest_color = "Unknown"
    threshold = 60  # Adjusted threshold for matching lighter skin tones
    delta_e_function = cv2.COLOR_BGR2Lab  # Using LAB color space for Delta E calculation

    for name, (bgr, _) in skin_colors.items():
        bgr = np.array(bgr, dtype=np.float32)
        avg_bgr = np.array(avg_bgr, dtype=np.float32)

        # Compute Delta E using LAB color space
        delta_e = cv2.norm(bgr, avg_bgr, cv2.NORM_L2)
        
        if delta_e < min_diff and delta_e < threshold:
            min_diff = delta_e
            closest_color = name

    return closest_color

def open_camera():
    global cap
    cap = cv2.VideoCapture(0)

    try:
        start_time = time.time()
        duration = 5  # 5 seconds

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            elapsed_time = time.time() - start_time
            remaining_time = max(0, duration - int(elapsed_time))

            cv2.putText(frame, f"Time Left: {remaining_time} sec", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Display skin color legends at the bottom
            y_offset = frame.shape[0] - 40  # Starting at the bottom of the frame
            for name, (bgr, description) in skin_colors.items():
                cv2.rectangle(frame, (10, y_offset), (50, y_offset + 30), bgr, -1)
                cv2.putText(frame, name, (60, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset -= 40  # Move up for the next color

            cv2.imshow('Face Detection with Skin Color Detection', frame)

            if remaining_time == 0:
                start_time = time.time()
                print("Processing...")

                for (x, y, w, h) in faces:
                    if x is not None and y is not None and w is not None and h is not None:
                        face_region = frame[y:y+h, x:x+w]
                        avg_bgr = cv2.mean(face_region)[:3]
                        avg_bgr = tuple(map(int, avg_bgr))
                        color_name = get_skin_color(avg_bgr)

                        color_box_x = x + w + 10
                        color_box_y = y
                        cv2.rectangle(frame, (color_box_x, color_box_y), (color_box_x + 30, color_box_y + 30), avg_bgr, -1)
                        cv2.putText(frame, color_name, (color_box_x + 40, color_box_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        engine = pyttsx3.init()
                        engine.say(f"Hi Luxshini, your detected skin color is {color_name}")
                        engine.runAndWait()
                        break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nExiting...")

    cap.release()
    cv2.destroyAllWindows()

# Function for uploading image
def upload_image():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        image = cv2.imread(file_path)

        if image is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                if x is not None and y is not None and w is not None and h is not None:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    avg_bgr = cv2.mean(image[y:y+h, x:x+w])[:3]
                    avg_bgr = tuple(map(int, avg_bgr))
                    color_name = get_skin_color(avg_bgr)

                    color_box_x = x + w + 10
                    color_box_y = y
                    cv2.rectangle(image, (color_box_x, color_box_y), (color_box_x + 30, color_box_y + 30), avg_bgr, -1)
                    cv2.putText(image, color_name, (color_box_x + 40, color_box_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    engine = pyttsx3.init()
                    engine.say(f"Hi Luxshini, your detected skin color is {color_name}")
                    engine.runAndWait()

                    cv2.imshow('Image Face Detection', image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
        else:
            messagebox.showerror("Error", "Invalid image path or unsupported format.")

# GUI for selecting options
def gui():
    window = tk.Tk()
    window.title("Face Detection & Skin Color Detection")
    window.geometry("300x200")
    window.configure(bg="#F0F0F0")

    tk.Label(window, text="Select an Option:", font=("Arial", 14), bg="#F0F0F0").pack(pady=10)

    btn_open_camera = tk.Button(window, text="Open Camera", command=open_camera, font=("Arial", 12), bg="#4CAF50", fg="white", padx=10, pady=5)
    btn_open_camera.pack(pady=5)

    btn_upload_image = tk.Button(window, text="Upload Image", command=upload_image, font=("Arial", 12), bg="#008CBA", fg="white", padx=10, pady=5)
    btn_upload_image.pack(pady=5)

    window.mainloop()

if __name__ == "__main__":
    gui()
