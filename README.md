

# Face Detection and Skin Color Detection

This Python-based project combines facial detection with skin color analysis using OpenCV and Tkinter. It features a camera-based interface for live detection and an image uploader for static image analysis.

---

## Features
- **Face Detection**: Identifies faces in real-time using the Haar Cascade classifier.
- **Skin Color Detection**: Detects and labels skin tones from a predefined set using color matching.
- **Real-Time Camera Input**: Captures live feed from the camera for detection.
- **Image Upload Support**: Allows users to upload images for face and skin tone analysis.
- **Voice Notification**: Notifies users of the detected skin tone via text-to-speech.

---

## Requirements
1. Python 3.x
2. OpenCV
3. NumPy
4. pyttsx3
5. Tkinter

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-github-username>/face-skin-detection.git
   cd face-skin-detection
   ```
2. Install the required packages:
   ```bash
   pip install opencv-python-headless numpy pyttsx3
   ```
3. Run the application:
   ```bash
   python main.py
   ```

---

## How to Use
1. Launch the application.
2. Choose between **Open Camera** or **Upload Image**:
   - **Open Camera**: Use your device's camera to detect faces and skin tones in real-time.
   - **Upload Image**: Analyze a selected image to detect faces and determine skin tones.
3. Quit the application by pressing **q** in the camera feed window or closing the GUI.

---

## Example Output
- Detected faces will be highlighted in the image or video feed.
- Skin tone will be labeled and announced via text-to-speech.

---

## Contributing
Feel free to fork the repository and submit pull requests to improve the project.

---

## License
This project is open-source and available under the [MIT License](LICENSE).
```

