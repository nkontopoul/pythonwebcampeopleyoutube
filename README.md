Program Overview
This Python program performs real-time person detection on a live YouTube video stream. It uses the YOLOv5 model for object detection, specifically to identify humans in the video frames. The program draws bounding boxes around detected persons and displays the live video with these annotations.

Technologies Used
Python: The primary programming language used to write the script.
PyTorch: A deep learning framework used to load and run the YOLOv5 model.
OpenCV: A computer vision library used to handle video capture and display frames with annotations.
Streamlink: A command-line utility that extracts video streams from various services (like YouTube) and provides a direct URL for streaming.
YOLOv5: A state-of-the-art object detection model known for its speed and accuracy. The yolov5s variant is a smaller, faster version suitable for real-time applications.
ImageIO and ImageIO-FFMPEG: (Not used in the final implementation) These were initially used for reading video streams but replaced by cv2.VideoCapture for simplicity and compatibility.
How It Works
Stream URL Extraction: The program uses streamlink to fetch the actual streaming URL of a live YouTube video.
Video Capture Initialization: OpenCV's cv2.VideoCapture is used to open the video stream.
Frame Processing:
For each frame captured from the video stream, the frame is resized for faster processing.
The YOLOv5 model processes the frame to detect persons.
For each detected person, a bounding box is drawn around them, and a label "Human" is added.
Display: The processed frame with bounding boxes and labels is displayed in a window.
Exit Mechanism: The program continuously processes and displays frames until the user presses 'q' to exit.
Requirements
To run this program, you need to install the required Python packages. Below is the requirements.txt file that lists all the dependencies:
