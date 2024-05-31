import torch
import cv2
import numpy as np
import subprocess
import json

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# YouTube video URL
youtube_url = 'https://www.youtube.com/watch?v=wZYX5pbR4VY'  # Replace with your YouTube video URL

# Use streamlink to fetch the stream URL
def get_stream_url(youtube_url):
    result = subprocess.run(
        ['streamlink', '--json', youtube_url, 'best'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        raise Exception(f"Streamlink error: {result.stderr.decode()}")
    stream_info = json.loads(result.stdout)
    return stream_info['url']

# Get the actual stream URL
try:
    video_url = get_stream_url(youtube_url)
except Exception as e:
    print(f"Error fetching stream URL: {e}")
    exit()

# Initialize video capture
cap = cv2.VideoCapture(video_url)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Function to count persons in a frame and draw bounding boxes
def count_persons(frame):
    results = model(frame)
    persons = 0
    for detection in results.xyxy[0]:
        if detection[5] == 0:  # The class ID for 'person' in COCO dataset
            persons += 1
            # Draw bounding box
            x1, y1, x2, y2 = map(int, detection[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Add label
            cv2.putText(frame, 'Human', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return persons, frame

# Loop through video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for faster processing (optional)
    frame_resized = cv2.resize(frame, (1280, 720))

    # Detect persons in the frame
    persons, frame_with_detections = count_persons(frame_resized)

    # Display the frame with detection
    cv2.imshow('YOLOv5 Person Detection', frame_with_detections)
    print(f'Persons detected: {persons}')

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
