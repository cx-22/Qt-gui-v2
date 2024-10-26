import torch
import cv2
import numpy as np
import sys
#import warnings
import logging
import os
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QPushButton,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QComboBox,
)
from functools import partial
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import QTimer
from ultralytics import YOLO
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, ssd300_vgg16, retinanet_resnet50_fpn


'''
import GPUtil
import time

# Function to monitor GPU usage
def monitor_gpu():
    while True:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"GPU ID: {gpu.id}, GPU Usage: {gpu.memoryUtil * 100:.1f}%, GPU Memory Used: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
        print("-" * 50)  # Separator
        time.sleep(1)  # Refresh every second

# Run the monitoring function in a separate thread
import threading
threading.Thread(target=monitor_gpu, daemon=True).start()

'''


#warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger('ultralytics').setLevel(logging.WARNING)

yolov8m = YOLO('yolov8m.pt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor()  # Convert image to PyTorch tensor
])


frcnn = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
frcnn.eval()

ssd = ssd300_vgg16(pretrained=True).to(device)
ssd.eval()

retinaNet = retinanet_resnet50_fpn(pretrained=True).to(device)
retinaNet.eval()




class MediaProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Method Comparison")
        self.setGeometry(100, 100, 1200, 675)

        # Set font size for UI elements
        font = QFont()
        font.setPointSize(12)
        self.setFont(font)

        # Layouts
        self.main_layout = QHBoxLayout()  # Main layout for media and controls
        self.control_layout = QVBoxLayout()  # For buttons and dropdowns

        # Load Source Folder button
        self.load_button = QPushButton("Load Source Folder")
        self.load_button.clicked.connect(self.load_folder)
        self.load_button.setFixedHeight(50)  # Make the button larger
        self.load_button.setFixedWidth(200)

        # Original Media label
        self.original_media_label = QLabel("Original Media")
        self.original_media_label.setAlignment(QtCore.Qt.AlignCenter)  # Center the text
        self.original_media_label.setFixedSize(720, 720)  # Set fixed size of 720x720px

        # Processed Media label
        self.processed_media_label = QLabel("Processed Media")
        self.processed_media_label.setAlignment(QtCore.Qt.AlignCenter)  # Center the text
        self.processed_media_label.setFixedSize(720, 720)  # Set fixed size of 720x720px

        # Media display layout
        self.media_layout = QHBoxLayout()  # Use horizontal layout for side-by-side
        self.media_layout.addWidget(self.original_media_label)
        self.media_layout.addWidget(self.processed_media_label)

        # Control Widgets
        self.operation_combo_original = QComboBox()
        self.operation_combo_original.addItem("Grayscale")
        self.operation_combo_original.addItem("Flip Upside Down")
        self.operation_combo_original.addItem("Yolov8m")
        self.operation_combo_original.addItem("Faster R-CNN")
        self.operation_combo_original.addItem("SSD")
        self.operation_combo_original.addItem("RetinaNet")
        self.operation_combo_original.setFixedHeight(50)  # Make the combo box larger
        self.operation_combo_original.setFixedWidth(200)

        self.operation_combo_processed = QComboBox()
        self.operation_combo_processed.addItem("Grayscale")
        self.operation_combo_processed.addItem("Flip Upside Down")
        self.operation_combo_processed.addItem("Yolov8m")
        self.operation_combo_processed.addItem("Faster R-CNN")
        self.operation_combo_processed.addItem("SSD")
        self.operation_combo_processed.addItem("RetinaNet")
        self.operation_combo_processed.setFixedHeight(50)  # Make the combo box larger
        self.operation_combo_processed.setFixedWidth(200)

        self.file_combo = QComboBox()
        self.file_combo.currentIndexChanged.connect(self.load_selected_file)
        self.file_combo.setFixedHeight(50)  # Make the combo box larger

        self.convert_button = QPushButton("Convert")
        self.convert_button.clicked.connect(self.convert_media)
        self.convert_button.setFixedHeight(50)  # Make the button larger
        self.convert_button.setFixedWidth(200)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.pause_button.setFixedHeight(50)  # Make the button larger
        self.pause_button.setFixedWidth(200)

        # Add padding to buttons and dropdowns
        for button in [self.load_button, self.convert_button, self.pause_button]:
            button.setStyleSheet("padding: 10px; font-size: 16px;")  # Increase font size

        for combo in [self.operation_combo_original, self.operation_combo_processed, self.file_combo]:
            combo.setStyleSheet("padding: 5px; font-size: 16px;")  # Increase font size

        # Add controls to the control layout
        self.control_layout.addWidget(self.load_button)  # Move Load Source Folder to the top
        self.control_layout.addWidget(self.file_combo)
        self.control_layout.addWidget(QLabel("Original Operation"))
        self.control_layout.addWidget(self.operation_combo_original)
        self.control_layout.addWidget(QLabel("Processed Operation"))
        self.control_layout.addWidget(self.operation_combo_processed)
        self.control_layout.addWidget(self.convert_button)
        self.control_layout.addWidget(self.pause_button)

        # Combine layouts
        self.main_layout.addLayout(self.control_layout)
        self.main_layout.addLayout(self.media_layout)

        # Set the central widget
        container = QWidget()
        container.setLayout(self.main_layout)
        self.setCentralWidget(container)

        # Timer for updating video frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.media_type = None  # Will store 'image' or 'video'
        self.cap = None  # For video capturing
        self.media_files = []  # List of media files from the folder
        self.current_frame = None  # Store the current image or video frame
        self.is_paused = False  # Pause state

        # Define the mapping of operation names to their functions
        self.operation_functions = {
            "Grayscale": self.apply_grayscale,
            "Flip Upside Down": self.apply_flip_upside_down,
            "Yolov8m": self.apply_yolov8m,
            "Faster R-CNN": partial(self.apply_frcnn_or_sdd, model="frcnn"),
            "SSD": partial(self.apply_frcnn_or_sdd, model="ssd"),
            "RetinaNet": self.appl_rNet
        }

    def load_folder(self):
        # Open a file dialog to select a folder containing media files
        folder_path = QFileDialog.getExistingDirectory(self, "Open Source Folder", "")
        if folder_path:
            self.media_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.mp4', '.avi', '.mov', '.mkv'))]
            self.folder_path = folder_path
            self.file_combo.clear()
            self.file_combo.addItems(self.media_files)

    def load_selected_file(self):
        selected_file = self.file_combo.currentText()
        if selected_file:
            file_path = os.path.join(self.folder_path, selected_file)
            file_extension = os.path.splitext(file_path)[-1].lower()
            if file_extension in ['.png', '.jpg', '.jpeg', '.bmp']:
                self.load_image(file_path)
            elif file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
                self.load_video(file_path)
            else:
                print("Unsupported file format.")

    def load_image(self, file_path):
        # Load the image using OpenCV
        self.image = cv2.imread(file_path)
        if self.image is not None:
            self.media_type = 'image'
            self.current_frame = self.image.copy()
            self.display_frame(self.image, self.original_media_label)
            self.process_and_display_both(self.image)  # Process the image immediately for both sides
        else:
            print("Error loading image.")

        self.timer.stop()  # Stop video playback if switching to an image

    def load_video(self, file_path):
        # Capture the video using OpenCV
        self.cap = cv2.VideoCapture(file_path)
        if self.cap.isOpened():
            self.media_type = 'video'
            self.is_paused = False  # Reset pause state
            self.timer.start(30)  # Update every 30ms (approx 30 FPS)
        else:
            print("Error opening video file.")

    def update_frame(self):
        if self.media_type == 'video' and self.cap.isOpened() and not self.is_paused:
            ret, frame = self.cap.read()
            if ret:
                # Store the current video frame
                self.current_frame = frame
                self.display_frame(frame, self.original_media_label)
                
                # Process and display both operations
                self.process_and_display_both(frame)
            else:
                # Loop the video
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the first frame

    def convert_media(self):
        # Convert the currently displayed frame or image
        if self.current_frame is not None:
            self.process_and_display_both(self.current_frame)

    def process_and_display_both(self, frame):
        # Process the frame based on the selected operation for the left side
        operation_name_original = self.operation_combo_original.currentText()
        operation_name_processed = self.operation_combo_processed.currentText()

        # Create a copy of the frame for original processing
        original_frame_copy = frame.copy()
        # Process the original frame based on the selected operation for the left side
        if operation_name_original in self.operation_functions:
            original_frame = self.operation_functions[operation_name_original](original_frame_copy)
            self.display_frame(original_frame, self.original_media_label)

        # Create a copy of the frame for processed processing
        processed_frame_copy = frame.copy()
        # Process the original frame for the right side (processed media)
        if operation_name_processed in self.operation_functions:
            processed_frame = self.operation_functions[operation_name_processed](processed_frame_copy)
            self.display_frame(processed_frame, self.processed_media_label)

    def apply_grayscale(self, frame):
        """Convert the given frame to grayscale."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def apply_flip_upside_down(self, frame):
        """Flip the given frame upside down."""
        return cv2.flip(frame, 0)
    

    '''
                MOOOOOOOOOOOODDDDDDDDDDDDDEEEEEEEEEEELLLLLLLLLLSSSSSSSSSSSSSSSSSS
    '''


    '''
            YOLLOOOO V8888888
    '''

    def apply_yolov8m(self, frame):
        outframe = frame
        output = yolov8m(frame)
        boxes = output[0].boxes.xyxy.cpu().numpy()
        class_ids = output[0].boxes.cls.cpu().numpy()  # Class IDs
        confidences = output[0].boxes.conf.cpu().numpy()
        
        # Filter for 'person' class (class_id = 0)
        peopleBoxes = boxes[class_ids == 0]

        pop = len(peopleBoxes)

        # Draw boxes for every person found with colors based on confidence, blue = higher
        for i, box in enumerate(peopleBoxes):
            x1, y1, x2, y2 = map(int, box)
            confidence = confidences[i]
            color = (int(255 * confidence), 0, int(255 * (1 - confidence)))
            cv2.rectangle(outframe, (x1, y1), (x2, y2), color, 2)
            
        cv2.putText(outframe, f"Found: {pop}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1.3, (255, 255, 255), 2)
                
        return outframe

    '''
            FASTERRRRRRRCNN     ANDD SSDDDD
    '''

    def apply_frcnn_or_sdd(self, frame, model):
        outFrame = frame.copy()

        # Convert the frame to a tensor
        image = transform(outFrame).unsqueeze(0).to(device)  # Add a batch dimension and move to GPU

        if model == "frcnn":
            with torch.no_grad():
                outputs = frcnn(image)
        elif model == "ssd":
            with torch.no_grad():
                outputs = ssd(image)


        # Extract bounding boxes, scores, and labels
        boxes = outputs[0]['boxes'].cpu().numpy()  # Move boxes back to CPU
        scores = outputs[0]['scores'].cpu().numpy()  # Move scores back to CPU
        labels = outputs[0]['labels'].cpu().numpy()  # Move labels back to CPU

        # Filter results based on a confidence threshold
        confidence_threshold = 0.5
        indices = scores > confidence_threshold
        boxes = boxes[indices]
        scores = scores[indices]
        labels = labels[indices]

        # Initialize a counter for detected persons
        person_count = 0

        # Draw boxes only for the 'person' class (label 1)
        for box, score, label in zip(boxes, scores, labels):
            if label.item() == 1:  # Check if the detected label is 'person'
                x1, y1, x2, y2 = box.astype(int)

                # Draw the bounding box with dynamic color
                confidence = score.item()  # Get the confidence score
                color = (int(255 * confidence), 0, int(255 * (1 - confidence)))  # Color gradient from blue to red
                
                cv2.rectangle(outFrame, (x1, y1), (x2, y2), color, 2)  # Draw box with dynamic color

                # Increment the person count
                person_count += 1

        # Add the count of detected persons to the frame
        cv2.putText(outFrame, f"Found: {person_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.3, (255, 255, 255), 2)

        return outFrame
    
    def appl_rNet(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert image to tensor and normalize
        image_tensor = transform(image_rgb).unsqueeze(0).to(device)  # Add batch dimension and move to GPU

        # Perform inference
        with torch.no_grad():
            predictions = retinaNet(image_tensor)

        # Filter predictions for person class (COCO class index for 'person' is 1)
        person_boxes = predictions[0]['boxes'][predictions[0]['labels'] == 1]
        person_confidence = predictions[0]['scores'][predictions[0]['labels'] == 1]
        person_count = 0
        # Draw boxes around detected people
        for box, score in zip(person_boxes, person_confidence):
            if score >= 0.5:  # You can adjust the threshold
                # Convert box coordinates to integers
                color = (int(255 * score), 0, int(255 * (1 - score)))
                x1, y1, x2, y2 = map(int, box.int().tolist())  # Ensure they are integers
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Draw box
                person_count += 1
        cv2.putText(frame, f"Found: {person_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
            1.3, (255, 255, 255), 2)
        return frame

    






#   ENDDDD MODEEEEELSSSSSSSSSSS

    def display_frame(self, frame, label):
            """Display the given frame on the specified label, scaled to fit within 720x720 pixels."""
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytes_per_line = channel * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Scale the image to fit within the 720x720 label, maintaining the aspect ratio
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(720, 720, QtCore.Qt.KeepAspectRatio)

            label.setPixmap(scaled_pixmap)

    def toggle_pause(self):
            # Toggle pause state
            if self.media_type == 'video':
                self.is_paused = not self.is_paused
                if self.is_paused:
                    self.pause_button.setText("Play")
                    self.timer.stop()  # Stop the timer if paused
                else:
                    self.pause_button.setText("Pause")
                    self.timer.start(30)  # Restart the timer to continue playback

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MediaProcessor()
    window.show()
    sys.exit(app.exec_())
