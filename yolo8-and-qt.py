import torch
import cv2
import numpy as np
import sys
#import warnings
import logging
import os
from ultralytics import YOLO
from PyQt5 import QtCore
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
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import QTimer

logging.getLogger('ultralytics').setLevel(logging.WARNING)
yolov8m = YOLO('yolov8m.pt')

class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.current_frame = None
        self.current_outFrame = None
        self.paused = False
        self.cap = None

        # Timer to help display video frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.frame_manage)

        # Main window name and dimensions
        self.setWindowTitle("Method Comparison")
        self.setGeometry(100, 100, 1200, 675)

        # Set font for all text in window
        font = QFont()
        font.setPointSize(12)
        self.setFont(font)

        self.main_layout = QHBoxLayout()
        self.io_side = QHBoxLayout()  # Layout for both og and processed files
        self.ui_side = QVBoxLayout()  # Layout for UI

        # Button to selected source folder
        self.load_button = QPushButton("Open Source Folder")
        self.load_button.clicked.connect(self.open_folder)
        self.load_button.setFixedHeight(50)
        self.load_button.setFixedWidth(200)

        # Source folder dropdown
        self.files_dropdown = QComboBox()
        self.files_dropdown.currentIndexChanged.connect(self.load_file)
        self.files_dropdown.setFixedHeight(50)

        self.ui_side.addWidget(self.load_button)
        self.ui_side.addWidget(self.files_dropdown)

        # Original file label
        self.og_label = QLabel("Original")
        self.og_label.setAlignment(QtCore.Qt.AlignCenter)  # Center contentrs
        self.og_label.setFixedSize(720, 720)  # Fix size 720/720

        # Processed file label
        self.processed_label = QLabel("Output")
        self.processed_label.setAlignment(QtCore.Qt.AlignCenter)
        self.processed_label.setFixedSize(720, 720)

        self.io_side = QHBoxLayout()  # Use horizontal layout for side-by-side
        self.io_side.addWidget(self.og_label)
        self.io_side.addWidget(self.processed_label)

        self.main_layout.addLayout(self.ui_side)
        self.main_layout.addLayout(self.io_side)

        # Set the central widget
        container = QWidget()
        container.setLayout(self.main_layout)
        self.setCentralWidget(container)


    # FUNCTIONNNNSS

    # This function will open a dialog prompt for a folder
    def open_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Open Source Folder", "")
        if folder_path:
            self.folder_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.mp4', '.avi', '.mov', '.mkv'))]
            self.folder_path = folder_path
            self.files_dropdown.clear()
            self.files_dropdown.addItems(self.folder_files)


    def load_file(self):

        # First get the name of the file from the dropdown
        file = self.files_dropdown.currentText()

        # Then use that name to get the full path to that file
        full_path = os.path.join(self.folder_path, file)

        # Then looks at just the extension and determines if its an image or video
        extension = os.path.splitext(full_path)[-1]
        if extension in ['.png', '.jpg', '.jpeg']:
            self.load_image(full_path)
        elif extension in ['.mp4', '.avi', '.mov', '.mkv']:
            self.load_video(full_path)
        else:
            print("Unsupported file format.")

    
    def load_image(self, path):
        self.timer.stop() # Ends any previous video playing
        self.current_frame = cv2.imread(path)
        self.process()
        self.display(self.current_frame, self.og_label)
        self.display(self.current_outFrame, self.processed_label)

    def load_video(self, path):
        self.cap = cv2.VideoCapture(path)
        if self.cap.isOpened:
            self.paused = False
            self.timer.start(30)  # Update every 30ms (approx 30 FPS)
        else:
            print("Couldn't open video file")
    
    def frame_manage(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            self.process()
            self.display(self.current_frame, self.og_label)
            self.display(self.current_outFrame, self.processed_label)
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loops the video
    
    # Display given frame to given label
    def display(self, frame, label):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytes_per_line = channel * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Scale to fit the label and maintain its aspect ratio
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(720, 720, QtCore.Qt.KeepAspectRatio)

            label.setPixmap(scaled_pixmap)

    def process(self):
        outFrame = self.current_frame.copy()
        self.current_outFrame = cv2.cvtColor(outFrame, cv2.COLOR_BGR2GRAY)
'''
    
    def process(self):
        outFrame = self.current_frame.copy()
        output = yolov8m(outFrame)
        boxes = output[0].boxes.xyxy.cpu().numpy()
        ids = output[0].boxes.cls.cpu().numpy()  # Class IDs
        confidences = output[0].boxes.conf.cpu().numpy()
        
        peopleBoxes = boxes[ids == 0] # Filter to just people (id 0)

        pop = len(peopleBoxes) # Get number of people found

        # Draw boxes for every person found with colors based on confidence, blue = higher
        for i, box in enumerate(peopleBoxes):
            x1, y1, x2, y2 = map(int, box)
            confidence = confidences[i]
            color = (int(255 * confidence), 0, int(255 * (1 - confidence)))
            cv2.rectangle(outFrame, (x1, y1), (x2, y2), color, 2)
            
        cv2.putText(outFrame, f"Found: {pop}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1.3, (255, 255, 255), 2)
                
        self.current_outFrame = outFrame
    '''


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = Window()
    myWindow.show()
    sys.exit(app.exec_())