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
from functools import partial
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import QTimer

logging.getLogger('ultralytics').setLevel(logging.WARNING)
yolov8m = YOLO('yolov8m.pt')

class Window:
    def __init__(self, name, breed):
        self.setWindowTitle("Method Comparison")
        self.setGeometry(100, 100, 1200, 675)

        font = QFont()
        font.setPointSize(12)
        self.setFont(font)

        self.io_side = QHBoxLayout()  # Main layout for media and controls
        self.ui_side = QVBoxLayout()  # For buttons and dropdowns

                # Load Source Folder button
        self.load_button = QPushButton("Open Source Folder")
        self.load_button.clicked.connect(self.open_folder)
        self.load_button.setFixedHeight(50)  # Make the button larger
        self.load_button.setFixedWidth(200)

                # Original Media label
        self.og_label = QLabel("Original")
        self.og_label.setAlignment(QtCore.Qt.AlignCenter)  # Center the text
        self.og_label.setFixedSize(720, 720)  # Set fixed size of 720x720px

        # Processed Media label
        self.processed_label = QLabel("Output")
        self.processed_label.setAlignment(QtCore.Qt.AlignCenter)  # Center the text
        self.processed_label.setFixedSize(720, 720)  # Set fixed size of 720x720px

        self.io_layout = QHBoxLayout()  # Use horizontal layout for side-by-side
        self.io_layout.addWidget(self.og_label)
        self.io_layout.addWidget(self.processed_label)