import cv2
import sys
import os
import logging
import cx22 # type: ignore
import numpy as np
import re
#import 
from yt_dlp import YoutubeDL
from ultralytics import YOLO
import pyqtgraph as pg
from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QSpinBox,
    QCheckBox,
    QSlider
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal

import time

logging.getLogger('ultralytics').setLevel(logging.WARNING)
yolov8m = YOLO('yolov8m.pt')

class VideoHandler(QThread):
    sendFrame = pyqtSignal(np.ndarray)

    def __init__(self, win):
        super().__init__()
        self.window = win
        self.isRunning = True

    def run(self):
        while(self.isRunning):
            frame = self.window.currentFrame
            frame = self.window.effects[self.window.currentEffect]() if self.window.effects else frame
            time.sleep(0.01)
            self.sendFrame.emit(frame)

    
    def stop(self):
        self.isRunning = False
        self.quit()
        self.wait()


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.showFullScreen()
        screen = QApplication.primaryScreen()
        self.screen_width = screen.size().width()
        self.screen_height = screen.size().height()
        
        # VARIABLES
        self.handler = VideoHandler(self)
        self.handler.sendFrame.connect(self.update_frame)
        self.current_media = None
        self.currentFrame = None
        self.currentOutFrame = None

        self.default_url = "https://www.youtube.com/live/z-kjpAVKvyo?si=Bhz39xU9YOHq54kP"

        self.paused = False
        self.cap = None
        self.frameDelay = int(1000/30)
        self.timer = QTimer()
        self.timer.timeout.connect(self.video_manage)
        self.option = None

        self.targetKeys = list(yolov8m.names.keys())
        self.targetNames = list(yolov8m.names.values())
        self.target = 0
        self.targetString = "person"
        self.pop = 0
        self.max_pop = 20
        
        self.plot_graph = pg.PlotWidget()
        self.plot_graph.setFixedHeight( int(self.screen_height * 0.3))
        self.plot_graph.setBackground("w")
        pen = pg.mkPen(color=(255, 0, 0))
        self.plot_graph.setTitle(f"# of {self.targetString}", color="b", size="20pt")
        styles = {"color": "red", "font-size": "18px"}
        self.plot_graph.setLabel("left", "#", **styles)
        self.plot_graph.setLabel("bottom", "secs", **styles)
        self.plot_graph.addLegend()
        self.plot_graph.showGrid(x=True, y=True)
        self.plot_graph.setYRange(0, self.max_pop)
        self.timeList = list(range(10))
        self.popList = [0] * 10

        self.line = self.plot_graph.plot(
            self.timeList,                              
            self.popList,
            name = "erm",
            pen = pen,
            symbol="+",
            symbolSize = 15,
            symbolBrush="b"
            )

        self.timerD = QtCore.QTimer()
        self.timerD.start()
        self.timerD.setInterval(1000)
        self.timerD.timeout.connect(self.update_plot)

        self.windowTitle = "yolov8 qtv2"
        self.setAcceptDrops(True)
        
        font = QFont()
        font.setPointSize(12)
        self.setFont(font)

        self.menu_bar = QHBoxLayout()
        self.options = QHBoxLayout()
        self.options.setAlignment(Qt.AlignCenter)
        self.io_bar = QHBoxLayout()
        self.video_controls = QHBoxLayout()
        self.graph_space = QHBoxLayout()

        self.load_button = QPushButton("Load")
        self.load_button.clicked.connect(self.load_file)
        self.load_button.setFixedHeight( int(self.screen_height * 0.05))
        self.load_button.setFixedWidth( int(self.screen_width * 0.1))

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_file)
        self.save_button.setFixedHeight( int(self.screen_height * 0.05))
        self.save_button.setFixedWidth( int(self.screen_width * 0.1))

        self.default_button = QPushButton("Default")

        self.url_bar = QLineEdit()
        self.url_bar.setPlaceholderText(self.default_url)
        self.url_bar.returnPressed.connect(self.open_url)
        self.url_bar.setFixedHeight( int(self.screen_height * 0.05))

        self.effect_label = QLabel("Effect")
        self.effect_label.setFixedHeight(int(self.screen_height * 0.05))

        self.effect_dropdown = QComboBox()
        self.effect_dropdown.currentIndexChanged.connect(self.set_effect)
        self.effect_dropdown.setFixedHeight(int(self.screen_height * 0.05))





        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.exit_qt)
        self.exit_button.setFixedHeight( int(self.screen_height * 0.05))
        self.exit_button.setFixedWidth( int(self.screen_width * 0.1))

        self.effect_dropdown.addItem("Grayscale")
        self.effect_dropdown.addItem("Color Quantize")
        self.effect_dropdown.addItem("Yolov8m")
        self.effect_dropdown.addItem("Sobel")

        self.menu_bar.addWidget(self.load_button, alignment=Qt.AlignTop)
        self.menu_bar.addWidget(self.save_button, alignment=Qt.AlignTop)
        self.menu_bar.addWidget(self.default_button, alignment=Qt.AlignTop)
        self.menu_bar.addWidget(self.url_bar, alignment=Qt.AlignTop)
        self.menu_bar.addWidget(self.effect_label, alignment=Qt.AlignTop)
        self.menu_bar.addWidget(self.effect_dropdown, alignment=Qt.AlignTop)
        self.menu_bar.addWidget(self.exit_button, alignment=Qt.AlignTop)

        self.og_label = QLabel("Original")
        self.og_label.setAlignment(QtCore.Qt.AlignCenter)
        self.og_label.setFixedHeight( int(self.screen_height * 0.5))

        self.processed_label = QLabel("Processed")
        self.processed_label.setAlignment(QtCore.Qt.AlignCenter)
        self.processed_label.setFixedHeight( int(self.screen_height * 0.5))

        self.io_bar.addWidget(self.og_label)
        self.io_bar.addWidget(self.processed_label)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.pause_button.setFixedHeight( int(self.screen_height * 0.05))
        self.pause_button.setFixedWidth( int(self.screen_width * 0.1))

        self.b_frame_button = QPushButton("Frame Back")
        self.b_frame_button.clicked.connect(self.b_frame)
        self.b_frame_button.setFixedHeight( int(self.screen_height * 0.05))
        self.b_frame_button.setFixedWidth( int(self.screen_width * 0.1))

        self.f_frame_button = QPushButton("Frame Forward")
        self.f_frame_button.clicked.connect(self.f_frame)
        self.f_frame_button.setFixedHeight( int(self.screen_height * 0.05))
        self.f_frame_button.setFixedWidth( int(self.screen_width * 0.1))

        self.speed_input = QSlider(self)
        self.speed_input.setOrientation(Qt.Horizontal)
        self.speed_input.setMinimum(5)
        self.speed_input.setMaximum(80)
        self.speed_input.setValue(30)
        self.speed_input.setTickPosition(QSlider.TicksBelow)
        self.speed_input.setTickInterval(5)
        self.speed_input.valueChanged.connect(self.set_speed)
        self.speed_input.setFixedWidth(int(self.screen_width * 0.3))

        self.speed_label = QLabel("FPS: " + str(self.speed_input.value()), self)
        self.speed_label.setFixedWidth( int(self.screen_width * 0.04))

        self.video_controls.addWidget(self.pause_button)
        self.video_controls.addWidget(self.b_frame_button)
        self.video_controls.addWidget(self.f_frame_button)
        self.video_controls.addWidget(self.speed_label)
        self.video_controls.addWidget(self.speed_input)

        self.video_controls.setAlignment(Qt.AlignCenter)

        self.graph_space.addWidget(self.plot_graph, alignment=Qt.AlignBottom)

        self.main_layout = QVBoxLayout()
        self.main_layout.addLayout(self.menu_bar)
        self.main_layout.addLayout(self.options)
        self.main_layout.addLayout(self.io_bar)
        self.main_layout.addLayout(self.video_controls)
        self.main_layout.addLayout(self.graph_space)

        container = QWidget()
        container.setLayout(self.main_layout)
        self.setCentralWidget(container)

        self.effects = {
            "Grayscale": self.apply_grayscale,
            "Yolov8m": self.apply_yolov8m,
            "Color Quantize": self.apply_quantize,
            "Sobel" : self.sobel,
        }

        self.currentEffect = self.effect_dropdown.currentText()
        self.divisor = 50

        self.blur = 3
        self.contrast_threshold = 80
        self.on_black = False
        self.dvh = 0
        self.y_percentage = 0

        self.currentFileName = ""
        self.currentFileExt = ""

        self.default_button.clicked.connect(self.load_default)
        self.default_button.setFixedHeight( int(self.screen_height * 0.05))
        self.default_button.setFixedWidth( int(self.screen_width * 0.1))
    

    def update_plot(self):
        self.timeList = self.timeList[1:]
        self.timeList.append(self.timeList[-1] + 1)
        self.popList = self.popList[1:]
        self.popList.append(self.pop)
        self.line.setData(self.timeList, self.popList)

    def show_graph(self):
        self.plot_graph.show()
        self.timerD.start()

    def close_graph(self):
        self.plot_graph.hide()
        self.timerD.stop()
        self.timeList = [0] * 10
        self.popList = [0] * 10

    def update_frame(self, frame):
        self.currentOutFrame = frame
        self.display(self.currentOutFrame, self.processed_label)

    def video_manage(self):
        ret, frame = self.cap.read()
        if ret:
            self.currentFrame = frame
            self.display(self.currentFrame, self.og_label)
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def load_default(self):
        self.url_bar.setText(self.default_url)
        self.open_url()
    
    def open_url(self):
        if "youtube.com" in self.url_bar.text() or "youtu.be/" in self.url_bar.text():
            ydl_opts = {
                'format': 'best[ext=mp4]',
                'quiet': True,
            }
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.url_bar.text(), download=False)
                video_url = info['url']
                self.cap = cv2.VideoCapture(video_url)
                video_title = info.get('title', 'Unknown Title')
                illegals = r'[<>:"/\\|?*\0]'
                temp = re.sub(illegals, '-', video_title)
                self.currentFileName = temp.strip().strip('.')
        else:
            self.currentFileName = "urlvideo"
            self.cap = cv2.VideoCapture(self.url_bar.text())
        if self.cap.isOpened:
            self.current_media = "vid"
            self.paused = False
            self.pause_button.setText("Pause")

            self.timer.start(self.frameDelay)
            if self.currentEffect == "Yolov8m":
                self.close_graph()
                self.show_graph()
            
            self.video_manage()
            self.handler.start()
        else:
            print("Whopps! Couldn't open video")

    def load_file(self):
        filePath, _ = QFileDialog.getOpenFileName(None, "Open File", "", "All Files (*);;Images (*.png *.jpg *.jpeg);;Videos (*.mp4 *.avi *.mov *.mkv)")
        if filePath:
            if filePath.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG')):
                self.currentFileExt = os.path.splitext(filePath)[1]
                self.currentFileName = os.path.basename(filePath).split('.')[0]
                self.current_media = "img"

                self.timer.stop()
                self.close_graph()
                self.cap = None
                self.currentFrame = cv2.imread(filePath)
                self.convert_image()
            if filePath.lower().endswith(('.mp4', '.mov', '.mkv', '.avi')):
                self.cap = cv2.VideoCapture(filePath)
                self.currentFile = filePath
                if self.cap.isOpened:
                    self.currentFileExt = os.path.splitext(filePath)[1]
                    self.currentFileName = os.path.basename(filePath).split('.')[0]
                    self.current_media = "vid"
                    self.paused = False
                    self.pause_button.setText("Pause")
                    self.timer.start(self.frameDelay)
                    if self.currentEffect == "Yolov8m":
                        self.close_graph()
                        self.show_graph()
                    self.video_manage()
                    self.handler.start()
                else:
                    print("Whopps! Couldn't open video")

    def save_file(self):
        filenames = os.listdir()
        temp = 0
        if self.current_media == "vid":
            for filename in filenames:
                if self.currentFileName in filename and ".png" in filename:
                    temp = temp + 1
            cv2.imwrite(self.currentFileName + "_" + str(temp) + ".png", self.currentOutFrame)
        else:
            for filename in filenames:
                if self.currentFileName in filename and self.currentFileExt in filename:
                    temp = temp + 1
            cv2.imwrite(self.currentFileName + "_" + str(temp) + self.currentFileExt, self.currentOutFrame)
    
    def exit_qt(self):
        self.handler.stop()
        self.close()

    def convert_image(self):
        self.currentOutFrame = self.effects[self.currentEffect]()
        self.display(self.currentFrame, self.og_label)
        self.display(self.currentOutFrame, self.processed_label)
    
    def display(self, frame, label):
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytes_per_line = channel * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Scale to fit the label and maintain its aspect ratio
            pixmap = QPixmap.fromImage(q_image)

            scaled_pixmap = pixmap.scaled(int(self.screen_width * 0.5), int(self.screen_height * 0.5), QtCore.Qt.KeepAspectRatio)
            label.setPixmap(scaled_pixmap)

        else:
            print("whoooppa!!")

    def clear_option_layout(self):
        for i in range(self.options.count()):
            item = self.options.takeAt(0)  # Remove the item at index 0
            widget = item.widget()
            if widget:
                widget.deleteLater()

    
    def set_effect(self):
        self.currentEffect = self.effect_dropdown.currentText()
        ''' try:
            self.options_textbar.returnPressed.disconnect()
        except TypeError:
            pass'''

        if (self.currentEffect == "Color Quantize"):
            self.clear_option_layout()

            self.qdivisor_input = QSlider(self)
            self.qdivisor_input.setOrientation(Qt.Horizontal)
            self.qdivisor_input.setMinimum(1)
            self.qdivisor_input.setMaximum(255)
            self.qdivisor_input.setValue(50)
            self.qdivisor_input.setTickPosition(QSlider.TicksBelow)
            self.qdivisor_input.setTickInterval(5)
            self.qdivisor_input.valueChanged.connect(self.set_divisor)
            self.qdivisor_input.setFixedWidth(int(self.screen_width * 0.4))

            self.options_label = QLabel("Divisor: " + str(self.qdivisor_input.value()))
            self.options_label.setFixedWidth( int(self.screen_width * 0.1))

            self.options.addWidget(self.options_label)
            self.options.addWidget(self.qdivisor_input)

        elif (self.currentEffect == "Yolov8m"):
            self.clear_option_layout()
            self.options_label = QLabel("Target")
            self.options_textbar = QLineEdit()
            self.options_textbar.setPlaceholderText("person")
            self.options_textbar.returnPressed.connect(self.set_target)

            self.options.addWidget(self.options_label)
            self.options.addWidget(self.options_textbar)

        elif (self.currentEffect == "Sobel"):

            self.clear_option_layout()
            self.blur_label = QLabel("Blur: ")
            self.ct_label = QLabel("Threshold: ")
            self.yp_label = QLabel("Cut Top%: ")

            self.blur_label.setFixedWidth( int(self.screen_width * 0.03))
            self.ct_label.setFixedWidth( int(self.screen_width * 0.05))
            self.yp_label.setFixedWidth( int(self.screen_width * 0.05))

            self.blur_input = QSpinBox()
            self.blur_input.setMinimum(1)
            self.blur_input.setMaximum(31)
            self.blur_input.setValue(5)
            self.blur_input.setSingleStep(2)
            self.blur_input.valueChanged.connect(self.set_blur)

            self.ct_input = QSlider(self)
            self.ct_input.setOrientation(Qt.Horizontal)
            self.ct_input.setMinimum(1)
            self.ct_input.setMaximum(500)
            self.ct_input.setValue(50)
            self.ct_input.setTickPosition(QSlider.TicksBelow)
            self.ct_input.setTickInterval(10)
            self.ct_input.valueChanged.connect(self.set_ct)
            self.ct_input.setFixedWidth(int(self.screen_width * 0.1))

            self.black_check = QCheckBox("On Black")
            self.black_check.stateChanged.connect(self.set_black)


            self.all_lines_check = QCheckBox("All Lines")
            self.all_lines_check.stateChanged.connect(self.set_all_lines)

            self.amap_check = QCheckBox("Map")
            self.amap_check.stateChanged.connect(self.set_map)

            self.sobel_h = QCheckBox("Horizontal")
            self.sobel_h.stateChanged.connect(self.set_lines)

            self.sobel_v = QCheckBox("Vertical")
            self.sobel_v.stateChanged.connect(self.set_lines)

            self.sobel_d = QCheckBox("Diagonal")
            self.sobel_d.stateChanged.connect(self.set_lines)

            self.blur_input.setFixedWidth( int(self.screen_width * 0.05))
            self.black_check.setFixedWidth( int(self.screen_width * 0.05))
            self.all_lines_check.setFixedWidth( int(self.screen_width * 0.05))
            self.amap_check.setFixedWidth( int(self.screen_width * 0.03))
            self.sobel_h.setFixedWidth( int(self.screen_width * 0.06))
            self.sobel_v.setFixedWidth( int(self.screen_width * 0.05))
            self.sobel_d.setFixedWidth( int(self.screen_width * 0.05))

            self.yp_input = QSlider(self)
            self.yp_input.setOrientation(Qt.Horizontal)
            self.yp_input.setMinimum(0)
            self.yp_input.setMaximum(99)
            self.yp_input.setValue(0)
            self.yp_input.setTickPosition(QSlider.TicksBelow)
            self.yp_input.setTickInterval(5)
            self.yp_input.valueChanged.connect(self.set_yp)
            self.yp_input.setFixedWidth(int(self.screen_width * 0.1))

            self.options.addWidget(self.blur_label)
            self.options.addWidget(self.blur_input)
            self.options.addWidget(self.ct_label)
            self.options.addWidget(self.ct_input)
            self.options.addWidget(self.black_check)
            self.options.addWidget(self.all_lines_check)
            self.options.addWidget(self.amap_check)
            self.options.addWidget(self.sobel_h)
            self.options.addWidget(self.sobel_v)
            self.options.addWidget(self.sobel_d)
            self.options.addWidget(self.yp_label)
            self.options.addWidget(self.yp_input)
            

        if self.currentEffect == "Yolov8m" and self.current_media == "vid":
            self.show_graph()
        else:
            self.close_graph()

        #if self.current_media == "img" and self.currentFrame != None:
        if self.current_media == "img":
            self.convert_image()
        if self.current_media == "vid" and self.paused:
            self.convert_image()

    def set_blur(self):
        self.blur = self.blur_input.value()
        self.convert_image()
    
    def set_ct(self):
        self.contrast_threshold = self.ct_input.value()
        self.convert_image()
    
    def set_black(self, state):
        if state == 2:
            self.on_black = True
        else:
            self.on_black = False
        self.convert_image()
    
    def set_all_lines(self, state):
        if state == 2:
            self.amap_check.setChecked(False)
            self.sobel_h.setChecked(False)
            self.sobel_v.setChecked(False)
            self.sobel_d.setChecked(False)
            self.dvh = 0
        self.convert_image()
    
    def set_map(self, state):
        if state == 2:
            self.all_lines_check.setChecked(False)
            self.sobel_h.setChecked(False)
            self.sobel_v.setChecked(False)
            self.sobel_d.setChecked(False)
            self.dvh = 1
        self.convert_image()
    
    def set_lines(self):
        if self.sobel_h.isChecked() or self.sobel_v.isChecked() or self.sobel_d.isChecked():
            self.amap_check.setChecked(False)
            self.all_lines_check.setChecked(False)
            temp = 1
            if self.sobel_h.isChecked():
                temp = temp * 2
            if self.sobel_v.isChecked():
                temp = temp * 3
            if self.sobel_d.isChecked():
                temp = temp * 5
            self.dvh = temp
        else:
            self.dvh = 7
        self.convert_image()
    
    def set_yp(self):
        self.y_percentage = (self.yp_input.value() / 100)
        self.convert_image()


    
    def set_target(self):
        self.targetString = self.options_bar.text()
        if self.targetString in self.targetNames:
            self.target = self.targetKeys[self.targetNames.index(self.targetString)]
            self.plot_graph.setTitle(f"# of {self.targetString}", color="b", size="20pt")
            self.convert_image()
        
    def set_divisor(self):
        self.options_label.setText("FPS: " + str(self.qdivisor_input.value()))  
        self.divisor = self.speed_input.value()
        self.convert_image()

    def toggle_pause(self):
        if self.current_media == "vid":
            if self.paused:
                self.timer.start(self.frameDelay)
                self.timerD.start()
                self.paused = False
                self.pause_button.setText("Pause")
            else:
                self.timer.stop()
                self.timerD.stop()
                self.paused = True
                self.pause_button.setText("Play")
            
    def b_frame(self):
        if self.current_media == "vid":
            framePos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, (framePos - 2))
            ret, self.currentFrame = self.cap.read()
            if ret:
                self.convert_image()
    
    def f_frame(self):
        if self.current_media == "vid":
            framePos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, (framePos))
            ret, self.currentFrame = self.cap.read()
            if ret:
                self.convert_image()
    
    def set_speed(self):
        if self.current_media == "vid":
            self.frameDelay = int(1000/self.speed_input.value())
            self.timer.stop()
            self.timer.start(self.frameDelay)
            #self.paused = False
            #self.pause_button.setText("Pause")   
            self.speed_label.setText("FPS: " + str(self.speed_input.value()))   
    
    def apply_quantize(self):
        return cx22.quantize(self.currentFrame, self.divisor)
    
    def sobel(self):
        man = self.currentFrame.copy() 
        return cx22.sobel_filter(man, self.blur, self.contrast_threshold, self.on_black, self.dvh, self.y_percentage)
    

    def apply_yolov8m(self):
        outframe = self.currentFrame.copy()
        output = yolov8m(outframe)
        boxes = output[0].boxes.xyxy.cpu().numpy()
        class_ids = output[0].boxes.cls.cpu().numpy()  # Class IDs
        confidences = output[0].boxes.conf.cpu().numpy()
        
        # Filter for target
        targetBoxes = boxes[class_ids == self.target]

        self.pop = len(targetBoxes)

        if self.pop > int(0.8 * self.max_pop):
            self.max_pop = self.max_pop * 1.25
            self.plot_graph.setYRange(0, self.max_pop)

        ypos = int(outframe.shape[0] / 10)
        scale = (outframe.shape[1] / 1000) + 1
        width = int(scale * 2)

        # Draw boxes for every person found with colors based on confidence, blue = higher
        for i, box in enumerate(targetBoxes):
            x1, y1, x2, y2 = map(int, box)
            confidence = confidences[i]
            color = (int(255 * confidence), 0, int(255 * (1 - confidence)))
            cv2.rectangle(outframe, (x1, y1), (x2, y2), color, 2)
        
        cv2.putText(outframe, f"Found: {self.pop}", (20, ypos), cv2.FONT_HERSHEY_SIMPLEX, 
                    scale, (0, 0, 0), int(width * 2))    
        cv2.putText(outframe, f"Found: {self.pop}", (20, ypos), cv2.FONT_HERSHEY_SIMPLEX, 
                            scale, (255, 255, 255), width)
                
        return outframe

    def apply_grayscale(self):
        return cv2.cvtColor(self.currentFrame, cv2.COLOR_BGR2GRAY)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = Window()
    myWindow.show()
    sys.exit(app.exec_())
