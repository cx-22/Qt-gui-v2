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
    QSpinBox
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import QTimer, Qt

logging.getLogger('ultralytics').setLevel(logging.WARNING)
yolov8m = YOLO('yolov8m.pt')

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        # VARIABLES
        self.current_media = None
        self.currentFrame = None
        self.currentOutFrame = None

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
        
        self.plot_graph = pg.PlotWidget()
        self.plot_graph.setMinimumHeight(300)
        self.plot_graph.setBackground("w")
        pen = pg.mkPen(color=(255, 0, 0))
        self.plot_graph.setTitle(f"# of {self.targetString}", color="b", size="20pt")
        styles = {"color": "red", "font-size": "18px"}
        self.plot_graph.setLabel("left", "#", **styles)
        self.plot_graph.setLabel("bottom", "secs", **styles)
        self.plot_graph.addLegend()
        self.plot_graph.showGrid(x=True, y=True)
        self.plot_graph.setYRange(0, 30)
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
        self.io_bar = QHBoxLayout()
        self.video_controls = QHBoxLayout()
        self.graph_space = QHBoxLayout()

        self.menu_bar.setAlignment(QtCore.Qt.AlignCenter)
        self.video_controls.setAlignment(QtCore.Qt.AlignCenter)

        self.load_button = QPushButton("Load")
        self.load_button.clicked.connect(self.load_file)
        self.load_button.setFixedHeight = 50
        self.load_button.setFixedWidth = 50

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_file)
        self.save_button.setFixedHeight = 50
        self.save_button.setFixedWidth = 50
        
        self.url_bar = QLineEdit()
        self.url_bar.setPlaceholderText("https://youtu.be/Hu0XosgxCyU?si=2_PF8mB67z6NiYDY")
        self.url_bar.returnPressed.connect(self.open_url)

        self.options_label = QLabel("")
        self.options_label.setFixedHeight(50)

        self.options_bar = QLineEdit()
        self.options_bar.setPlaceholderText("")

        self.effect_label = QLabel("Effect")
        self.effect_label.setFixedHeight(50)

        self.effect_dropdown = QComboBox()
        self.effect_dropdown.currentIndexChanged.connect(self.set_effect)
        self.effect_dropdown.setFixedHeight(50)

        self.effect_dropdown.addItem("Grayscale")
        self.effect_dropdown.addItem("Color Quantize")
        self.effect_dropdown.addItem("Scale Up")
        self.effect_dropdown.addItem("Yolov8m")
        self.effect_dropdown.addItem("Sobel")

        self.menu_bar.addWidget(self.load_button)
        self.menu_bar.addWidget(self.save_button)
        self.menu_bar.addWidget(self.url_bar)
        self.menu_bar.addWidget(self.effect_label)
        self.menu_bar.addWidget(self.effect_dropdown)
        self.menu_bar.addWidget(self.options_label)
        self.menu_bar.addWidget(self.options_bar)

        self.og_label = QLabel("Original")
        self.og_label.setAlignment(QtCore.Qt.AlignCenter)

        self.processed_label = QLabel("Processed")
        self.processed_label.setAlignment(QtCore.Qt.AlignCenter)

        self.io_bar.addWidget(self.og_label)
        self.io_bar.addWidget(self.processed_label)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.pause_button.setFixedHeight = 50
        self.pause_button.setFixedWidth = 50

        self.b_frame_button = QPushButton("Frame Back")
        self.b_frame_button.clicked.connect(self.b_frame)
        self.b_frame_button.setFixedHeight = 50
        self.b_frame_button.setFixedWidth = 50

        self.f_frame_button = QPushButton("Frame Forward")
        self.f_frame_button.clicked.connect(self.f_frame)
        self.f_frame_button.setFixedHeight = 50
        self.f_frame_button.setFixedWidth = 50

        self.speed_label = QLabel("FPS")

        self.speed_input = QSpinBox()
        self.speed_input.setMinimum(5)
        self.speed_input.setMaximum(80)
        self.speed_input.setValue(30)
        self.speed_input.setSingleStep(5)
        self.speed_input.valueChanged.connect(self.set_speed)

        self.video_controls.addWidget(self.pause_button)
        self.video_controls.addWidget(self.b_frame_button)
        self.video_controls.addWidget(self.f_frame_button)
        self.video_controls.addWidget(self.speed_label)
        self.video_controls.addWidget(self.speed_input)

        self.graph_space.addWidget(self.plot_graph)

        self.main_layout = QVBoxLayout()
        self.main_layout.addLayout(self.menu_bar)
        self.main_layout.addLayout(self.io_bar)
        self.main_layout.addLayout(self.video_controls)
        self.main_layout.addLayout(self.graph_space)

        container = QWidget()
        container.setLayout(self.main_layout)
        self.setCentralWidget(container)

        self.effects = {
            "Grayscale": self.apply_grayscale,
            "Yolov8m": self.apply_yolov8m,
            "Scale Up": self.apply_scale_up,
            "Color Quantize": self.apply_quantize,
            "Sobel" : self.sobel,
        }

        self.currentEffect = self.effect_dropdown.currentText()
        self.scaleX = 2
        self.scaleY = 2
        self.divisor = 50

        self.blur = 3
        self.contrast_threshold = 80
        self.on_black = False
        self.dvh = 0
        self.y_percentage = 0

        self.currentFileName = ""
        self.currentFileExt = ""
        
    def video_manage(self):
        ret, frame = self.cap.read()
        if ret:
            self.currentFrame = frame
            self.currentOutFrame = self.effects[self.currentEffect]()
            self.display(self.currentFrame, self.og_label)
            self.display(self.currentOutFrame, self.processed_label)
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

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
    
    def open_url(self):
        if "youtube.com" in self.url_bar.text() or "youtu.be/":
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
            self.timer.start(self.frameDelay)
            if self.currentEffect == "Yolov8m":
                self.close_graph()
                self.show_graph()
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
                    self.timer.start(self.frameDelay)
                    if self.currentEffect == "Yolov8m":
                        self.close_graph()
                        self.show_graph()
                else:
                    print("Whopps! Couldn't open video")

    def save_file(self):
        filenames = os.listdir()
        temp = 0
        #print("self.currentFileName is " + self.currentFileName)
        #print("self.currentFileExt is " + self.currentFileExt)
        if self.current_media == "vid":
            for filename in filenames:
                if self.currentFileName in filename and ".png" in filename:
                    temp = temp + 1
            #print("this is a video, save ?")
            cv2.imwrite(self.currentFileName + "_" + str(temp) + ".png", self.currentOutFrame)
        else:
            for filename in filenames:
                if self.currentFileName in filename and self.currentFileExt in filename:
                    temp = temp + 1
                    #print("this passed!! " + filename)
            #print("this is an image, save ?")
            cv2.imwrite(self.currentFileName + "_" + str(temp) + self.currentFileExt, self.currentOutFrame)


    
    def convert_image(self):
        self.currentOutFrame = self.effects[self.currentEffect]()
        self.display(self.currentFrame, self.og_label)
        self.display(self.currentOutFrame, self.processed_label)
        self.adjustSize()
    
    def display(self, frame, label):
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytes_per_line = channel * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Scale to fit the label and maintain its aspect ratio
            pixmap = QPixmap.fromImage(q_image)
            if self.currentEffect != "Scale Up":
                scaled_pixmap = pixmap.scaled(720, 720, QtCore.Qt.KeepAspectRatio)
                label.setPixmap(scaled_pixmap)
            else:
                label.setPixmap(pixmap)
        else:
            print("whoooppa!!")
    
    def set_effect(self):
        self.currentEffect = self.effect_dropdown.currentText()
        try:
            self.options_bar.returnPressed.disconnect()
        except TypeError:
            pass

        if (self.currentEffect == "Scale Up"):
            self.options_label.setText("Scale X and Y")
            self.options_bar.setPlaceholderText("2, 2")
            self.options_bar.returnPressed.connect(self.set_scale)
        elif (self.currentEffect == "Color Quantize"):
            self.options_label.setText("Divisor")
            self.options_bar.setPlaceholderText("50")
            self.options_bar.returnPressed.connect(self.set_divisor)
        elif (self.currentEffect == "Yolov8m"):
            self.options_label.setText("Target")
            self.options_bar.setPlaceholderText("person")
            self.options_bar.returnPressed.connect(self.set_target)
        elif (self.currentEffect == "Sobel"):
            self.options_label.setText("blur, ct, onblack, dvh, yp dvh: 0 = all, 1 am, /2 d, /3 v, /5 h")
            self.options_bar.setPlaceholderText("3, 80, false, 0, 0")
            self.options_bar.returnPressed.connect(self.set_sobel)

        if self.currentEffect == "Yolov8m" and self.current_media == "vid" and self.paused == False:
            self.show_graph()
        else:
            self.close_graph()

        #if self.current_media == "img" and self.currentFrame != None:
        if self.current_media == "img":
            self.convert_image()
        if self.current_media == "vid" and self.paused:
            self.convert_image()

    def set_sobel(self):
        values = self.options_bar.text().split(", ")
        if (int(values[0]) % 2 == 1):
            self.blur = int(values[0])  
        self.contrast_threshold = int(values[1])
        self.on_black = values[2].lower() == "true"
        self.dvh = int(values[3])
        if float(values[4]) < 1 :
            self.y_percentage = float(values[4])
        self.convert_image()
    
    def set_target(self):
        self.targetString = self.options_bar.text()
        self.plot_graph.setTitle(f"# of {self.targetString}", color="b", size="20pt")
        self.target = self.targetKeys[self.targetNames.index(self.targetString)]
        self.convert_image()
    
    def set_scale(self):

        values = self.options_bar.text().split(", ")
        
        self.scaleX = int(values[0])  
        self.scaleY = int(values[1])

        if (self.scaleX > 5):
            self.scaleX = 1
        if (self.scaleY > 5):
            self.scaleY = 1
        self.convert_image()
        
    def set_divisor(self):
        self.divisor = int(self.options_bar.text())
        if (self.divisor > 0):
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
            self.paused = False
            self.pause_button.setText("Pause")      

    def apply_scale_up(self):
        return cx22.resizeLarger(self.currentFrame, self.scaleX, self.scaleY)
    
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
