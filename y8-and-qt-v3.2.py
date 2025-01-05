import cv2
import sys
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import re
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
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal

import torch

logging.getLogger('ultralytics').setLevel(logging.WARNING)

yolov8s = YOLO('yolov8s.pt')
yolov8m = YOLO('yolov8m.pt')
yolov8l = YOLO('yolov8l.pt')
yolov5m = YOLO('yolov5m.pt')
#Find yolov7.pt yourself
#yolov7m = torch.hub.load("WongKinYiu/yolov7",f"/yolov7",trust_repo=True)
yolov11m = YOLO('yolo11m.pt')

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available. GPU will be used.")
    print("GPU Name:", torch.cuda.get_device_name(0))
    yolov8s.to('cuda')
    yolov8m.to('cuda')
    yolov8l.to('cuda')
    yolov5m.to('cuda')
    #yolov7m.to('cuda')
    yolov11m.to('cuda')
else:
    print("CUDA is not available. Using CPU.")


class Worker(QThread):
    def __init__(self, win, frame):
        super().__init__()
        self.window = win
        self.frame = frame

    def run(self):
            self.outframe = self.window.effects[self.window.currentEffect](self.frame)
            self.quit()

class DetachedWindow(QMainWindow):
    closed_signal = pyqtSignal()

    def __init__(self, lr):
        super().__init__()
        if lr == 0:
            self.setWindowTitle("Original")
        else:
            self.setWindowTitle("Processed")
        self.setGeometry(200, 200, 800, 600)

        stylesheet = """
                QPushButton {
                    padding: 3px;      /* Add padding around the text */
                    font-size: 25px;   /* Set font size */
                }
            """

        size_bar = QHBoxLayout()
        size_bar.setAlignment(Qt.AlignLeft)
        
        small_button = QPushButton("Smaller")
        small_button.clicked.connect(self.smaller)
        small_button.setStyleSheet(stylesheet)

        big_button = QPushButton("Bigger")
        big_button.clicked.connect(self.bigger)
        big_button.setStyleSheet(stylesheet)

        size_bar.addWidget(small_button, alignment=Qt.AlignLeft)
        size_bar.addWidget(big_button, alignment=Qt.AlignLeft)

        self.pixmap = None
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        main_layout = QVBoxLayout()
        main_layout.addLayout(size_bar)
        main_layout.addWidget(self.image_label)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def smaller(self):
        new_width = int(self.width() * 0.7)
        new_height = int(self.height() * 0.7)
        self.resize(new_width, new_height)
        if self.pixmap is not None:
            scaled_pixmap = self.pixmap.scaled(int(self.size().width() * 0.9), int(self.size().height() * 0.9), QtCore.Qt.KeepAspectRatio)           
            self.image_label.setPixmap(scaled_pixmap)

    def bigger(self):
        new_width = int(self.width() * 1.3)
        new_height = int(self.height() * 1.3)
        self.resize(new_width, new_height)
        if self.pixmap is not None:
            scaled_pixmap = self.pixmap.scaled(int(self.size().width() * 0.9), int(self.size().height() * 0.9), QtCore.Qt.KeepAspectRatio)           
            self.image_label.setPixmap(scaled_pixmap)


    def display(self, frame):
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytes_per_line = channel * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = self.pixmap.scaled(int(self.size().width() * 0.9), int(self.size().height() * 0.9), QtCore.Qt.KeepAspectRatio)           
            self.image_label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        self.closed_signal.emit()
        event.accept()



class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.showFullScreen()
        self.setAcceptDrops(True)

        screen = QApplication.primaryScreen()
        self.screen_width = screen.size().width()
        self.screen_height = screen.size().height()

        cmap = plt.get_cmap('rainbow')
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        gradient = np.repeat(gradient, 20, axis=0)
        cmap_legend = cmap(gradient)
        cmap_legend = (cmap_legend[:, :, :3] * 255).astype(np.uint8)
        cmap_height, cmap_width, cmap_channel = cmap_legend.shape
        cmap_bytes_per_line = cmap_channel * cmap_width
        cmap_qimage = QImage(cmap_legend.data, cmap_width, cmap_height, cmap_bytes_per_line, QImage.Format_RGB888)
        scaled_cmap_qimage = cmap_qimage.scaled(int(cmap_width / 2), cmap_height)
        self.cmap_pixmap = QPixmap.fromImage(scaled_cmap_qimage)

        self.workers = []
        self.frameGrab = True

        self.is_detached_left = False
        self.is_detached_right = False
        self.detach_left = None
        self.detach_right = None
        
        self.current_media = None
        self.currentFrame = None
        self.currentOutFrame = None

        self.fullscreen = True
        self.paused = True
        self.ogFPS = 30
        self.newFPS = 0
        self.counter = 0
        self.cap = None
        self.options = None
        self.video_url = None
        self.liveVideo = False

        self.timer = QTimer()
        self.timer.timeout.connect(self.video_manage)
        self.delay = int(1000 / self.ogFPS)

        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.find_fps)

        self.targetKeys = list(yolov8m.names.keys())
        self.targetNames = list(yolov8m.names.values())
        self.target = 0
        self.targetString = "person"
        self.pop = 0
        self.max_pop = 20

        self.stylesheet = """
                QLabel, QPushButton, QSpinBox, QComboBox, QLineEdit {
                    text-align: left;  /* Left justify text */
                    padding: 3px;      /* Add padding around the text */
                    font-size: 20px;   /* Set font size */
                }
            """
        self.currentEffect = "Yolov8m"
        self.pastEffect = ""
        self.firstCycle = True
        
        self.plot_graph = pg.PlotWidget()
        self.plot_graph.setBackground("w")
        pen = pg.mkPen(color=(0, 0, 0))
        self.plot_graph.setTitle(f"{self.currentEffect} Number of people found", color="b", size="20pt")
        styles = {"color": "blue", "font-size": "14pt"}
        self.plot_graph.setLabel("bottom", "seconds", **styles)
        self.plot_graph.addLegend()
        self.plot_graph.showGrid(x=True, y=True)
        self.plot_graph.setYRange(0, self.max_pop)
        self.timeList = list(range(10))
        self.popList = [0] * 10

        axis_font = pg.Qt.QtGui.QFont()
        axis_font.setPointSize(15)
        self.plot_graph.getAxis("left").setStyle(tickFont=axis_font)
        self.plot_graph.getAxis("bottom").setStyle(tickFont=axis_font)

        self.line = self.plot_graph.plot(
            self.timeList,                              
            self.popList,
            pen = pen,
            symbol="+",
            symbolSize = 15,
            symbolBrush="b"
            )

        self.timerD = QtCore.QTimer()
        self.timerD.start()
        self.timerD.setInterval(1000)
        self.timerD.timeout.connect(self.update_plot)

        self.windowTitle = "qtv3.1"
        self.setAcceptDrops(True)

        self.menu_bar = QHBoxLayout()
        self.options = QHBoxLayout()
        self.options.setAlignment(Qt.AlignCenter)
        self.detach_bar = QHBoxLayout()
        self.io_bar = QHBoxLayout()
        self.video_controls = QHBoxLayout()
        self.graph_space = QHBoxLayout()

        self.og_label = QLabel("Original")
        self.og_label.setStyleSheet(self.stylesheet)
        self.processed_label = QLabel("Processed")
        self.processed_label.setStyleSheet(self.stylesheet)

        self.file_dropdown = QComboBox()
        self.file_dropdown.setStyleSheet(self.stylesheet)

        self.file_dropdown.addItem("File")
        self.file_dropdown.addItem("Load Image or Video")
        self.file_dropdown.addItem("Save Output as Image")
        self.file_dropdown.currentIndexChanged.connect(self.set_file_menu)

        self.preset_label = QLabel("Preset Cameras")
        self.preset_label.setStyleSheet(self.stylesheet)

        self.preset_dropdown = QComboBox()
        self.preset_dropdown.setStyleSheet(self.stylesheet)

        self.preset_dropdown.addItem("Live Thailand GMT+7")
        self.preset_dropdown.addItem("Live Tokyo GMT+9")
        self.preset_dropdown.addItem("Live St. Petersburg GMT+3")
        self.preset_dropdown.addItem("Live Finland GMT+2")
        self.preset_dropdown.addItem("Live Dublin GMT")
        self.preset_dropdown.addItem("Live Texas GMT-6")    
        self.preset_dropdown.addItem("Live New Orleans GMT-6")
        self.preset_dropdown.addItem("Live California GMT-8")
        self.preset_dropdown.addItem("Prerecorded Nascar")
        self.preset_dropdown.addItem("")

        self.url_bar = QLineEdit() 
        self.url_bar.returnPressed.connect(self.open_url)
        self.url_bar.setPlaceholderText("https://youtu.be/jNQXAC9IVRw?si=sliH5ck690ZVjVUo")
        self.url_bar.setStyleSheet(self.stylesheet)

        self.effect_label = QLabel("Models")
        self.effect_label.setStyleSheet(self.stylesheet)

        self.effect_dropdown = QComboBox()
        self.effect_dropdown.currentIndexChanged.connect(self.set_effect)
        self.effect_dropdown.setStyleSheet(self.stylesheet)

        self.mini_button = QPushButton("Minimize")
        self.mini_button.clicked.connect(self.minimize)
        self.mini_button.setStyleSheet(self.stylesheet)

        self.fullscreen_button = QPushButton("Exit Fullscreen")
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
        self.fullscreen_button.setStyleSheet(self.stylesheet)

        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.exit_qt)
        self.exit_button.setStyleSheet(self.stylesheet)

        self.effect_dropdown.addItem("Yolov8m")
        self.effect_dropdown.addItem("Yolov8s")
        self.effect_dropdown.addItem("Yolov8l")
        self.effect_dropdown.addItem("Yolov7m")
        self.effect_dropdown.addItem("Yolov5m")
        self.effect_dropdown.addItem("Yolov11m")

        self.menu_bar.addWidget(self.file_dropdown, alignment=Qt.AlignTop)
        self.menu_bar.addWidget(self.preset_label, alignment=Qt.AlignTop)
        self.menu_bar.addWidget(self.preset_dropdown, alignment=Qt.AlignTop)
        self.menu_bar.addWidget(self.url_bar, alignment=Qt.AlignTop)
        self.menu_bar.addWidget(self.effect_label, alignment=Qt.AlignTop)
        self.menu_bar.addWidget(self.effect_dropdown, alignment=Qt.AlignTop)
        self.menu_bar.addWidget(self.mini_button, alignment=Qt.AlignTop)
        self.menu_bar.addWidget(self.fullscreen_button, alignment=Qt.AlignTop)
        self.menu_bar.addWidget(self.exit_button, alignment=Qt.AlignTop)

        self.detach_button = QPushButton("Detach Windows")
        self.detach_button.clicked.connect(self.toggle_detach)
        self.detach_button.setStyleSheet(self.stylesheet)    

        self.detach_bar.addWidget(self.detach_button, alignment=Qt.AlignLeft)

        self.og_label.setAlignment(QtCore.Qt.AlignCenter)
        self.processed_label.setAlignment(QtCore.Qt.AlignCenter)

        self.io_bar.addWidget(self.og_label)
        self.io_bar.addWidget(self.processed_label)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.pause_button.setStyleSheet(self.stylesheet)

        self.b_frame_button = QPushButton("Frame Back")
        self.b_frame_button.clicked.connect(self.b_frame)
        self.b_frame_button.setStyleSheet(self.stylesheet)

        self.f_frame_button = QPushButton("Frame Forward")
        self.f_frame_button.clicked.connect(self.f_frame)
        self.f_frame_button.setStyleSheet(self.stylesheet)

        self.live_button = QPushButton("Live Catch Up")
        self.live_button.clicked.connect(self.live_catchup)
        self.live_button.setStyleSheet(self.stylesheet)

        self.frame_reset_button = QPushButton("Frame Reset")
        self.frame_reset_button.clicked.connect(self.frame_reset)        
        self.frame_reset_button.setStyleSheet(self.stylesheet)

        self.fps_in = QSpinBox(self)
        self.fps_in.setRange(1, 120)
        self.fps_in.setValue(30)
        self.fps_in.setSingleStep(1)
        self.fps_in.valueChanged.connect(self.set_fps_in)
        self.fps_in.setStyleSheet(self.stylesheet)

        self.fps_in_label = QLabel("Input FPS: ")
        self.fps_in_label.setStyleSheet(self.stylesheet)

        self.fps_out_label = QLabel("Output FPS: " + str(self.newFPS))
        self.fps_out_label.setStyleSheet(self.stylesheet)

        self.fps_ratio_label = QLabel("FPS Out/In: ")
        self.fps_ratio_label.setStyleSheet(self.stylesheet)

        self.video_controls.addWidget(self.live_button)
        self.video_controls.addWidget(self.frame_reset_button)
        self.video_controls.addWidget(self.fps_in_label)
        self.video_controls.addWidget(self.fps_in)
        self.video_controls.addWidget(self.fps_out_label)
        self.video_controls.addWidget(self.fps_ratio_label)
        self.video_controls.addWidget(self.pause_button)
        self.video_controls.addWidget(self.b_frame_button)
        self.video_controls.addWidget(self.f_frame_button)

        self.video_controls.setAlignment(Qt.AlignCenter)

        self.graph_space.addWidget(self.plot_graph, alignment=Qt.AlignBottom)

        self.main_layout = QVBoxLayout()
        self.main_layout.addLayout(self.menu_bar)
        self.main_layout.addLayout(self.options)
        self.main_layout.addLayout(self.detach_bar)
        self.main_layout.addLayout(self.io_bar)
        self.main_layout.addLayout(self.video_controls)
        self.main_layout.addLayout(self.graph_space)

        container = QWidget()
        container.setLayout(self.main_layout)
        self.setCentralWidget(container)

        self.effects = {
            "Yolov8m":   self.apply_yolo,
            "Yolov8s":   self.apply_yolo,
            "Yolov8l":   self.apply_yolo,
            #"Yolov7m":   self.apply_yolov7m,
            "Yolov5m":   self.apply_yolo,
            "Yolov11m":  self.apply_yolo,
            "Grayscale": self.apply_grayscale,
        }

        self.presets = {
            "Live California GMT-8": "https://www.youtube.com/live/PtChZ0D7tkE?si=ylVs8s5BlIH2_aU6",
            "Live Texas GMT-6": "https://www.youtube.com/live/otX-buqqS6Q?si=ulxJadK_wuVYe1SB",
            "Live New Orleans GMT-6": "https://www.youtube.com/live/z-kjpAVKvyo?si=Bhz39xU9YOHq54kP",
            "Live Finland GMT+2": "https://www.youtube.com/live/Cp4RRAEgpeU?si=xXJ9tIJUD17qD9zt",
            "Live Dublin GMT": "https://www.youtube.com/live/u4UZ4UvZXrg?si=A5FSMhUJjX0gY7Yb",
            "Live St. Petersburg GMT+3": "https://www.youtube.com/live/h1wly909BYw?si=Boe9gLUcLcp6Za55",
            "Live Tokyo GMT+9": "https://www.youtube.com/live/DjdUEyjx8GM?si=-umo4EzSyDXDNkqd",
            "Live Thailand GMT+7": "https://www.youtube.com/live/Q71sLS8h9a4?si=bGNflTmuPwexNC2k ",
            "Prerecorded Nascar": "https://youtu.be/HU7wIi3VriY?si=IVr5EqfsOnMgm9Ya",
        }
        
        self.preset_dropdown.currentIndexChanged.connect(self.load_preset)

        self.currentEffect = self.effect_dropdown.currentText()
        self.currentFileName = ""
        self.currentFileExt = ""
        self.load_preset()

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

    def video_manage(self):
        ret, frame = self.cap.read()
        if ret:
            self.currentFrame = frame
            self.display(frame, self.og_label)
            if self.is_detached_left:
                self.detach_left.display(frame)
            if self.frameGrab:
                self.frameGrab = False
                new_worker = Worker(self, self.currentFrame)
                new_worker.finished.connect(self.on_worker_finished)
                self.workers.append(new_worker)
                new_worker.start()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def on_worker_finished(self):
        dead_worker = self.sender()
        self.frameGrab = True
        if self.current_media == "vid":
            self.display(dead_worker.outframe, self.processed_label)
            if self.is_detached_right:
                self.detach_right.display(dead_worker.outframe)
            self.newFPS += 1

        if dead_worker in self.workers:
            try:
                self.workers.remove(dead_worker)
            except ValueError:
                pass
        dead_worker.deleteLater() 

    def kill_workers(self):
        if len(self.workers) != 0:
            for worker in self.workers:
                worker.wait()
                worker.deleteLater()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        filePath = event.mimeData().urls()[0].toLocalFile()
        if filePath:
            if filePath.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG')):
                self.currentFileExt = os.path.splitext(filePath)[1]
                self.currentFileName = os.path.basename(filePath).split('.')[0]
                self.current_media = "img"
                self.timer.stop()
                self.fps_timer.stop()
                self.close_graph()
                self.cap = None
                frame = cv2.imread(filePath)
                if frame is not None:
                    self.url_bar.setText("")
                    self.preset_dropdown.setCurrentText("")
                    self.currentFrame = frame
                    self.convert_image()
            if filePath.lower().endswith(('.mp4', '.mov', '.mkv', '.avi')):
                self.cap = cv2.VideoCapture(filePath)
                self.currentFile = filePath
                if self.cap.isOpened:
                    self.kill_workers()
                    self.liveVideo = False
                    self.firstFPS = int((self.cap.get(cv2.CAP_PROP_FPS)))
                    self.fps_in.setValue(self.firstFPS)
                    self.currentFileExt = os.path.splitext(filePath)[1]
                    self.currentFileName = os.path.basename(filePath).split('.')[0]
                    self.current_media = "vid"
                    self.paused = False
                    self.pause_button.setText("Pause")
                    self.preset_dropdown.setCurrentText("")
                    self.url_bar.setText("")

                    self.counter = self.fps_in.value()
                    self.timer.start(self.delay)
                    self.fps_timer.start(1000)

                    if "Yolo" in self.currentEffect:
                        self.close_graph()
                        self.show_graph()
                        self.display(self.currentFrame, self.processed_label)
    
    def load_preset(self):
        if self.preset_dropdown.currentText() != "":
            self.preset_url = self.presets[self.preset_dropdown.currentText()]
            self.url_bar.setText(self.preset_url)
            temp_live = True

            ydl_opts = {
                'format': 'best[ext=mp4]',
                'quiet': True,
            }
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.url_bar.text(), download=False)
                if info.get('is_live', False):
                    temp_live = True
                else:
                    temp_live = False
                self.video_url = info['url']
                self.cap = cv2.VideoCapture(self.video_url)
                video_title = info.get('title', 'Unknown Title')
                illegals = r'[<>:"/\\|?*\0]'
                temp = re.sub(illegals, '-', video_title)
                self.currentFileName = temp.strip().strip('.')

            if self.cap.isOpened:
                self.kill_workers()
                self.liveVideo = temp_live
                self.firstFPS = int((self.cap.get(cv2.CAP_PROP_FPS)))
                self.fps_in.setValue(self.firstFPS)
                self.current_media = "vid"
                self.paused = False
                self.pause_button.setText("Pause")
                self.timer.start(self.delay)
                self.fps_timer.start(1000)

                if "Yolo" in self.currentEffect:
                    self.close_graph()
                    self.show_graph()
                    self.display(self.currentFrame, self.processed_label)
    
    def open_url(self):
        if self.url_bar.text() != "":
            temp_live = True
            if "youtube.com" in self.url_bar.text() or "youtu.be/" in self.url_bar.text():
                ydl_opts = {
                    'format': 'best[ext=mp4]',
                    'quiet': True,
                }
                with YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(self.url_bar.text(), download=False)
                    if info.get('is_live', False):
                        temp_live = True
                    else:
                        temp_live = False
                    self.video_url = info['url']
                    self.cap = cv2.VideoCapture(self.video_url)
                    video_title = info.get('title', 'Unknown Title')
                    illegals = r'[<>:"/\\|?*\0]'
                    temp = re.sub(illegals, '-', video_title)
                    self.currentFileName = temp.strip().strip('.')
            else:
                self.currentFileName = "urlvideo"
                self.cap = cv2.VideoCapture(self.url_bar.text())
            if self.cap.isOpened:
                self.kill_workers()
                self.liveVideo = temp_live
                self.firstFPS = int((self.cap.get(cv2.CAP_PROP_FPS)))
                self.fps_in.setValue(self.firstFPS)
                self.current_media = "vid"
                self.paused = False
                self.pause_button.setText("Pause")
                self.preset_dropdown.setCurrentText("")
                self.timer.start(self.delay)
                self.fps_timer.start(1000)

                if "Yolo" in self.currentEffect:
                    self.close_graph()
                    self.show_graph()
                    self.display(self.currentFrame, self.processed_label)

    def set_file_menu(self):
        if self.file_dropdown.currentText() == "Load Image or Video":
            self.load_file()
            self.file_dropdown.setCurrentText("File")
        if self.file_dropdown.currentText() == "Save Output as Image":
            self.save_file()
            self.file_dropdown.setCurrentText("File")

    def load_file(self):
        filePath, _ = QFileDialog.getOpenFileName(None, "Open File", "",
                                                       "All Files (*);;Images (*.png *.jpg *.jpeg);;Videos (*.mp4 *.avi *.mov *.mkv)")

        if filePath:
            if filePath.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG')):
                self.currentFileExt = os.path.splitext(filePath)[1]
                self.currentFileName = os.path.basename(filePath).split('.')[0]
                self.current_media = "img"
                self.timer.stop()
                self.fps_timer.stop()
                self.close_graph()
                self.cap = None
                frame = cv2.imread(filePath)
                if frame is not None:
                    self.url_bar.setText("")
                    self.preset_dropdown.setCurrentText("")
                    self.currentFrame = frame
                    self.convert_image()
            if filePath.lower().endswith(('.mp4', '.mov', '.mkv', '.avi')):
                self.cap = cv2.VideoCapture(filePath)
                self.currentFile = filePath
                if self.cap.isOpened:
                    self.kill_workers()
                    self.liveVideo = False
                    self.firstFPS = int((self.cap.get(cv2.CAP_PROP_FPS)))
                    self.fps_in.setValue(self.firstFPS)
                    self.currentFileExt = os.path.splitext(filePath)[1]
                    self.currentFileName = os.path.basename(filePath).split('.')[0]
                    self.current_media = "vid"
                    self.paused = False
                    self.pause_button.setText("Pause")
                    self.url_bar.setText("")
                    self.preset_dropdown.setCurrentText("")

                    self.timer.start(self.delay)
                    self.fps_timer.start(1000)

                    if "Yolo" in self.currentEffect:
                        self.close_graph()
                        self.show_graph()
                        self.display(self.currentFrame, self.processed_label)

    def save_file(self):
        if self.currentOutFrame is not None:
            save_path, _ = QFileDialog.getSaveFileName(
                None,
                "Save File",
                self.currentFileName + ".png" if self.current_media == "vid" else self.currentFileName + self.currentFileExt,
                "PNG Files (*.png);;All Files (*)" if self.current_media == "vid" else f"All Files (*{self.currentFileExt})"
            )

            if save_path:
                cv2.imwrite(save_path, self.currentOutFrame)
        
    def minimize(self):
        self.setWindowState(Qt.WindowMinimized)
        self.fullscreen = False
        self.fullscreen_button.setText("Enter Fullscreen")

    def toggle_fullscreen(self):
        if self.fullscreen:
            self.showNormal()
            self.fullscreen = False
            self.fullscreen_button.setText("Enter Fullscreen")
        else:
            self.showFullScreen()
            self.fullscreen = True
            self.fullscreen_button.setText("Exit Fullscreen")
    
    def exit_qt(self):
        self.kill_workers()
        self.close()
        QtCore.QCoreApplication.quit()
        sys.exit(0)

    def convert_image(self):
        if self.currentFrame is not None:
            self.fps_out_label.setText("Output FPS: 0")
            self.currentOutFrame = self.effects[self.currentEffect](self.currentFrame)
            self.display(self.currentFrame, self.og_label)
            self.display(self.currentOutFrame, self.processed_label)
            if self.is_detached_left:
                self.detach_left.display(self.currentFrame)
            if self.is_detached_right:
                self.detach_right.display(self.currentOutFrame)

    def toggle_detach(self):
        self.detach_left = DetachedWindow(0)
        self.detach_left.closed_signal.connect(self.on_detach_end_left)
        self.detach_left.show()

        self.detach_right = DetachedWindow(1)
        self.detach_right.closed_signal.connect(self.on_detach_end_right)
        self.detach_right.show()

        self.is_detached_left = True
        self.is_detached_right = True

        if self.current_media == "img":
            self.convert_image()

    def on_detach_end_left(self):
        self.is_detached_left = False

    def on_detach_end_right(self):
        self.is_detached_right = False

    def display(self, frame, label):
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytes_per_line = channel * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(q_image)
            if self.plot_graph.isVisible():
                scaled_pixmap = pixmap.scaled(int(self.size().width() * 0.48), int(self.size().height() * 0.5), QtCore.Qt.KeepAspectRatio)
            else:
                scaled_pixmap = pixmap.scaled(int(self.size().width() * 0.48), int(self.size().height() * 0.7), QtCore.Qt.KeepAspectRatio)
            label.setPixmap(scaled_pixmap)

    def clear_option_layout(self):
        try:
            for i in range(self.options.count()):
                item = self.options.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
        except AttributeError:
            pass
    
    def set_effect(self):
        if not self.firstCycle:
            self.pastEffect = self.currentEffect
        self.currentEffect = self.effect_dropdown.currentText()

        print("past: " + self.pastEffect)
        print("current: " + self.currentEffect)

        if ("Yolo" in self.currentEffect):
            if ("Yolo" in self.pastEffect):
                if self.targetString == "person":
                    self.plot_graph.setTitle(f"{self.currentEffect} Number of people found", color="b", size="20pt")
                elif self.targetString == "bus":
                    self.plot_graph.setTitle(f"{self.currentEffect} Number of buses found", color="b", size="20pt")
                else:
                    self.plot_graph.setTitle(f"{self.currentEffect} Number of {self.targetString}s found", color="b", size="20pt")
            else:
                self.clear_option_layout()
                self.options_label = QLabel("Target: ")
                self.options_label.setStyleSheet(self.stylesheet)
                self.options_dropdown = QComboBox()
                self.options_dropdown.setStyleSheet(self.stylesheet)

                self.options_dropdown.addItem("person")
                self.options_dropdown.addItem("car")
                self.options_dropdown.addItem("bus")
                self.options_dropdown.addItem("truck")
                self.options_dropdown.addItem("bicycle")
                self.options_dropdown.addItem("motorcycle")
                self.options_dropdown.currentIndexChanged.connect(self.set_target)

                self.cmap_left = QLabel("-")
                self.cmap_left.setStyleSheet(self.stylesheet)
                self.cmap_right = QLabel("+")
                self.cmap_right.setStyleSheet(self.stylesheet)
                self.cmap_key = QLabel("")

                self.cmap_key.setPixmap(self.cmap_pixmap)

                self.options.addWidget(self.options_label)
                self.options.addWidget(self.options_dropdown)
                self.options.addWidget(self.cmap_left)
                self.options.addWidget(self.cmap_key)
                self.options.addWidget(self.cmap_right)

        else:
            self.clear_option_layout()

        if "Yolo" in self.currentEffect and self.current_media == "vid":
            self.show_graph()
        else:
            self.close_graph()

        if self.current_media == "img":
            self.convert_image()
        if self.current_media == "vid" and self.paused:
            self.convert_image()

        self.firstCycle = False
    
    def set_target(self):
        if self.targetString != self.options_dropdown.currentText():
            self.targetString = self.options_dropdown.currentText()
            self.target = self.targetKeys[self.targetNames.index(self.targetString)]
            if self.targetString == "person":
                self.plot_graph.setTitle(f"{self.currentEffect} Number of people found", color="b", size="20pt")
            elif self.targetString == "bus":
                self.plot_graph.setTitle(f"{self.currentEffect} Number of buses found", color="b", size="20pt")
            else:
                self.plot_graph.setTitle(f"{self.currentEffect} Number of {self.targetString}s found", color="b", size="20pt")
            if self.current_media != "vid":
                self.convert_image()
            else:
                self.close_graph()
                self.show_graph()
        
    def toggle_pause(self):
        if self.current_media == "vid":
            if self.paused:
                self.timer.start()
                self.timerD.start()
                self.paused = False
                self.pause_button.setText("Pause")

            else:
                self.timer.stop()
                self.timerD.stop()
                self.paused = True
                self.pause_button.setText("Play")
            
    def b_frame(self):
        if self.current_media == "vid" and self.paused and not self.liveVideo:
            self.kill_workers()
            framePos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, (framePos - 2))
            ret, self.currentFrame = self.cap.read()
            if ret:
                self.convert_image()
    
    def f_frame(self):
        if self.current_media == "vid" and self.paused and not self.liveVideo:
            self.kill_workers()
            framePos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, (framePos))
            ret, self.currentFrame = self.cap.read()
            if ret:
                self.convert_image()
    
    def set_fps_in(self):
        self.ogFPS = self.fps_in.value()
        self.delay = int(1000 / self.ogFPS)
        if not self.paused:
            self.timer.stop()
        if not self.paused:
            self.timer.start(self.delay)
    
    def find_fps(self):
        if not self.paused:
            self.fps_out_label.setText("Output FPS: " + str(self.newFPS))
            self.fps_ratio_label.setText(f"FPS Out/In: {self.newFPS / self.ogFPS:.2f}")
            self.newFPS = 0

    def frame_reset(self):
        if self.current_media == "vid":
            self.firstFPS = int((self.cap.get(cv2.CAP_PROP_FPS)))
            self.fps_in.setValue(self.firstFPS)
        
    def live_catchup(self):
        if self.liveVideo:
            self.kill_workers()
            self.cap.release()
            self.cap = cv2.VideoCapture(self.video_url)
    
    def apply_yolo(self, frame):
        if frame is not None:
            outframe = frame.copy()
            if "Yolo" in self.currentEffect:
                output = yolov8m(outframe)
            if self.currentEffect == "Yolov8s":
                output = yolov8s(outframe)
            if self.currentEffect == "Yolov8l":
                output = yolov8l(outframe)
            if self.currentEffect == "Yolov5m":
                output = yolov5m(outframe)
            if self.currentEffect == "Yolov11m":
                output = yolov11m(outframe)
            boxes = output[0].boxes.xyxy.cpu().numpy()
            class_ids = output[0].boxes.cls.cpu().numpy()
            confidences = output[0].boxes.conf.cpu().numpy()
            
            targetBoxes = boxes[class_ids == self.target]

            self.pop = len(targetBoxes)

            if self.pop > int(0.8 * self.max_pop):
                self.max_pop = self.max_pop * 1.25
                self.plot_graph.setYRange(0, self.max_pop)

            xpos = int(outframe.shape[1] * 0.8)
            ypos = int(outframe.shape[0] - 40)
            #scale = (outframe.shape[1] / 1000) + 1
            scale = outframe.shape[1] * 0.00078125
            #print("Width: " + str(outframe.shape[0]) + " Heiht" + str(outframe.shape[1]))
            thickness =  1 if outframe.shape[1] < 720 else 4

            for i, box in enumerate(targetBoxes):
                x1, y1, x2, y2 = map(int, box)
                confidence = confidences[i]
                rgba_color = plt.get_cmap('rainbow')(confidence)
                color = (int(rgba_color[2] * 255), int(rgba_color[1] * 255), int(rgba_color[0] * 255))
                cv2.rectangle(outframe, (x1, y1), (x2, y2), color, thickness)
            
            cv2.putText(outframe, f"Found: {self.pop}", (xpos, ypos), cv2.FONT_HERSHEY_SIMPLEX, 
                        scale, (0, 0, 0), thickness + 2)    
            cv2.putText(outframe, f"Found: {self.pop}", (xpos, ypos), cv2.FONT_HERSHEY_SIMPLEX, 
                                scale, (255, 255, 255), thickness)
            return outframe
    

    def apply_grayscale(self):
        return cv2.cvtColor(self.currentFrame, cv2.COLOR_BGR2GRAY)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = Window()
    myWindow.show()
    sys.exit(app.exec_())