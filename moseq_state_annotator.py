# a GUI asks for human annotation on videos (frame-wise)
# 1. video
# 2. moseq result file
# and returns:
# new result file containing your annotation

import sys
import cv2
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QSlider,
    QFileDialog, QHBoxLayout, QVBoxLayout, QComboBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap


class MoSeqAnnotator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MoSeq State Annotator")

        # Data holders
        self.video_path = None
        self.annotation = None
        self.df = None
        self.frame_count = 0
        self.current_frame_idx = 0
        self.cap = None
        self.state_map = {
            "turn": 0,
            "forward": 1,
            "still":2,
            "explore":3,
            "rear":4,
            "groom":5,
            "unsigned":-1
        }
        # UI elements
        self.video_label = QLabel("Load a video to begin.")
        self.video_label.setAlignment(Qt.AlignCenter)

        self.load_video_btn = QPushButton("Load Video")
        self.load_ann_btn = QPushButton("Load MoSeq Annotation")
        self.save_ann_btn = QPushButton("Save Updated Annotation")

        self.load_video_btn.clicked.connect(self.load_video)
        self.load_ann_btn.clicked.connect(self.load_annotation)
        self.save_ann_btn.clicked.connect(self.save_annotation)

        # Playback timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

        # Slider for frame scrub
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.valueChanged.connect(self.go_to_frame)

        # State selector
        self.state_box = QComboBox()
        self.state_box.addItems(self.state_map.keys())  # default 50 states

        # Interval buttons
        self.mark_start_btn = QPushButton("Mark Start")
        self.mark_stop_btn = QPushButton("Mark Stop and Assign State")

        self.mark_start_btn.clicked.connect(self.mark_start)
        self.mark_stop_btn.clicked.connect(self.mark_stop)

        self.start_frame = None

        # Layout
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.load_video_btn)
        control_layout.addWidget(self.load_ann_btn)
        control_layout.addWidget(self.save_ann_btn)

        interval_layout = QHBoxLayout()
        interval_layout.addWidget(self.state_box)
        interval_layout.addWidget(self.mark_start_btn)
        interval_layout.addWidget(self.mark_stop_btn)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addWidget(self.frame_slider)
        main_layout.addLayout(control_layout)
        main_layout.addLayout(interval_layout)

        self.setLayout(main_layout)

    # ------------------------------------------------------------------
    # VIDEO LOGIC
    # ------------------------------------------------------------------

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "*.mp4 *.avi *.mov")
        if not path:
            return
        self.video_path = path
        self.cap = cv2.VideoCapture(path)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_slider.setMaximum(self.frame_count - 1)

        _, frame = self.cap.read()
        self.show_frame(frame)

    def show_frame(self, frame):
        if frame is None:
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_frame = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_frame))

    def next_frame(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            return
        self.current_frame_idx += 1
        self.frame_slider.setValue(self.current_frame_idx)
        self.show_frame(frame)

    def go_to_frame(self, idx):
        if self.cap is None:
            return

        self.current_frame_idx = idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if ret:
            self.show_frame(frame)

    # ------------------------------------------------------------------
    # ANNOTATION LOGIC
    # ------------------------------------------------------------------

    def load_annotation(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select MoSeq Annotation", "", "*.npy *.csv")
        if not path:
            return

        if path.endswith(".npy"):
            self.annotation = np.load(path)
        else:
            df = pd.read_csv(path)
            # user must specify column 'state' manually
            state_cols = [col for col in df.columns if 'state' in col.lower()]
            df['moseq_state'] = df[state_cols].idxmax(axis=1).apply(
                lambda x: state_cols.index(x) if x in state_cols else np.nan)
            self.df = df
            if "human_labeled_state" in df.columns:
                self.annotation = df['human_labeled_state'].to_numpy().copy()
            else:
                self.annotation = df['moseq_state'].to_numpy().copy()

        print("Loaded annotation of length:", len(self.annotation))

    def mark_start(self):
        self.start_frame = self.current_frame_idx
        print("Start frame =", self.start_frame)

    def mark_stop(self):
        if self.start_frame is None:
            print("You must mark a start frame first.")
            return
        stop_frame = self.current_frame_idx
        if stop_frame < self.start_frame:
            print("Stop must be after start.")
            return

        chosen_state = int(self.state_map[self.state_box.currentText()])
        print(f"Assigning state {chosen_state} to frames {self.start_frame}:{stop_frame}")

        self.annotation[self.start_frame:stop_frame + 1] = chosen_state
        self.start_frame = stop_frame + 1

    def save_annotation(self):
        if self.annotation is None:
            print("No annotation loaded.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save Updated Annotation", "", "*.npy *.csv")
        if not path:
            return
        self.df["human_labeled_state"]= self.annotation

        if path.endswith(".npy"):
            np.save(path, self.annotation)
        else:
            self.df.to_csv(path, index=False)

        print("Annotation saved.")


# ----------------------------------------------------------------------
# RUN APP
# ----------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MoSeqAnnotator()
    window.resize(1100, 800)
    window.show()
    sys.exit(app.exec_())