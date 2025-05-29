import sys
import time
import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime
import csv # +++ TAMBAHAN: Untuk menyimpan data ke CSV

# Mengimpor modul logika yang sudah kita buat
import signal_proccesing as vp # Pastikan nama file ini sesuai

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QObject
from PyQt6.QtGui import QImage, QPixmap
import pyqtgraph as pg
import pyqtgraph.exporters

# --- PENGATURAN APLIKASI ---
JUMLAH_DATA_PLOT = 300 # Jumlah data point yang ditampilkan di grafik
UPDATE_HASIL_SETIAP = 1.0 # Detik, seberapa sering hasil BPM dan RR di-update
FOLDER_HASIL = "hasil_proses" # Nama folder output untuk plot dan CSV

class WorkerSignals(QObject):
    frame_update = pyqtSignal(np.ndarray)
    data_update = pyqtSignal(float, float, float) # timestamp, resp_plot_val, rppg_plot_val
    results_update = pyqtSignal(str, str) # respiration_rate, heart_rate
    finished = pyqtSignal(list, list, list, list) # +++ PERUBAHAN: Mengirim semua data mentah saat selesai

class VitalsWorker(QThread):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.signals = WorkerSignals()
        self.is_running = True
        # +++ TAMBAHAN: List untuk menyimpan semua data mentah yang akan di-emit
        self.all_timestamps = []
        self.all_respiration_data = []
        self.all_rppg_rgb_data = []
        self.all_live_rppg_plot_data = []


    def run(self):
        cap = cv2.VideoCapture(0)
        fps_aktual = cap.get(cv2.CAP_PROP_FPS)
        if fps_aktual == 0 or fps_aktual is None:
            fps_aktual = 30.0
        print(f"Kamera FPS terdeteksi/diatur: {fps_aktual}")

        mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        mp_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

        start_time = time.time()
        last_update_time = start_time
        
        # Menggunakan list instance yang sudah didefinisikan di __init__
        # agar bisa diakses saat emit 'finished'
        # timestamps, respiration_data, rppg_rgb_data, live_rppg_plot_data = [], [], [], []
        # --- PERUBAHAN: Menggunakan list dari instance ---
        self.all_timestamps.clear()
        self.all_respiration_data.clear()
        self.all_rppg_rgb_data.clear()
        self.all_live_rppg_plot_data.clear()


        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                print("Gagal membaca frame dari kamera.")
                break

            frame = cv2.flip(frame, 1)
            frame_height, frame_width, _ = frame.shape
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            y_avg = 0.0
            pos_signal_value = 0.0
            current_mean_rgb = np.array([0.0, 0.0, 0.0])

            results_pose = mp_pose.process(image_rgb)
            if results_pose.pose_landmarks:
                lm = results_pose.pose_landmarks.landmark
                if len(lm) > 12:
                    y_left = lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y * frame_height
                    y_right = lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame_height
                    y_avg = (y_left + y_right) / 2.0
                    cv2.circle(frame, (int(lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x * frame_width), int(y_left)), 8, (0, 255, 0), -1)
                    cv2.circle(frame, (int(lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame_width), int(y_right)), 8, (0, 255, 0), -1)

            results_face = mp_face.process(image_rgb)
            if results_face.detections:
                detection = results_face.detections[0]
                bboxC = detection.location_data.relative_bounding_box
                x,y,w,h = int(bboxC.xmin*frame_width), int(bboxC.ymin*frame_height), int(bboxC.width*frame_width), int(bboxC.height*frame_height)
                if w > 0 and h > 0:
                    face_roi = image_rgb[y:y+h, x:x+w]
                    if face_roi.size > 0:
                        mean_rgb_roi = np.mean(face_roi, axis=(0,1))
                        current_mean_rgb = np.array([mean_rgb_roi[0], mean_rgb_roi[1], mean_rgb_roi[2]])
                        mean_current_rgb = np.mean(current_mean_rgb)
                        if mean_current_rgb < 1e-9: mean_current_rgb = 1e-9
                        C = current_mean_rgb.T / mean_current_rgb
                        pos_signal_value = np.array([0,1,-1]) @ C + np.array([-2,1,1]) @ C
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
            
            current_time = time.time() - start_time
            self.all_timestamps.append(current_time)
            self.all_respiration_data.append(y_avg)
            self.all_rppg_rgb_data.append(current_mean_rgb)
            self.all_live_rppg_plot_data.append(pos_signal_value)
            
            if (time.time() - last_update_time) > UPDATE_HASIL_SETIAP:
                if len(self.all_timestamps) > 1:
                    current_timestamps_for_plot = self.all_timestamps[-JUMLAH_DATA_PLOT:]
                    sample_rate = 1.0 / np.mean(np.diff(current_timestamps_for_plot)) if len(current_timestamps_for_plot) > 1 else fps_aktual
                    
                    data_points_for_calc = int(sample_rate * 10)
                    if len(self.all_respiration_data) >= data_points_for_calc:
                        recent_resp_data = self.all_respiration_data[-data_points_for_calc:]
                        rr_result = vp.hitung_laju_napas(recent_resp_data, sample_rate)
                    else:
                        rr_result = "N/A (Kurang data)"

                    if len(self.all_rppg_rgb_data) >= data_points_for_calc:
                        recent_rgb_data = np.array(self.all_rppg_rgb_data[-data_points_for_calc:]).T
                        if recent_rgb_data.ndim == 2 and recent_rgb_data.shape[0] == 3:
                             bpm_result = vp.hitung_detak_jantung(recent_rgb_data, sample_rate)
                        else:
                            bpm_result = "N/A (Data RGB tidak valid)"
                    else:
                        bpm_result = "N/A (Kurang data)"
                    
                    self.signals.results_update.emit(rr_result, bpm_result)
                    last_update_time = time.time()

            filtered_resp = vp.filter_sinyal_respirasi(self.all_respiration_data, fps_aktual)
            filtered_pos = vp.bandpass_filter_rppg(self.all_live_rppg_plot_data, fps_aktual)
            self.signals.frame_update.emit(frame)
            self.signals.data_update.emit(
                current_time, 
                filtered_resp[-1] if len(filtered_resp) > 0 else 0.0, 
                filtered_pos[-1] if len(filtered_pos) > 0 else 0.0
            )

        # --- PERUBAHAN: Mengirim semua data mentah saat selesai ---
        self.signals.finished.emit(
            self.all_timestamps,
            self.all_respiration_data,
            self.all_rppg_rgb_data,
            self.all_live_rppg_plot_data # Meskipun ini untuk plot live, bisa juga disimpan
        )
        cap.release()
        mp_pose.close()
        mp_face.close()
        print("Worker thread selesai dan resource dilepaskan.")

    def stop(self):
        print("Perintah stop diterima oleh worker.")
        self.is_running = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.is_running = False
        self.setWindowTitle("Real-time Vitals Monitor (Modular)")
        self.setGeometry(100, 100, 1200, 600)

        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        self.video_label = QLabel("Tekan 'Mulai' untuk memulai")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white; font-size: 18px;")
        left_layout.addWidget(self.video_label, 5)

        control_layout = QHBoxLayout()
        self.start_button = QPushButton("Mulai")
        self.stop_button = QPushButton("Hentikan")
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        left_layout.addLayout(control_layout, 1)

        self.resp_plot_widget = pg.PlotWidget(title="Sinyal Pernapasan (RR)")
        self.resp_plot_widget.setLabel('bottom', 'Waktu (s)')
        self.resp_plot_widget.setLabel('left', 'Posisi Bahu (px)')
        self.resp_plot_widget.showGrid(x=True, y=True)
        self.plot_curve_resp = self.resp_plot_widget.plot(pen=pg.mkPen('c', width=2))

        self.rppg_plot_widget = pg.PlotWidget(title="Sinyal Detak Jantung (rPPG)")
        self.rppg_plot_widget.setLabel('bottom', 'Waktu (s)')
        self.rppg_plot_widget.setLabel('left', 'Amplitudo')
        self.rppg_plot_widget.showGrid(x=True, y=True)
        self.plot_curve_rppg = self.rppg_plot_widget.plot(pen=pg.mkPen('g', width=2))
        
        right_layout.addWidget(self.resp_plot_widget)
        right_layout.addWidget(self.rppg_plot_widget)
        
        results_layout = QHBoxLayout()
        self.rr_label = QLabel("Laju Napas: -")
        self.bpm_label = QLabel("Detak Jantung: -")
        results_layout.addWidget(self.rr_label)
        results_layout.addWidget(self.bpm_label)
        right_layout.addLayout(results_layout)

        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        main_layout.addWidget(left_widget, 2)
        main_layout.addWidget(right_widget, 1)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.start_button.clicked.connect(self.start_capture)
        self.stop_button.clicked.connect(self.stop_capture)
        
        self.timestamps_plot, self.resp_data_plot, self.rppg_data_plot = [], [], [] # Data untuk plot live

    def start_capture(self):
        if self.is_running:
            return
        self.is_running = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.rr_label.setText("Laju Napas: Mengukur...")
        self.bpm_label.setText("Detak Jantung: Mengukur...")
        
        self.timestamps_plot, self.resp_data_plot, self.rppg_data_plot = [], [], []
        self.plot_curve_resp.clear()
        self.plot_curve_rppg.clear()

        self.worker = VitalsWorker()
        self.worker.signals.frame_update.connect(self.update_frame)
        self.worker.signals.data_update.connect(self.update_plots)
        self.worker.signals.results_update.connect(self.update_results)
        # --- PERUBAHAN: Menghubungkan sinyal finished ke fungsi baru ---
        self.worker.signals.finished.connect(self.capture_finished_and_save_data)
        self.worker.start()
        print("Worker thread dimulai.")

    def stop_capture(self):
        if hasattr(self, 'worker') and self.is_running:
            print("Perintah stop dikirim ke MainWindow.")
            self.worker.stop()
            # self.stop_button.setEnabled(False) # Dinonaktifkan sementara, akan dihandle oleh capture_finished

    def update_frame(self, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
        self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def update_plots(self, timestamp, resp_val, rppg_val):
        self.timestamps_plot.append(timestamp)
        self.resp_data_plot.append(resp_val)
        self.rppg_data_plot.append(rppg_val)

        if len(self.timestamps_plot) > JUMLAH_DATA_PLOT:
            self.timestamps_plot.pop(0)
            self.resp_data_plot.pop(0)
            self.rppg_data_plot.pop(0)

        inverted_resp = np.array(self.resp_data_plot) * -1 
        self.plot_curve_resp.setData(self.timestamps_plot, inverted_resp)
        self.plot_curve_rppg.setData(self.timestamps_plot, self.rppg_data_plot)

    def update_results(self, rr, bpm):
        self.rr_label.setText(f"Laju Napas: {rr}")
        self.bpm_label.setText(f"Detak Jantung: {bpm}")

    # --- PERUBAHAN: Fungsi capture_finished dipecah ---
    def capture_finished_and_save_data(self, all_timestamps, all_respiration_data, all_rppg_rgb_data, all_live_rppg_plot_data):
        print("Sinyal 'finished' diterima oleh MainWindow.")
        self.is_running = False
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(True)
        
        if len(all_timestamps) > 10:
            self.save_plots() # Tetap simpan gambar plot
            # +++ TAMBAHAN: Panggil fungsi untuk menyimpan CSV +++
            self.save_data_to_csv(all_timestamps, all_respiration_data, all_rppg_rgb_data)
        else:
            print("Tidak cukup data untuk menyimpan plot atau CSV.")
            
        self.video_label.setText("Pengambilan data dihentikan. Tekan 'Mulai' lagi.")

    def closeEvent(self, event):
        print("Menutup aplikasi...")
        self.stop_capture()
        if hasattr(self, 'worker'):
            self.worker.wait()
        super().closeEvent(event)

    def save_plots(self):
        if not os.path.exists(FOLDER_HASIL):
            try:
                os.makedirs(FOLDER_HASIL)
                print(f"Folder '{FOLDER_HASIL}' telah dibuat.")
            except OSError as e:
                print(f"Gagal membuat folder '{FOLDER_HASIL}': {e}")
                return

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            exporter_resp = pg.exporters.ImageExporter(self.resp_plot_widget.getPlotItem())
            file_resp = os.path.join(FOLDER_HASIL, f"plot_respirasi_{timestamp_str}.png")
            exporter_resp.export(file_resp)
            print(f"Plot respirasi disimpan ke: {file_resp}")
        except Exception as e:
            print(f"Gagal menyimpan plot respirasi: {e}")

        try:
            exporter_rppg = pg.exporters.ImageExporter(self.rppg_plot_widget.getPlotItem())
            file_rppg = os.path.join(FOLDER_HASIL, f"plot_rppg_{timestamp_str}.png")
            exporter_rppg.export(file_rppg)
            print(f"Plot rPPG disimpan ke: {file_rppg}")
        except Exception as e:
            print(f"Gagal menyimpan plot rPPG: {e}")

    # +++ TAMBAHAN: FUNGSI BARU UNTUK MENYIMPAN DATA KE CSV +++
    def save_data_to_csv(self, timestamps, respiration_data, rppg_rgb_data):
        """Menyimpan data mentah yang terkumpul ke file CSV."""
        if not os.path.exists(FOLDER_HASIL):
            try:
                os.makedirs(FOLDER_HASIL) # Buat folder jika belum ada
            except OSError as e:
                print(f"Gagal membuat folder '{FOLDER_HASIL}' untuk CSV: {e}")
                return

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = os.path.join(FOLDER_HASIL, f"data_vitals_{timestamp_str}.csv")

        try:
            with open(csv_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Header CSV
                writer.writerow(['timestamp', 'raw_respiration_y', 'raw_r_value', 'raw_g_value', 'raw_b_value'])
                
                # Tulis data baris per baris
                for i in range(len(timestamps)):
                    ts = timestamps[i]
                    resp_y = respiration_data[i]
                    rgb_r = rppg_rgb_data[i][0] if len(rppg_rgb_data[i]) == 3 else 0.0
                    rgb_g = rppg_rgb_data[i][1] if len(rppg_rgb_data[i]) == 3 else 0.0
                    rgb_b = rppg_rgb_data[i][2] if len(rppg_rgb_data[i]) == 3 else 0.0
                    writer.writerow([ts, resp_y, rgb_r, rgb_g, rgb_b])
            print(f"Data mentah disimpan ke CSV: {csv_filename}")
        except Exception as e:
            print(f"Gagal menyimpan data ke CSV: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
