# main_gui_final.py
import tkinter as tk
from tkinter import ttk, font
import cv2
import collections
import numpy as np
from PIL import Image, ImageTk
import threading
import queue
import time

# Import modul kita
import video_processor as vp
import signal_processor as sp

# Import untuk integrasi Matplotlib dengan Tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class HealthMonitorApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # --- Konfigurasi Window & Variabel ---
        self.title("Real-time Health Monitor (Auto-Start v2)")
        self.geometry("1200x700")
        self.configure(bg="#2c3e50")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Variabel untuk multithreading
        self.data_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = None
        
        # Variabel lain
        self.FPS = 30
        self.BUFFER_SECONDS = 5
        self.BUFFER_SIZE = self.FPS * self.BUFFER_SECONDS
        
        # --- PERBAIKAN: Inisialisasi prosesor di awal ---
        self.face_detector = vp.initialize_face_detector()
        self.cap = None
        self.last_y = 0

        # --- Buat Widget GUI ---
        self._create_widgets()
        
        # --- Mulai Countdown ---
        self.start_countdown(3)

    def _create_widgets(self):
        # ... (Kode ini sama seperti sebelumnya, tidak ada perubahan) ...
        default_font = font.Font(family="Helvetica", size=12)
        title_font = font.Font(family="Helvetica", size=16, weight="bold")
        value_font = font.Font(family="Helvetica", size=20, weight="bold")

        main_frame = tk.Frame(self, bg="#2c3e50")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = tk.Frame(main_frame, bg="#34495e")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.video_label = tk.Label(left_frame, bg="black", text="Bersiap...", font=title_font, fg="white")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        data_frame = tk.Frame(left_frame, bg="#34495e")
        data_frame.pack(fill=tk.X, padx=10, pady=10)

        self.hr_label = tk.Label(data_frame, text="HR: -- BPM", font=value_font, fg="#1abc9c", bg="#34495e")
        self.hr_label.pack(side=tk.LEFT, expand=True)
        self.rr_label = tk.Label(data_frame, text="RR: -- napas/menit", font=value_font, fg="#3498db", bg="#34495e")
        self.rr_label.pack(side=tk.RIGHT, expand=True)

        right_frame = tk.Frame(main_frame, bg="#34495e")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(6, 6))
        self.fig.patch.set_facecolor('#34495e')

        self.ax1.set_title("Sinyal rPPG", color='white'); self.ax1.set_ylabel("Amplitudo", color='white'); self.ax1.tick_params(axis='both', colors='white'); self.ax1.set_facecolor('#2c3e50')
        self.line_rppg, = self.ax1.plot([], [], 'r-')
        self.ax2.set_title("Sinyal Pernapasan", color='white'); self.ax2.set_xlabel("Waktu (s)", color='white'); self.ax2.set_ylabel("Amplitudo", color='white'); self.ax2.tick_params(axis='both', colors='white'); self.ax2.set_facecolor('#2c3e50')
        self.line_resp, = self.ax2.plot([], [], 'b-')
        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas.draw()
        
        control_frame = tk.Frame(self, bg="#2c3e50")
        control_frame.pack(fill=tk.X, padx=10, pady=(5, 10))
        self.btn_quit = ttk.Button(control_frame, text="Keluar", command=self.on_closing)
        self.btn_quit.pack(expand=True, padx=5, ipady=5)

    def start_countdown(self, count):
        if count > 0:
            self.video_label.config(text=f"{count}", font=("Helvetica", 100, "bold"))
            self.after(1000, self.start_countdown, count - 1)
        else:
            self.video_label.config(text="")
            self.start_processing()

    def start_processing(self):
        self.cap = vp.initialize_webcam()
        self.worker_thread = threading.Thread(target=self.processing_worker, daemon=True)
        self.worker_thread.start()
        self.update_gui()

    def processing_worker(self):
        rgb_buffer = collections.deque(maxlen=self.BUFFER_SIZE)
        respiration_buffer = collections.deque(maxlen=self.BUFFER_SIZE)
        while not self.stop_event.is_set():
            try:
                data_packet = self.data_queue.get(timeout=1)
                rgb_buffer.append(data_packet['rgb'])
                respiration_buffer.append(data_packet['resp'])
                if len(rgb_buffer) == self.BUFFER_SIZE:
                    pos_signal = sp.pos_g(np.array(rgb_buffer).T, self.FPS)
                    filtered_rppg = sp.bandpass_filter(pos_signal, 0.8, 2.5, self.FPS)
                    hr = sp.calculate_rate(filtered_rppg, self.FPS)
                    filtered_resp = sp.bandpass_filter(np.array(respiration_buffer), 0.1, 0.5, self.FPS)
                    rr = sp.calculate_rate(filtered_resp, self.FPS)
                    self.result_queue.put({"hr": hr, "rr": rr, "rppg_signal": filtered_rppg, "resp_signal": filtered_resp})
            except queue.Empty:
                continue

    def update_gui(self):
        if self.stop_event.is_set() or not self.cap:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            # --- PERBAIKAN: Gunakan self.face_detector yang sudah pasti ada ---
            bbox, rgb_frame = vp.detect_face(frame, self.face_detector)

            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Kita dapatkan rgb_frame dari vp.detect_face, jadi tidak perlu convert lagi
                mean_rgb = vp.extract_mean_rgb(rgb_frame, bbox)
                movement = y - self.last_y if self.last_y > 0 else 0
                self.last_y = y
                self.data_queue.put({'rgb': mean_rgb, 'resp': movement})

            try:
                results = self.result_queue.get_nowait()
                self.hr_label.config(text=f"HR: {results['hr']:.1f} BPM")
                self.rr_label.config(text=f"RR: {results['rr']:.1f} napas/menit")
                time_axis = np.linspace(0, self.BUFFER_SECONDS, self.BUFFER_SIZE)
                self.line_rppg.set_data(time_axis, results['rppg_signal'])
                self.line_resp.set_data(time_axis, results['resp_signal'])
                self.ax1.relim(); self.ax1.autoscale_view()
                self.ax2.relim(); self.ax2.autoscale_view()
                self.canvas.draw()
            except queue.Empty:
                pass

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

        self.after(int(1000/self.FPS), self.update_gui)

    def on_closing(self):
        print("Menutup aplikasi...")
        self.stop_event.set()
        if self.worker_thread is not None:
            self.worker_thread.join()
        if self.cap is not None:
            self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = HealthMonitorApp()
    app.mainloop()