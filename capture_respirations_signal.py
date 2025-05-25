import cv2
import mediapipe as mp
import time
import csv
import numpy as np

# --- PENGATURAN ---
DURASI_REKAMAN = 10  # Detik
FPS_TARGET = 30      # Frames per second yang diinginkan
FILE_VIDEO_OUTPUT = 'rekaman_respirasi.mp4'
FILE_CSV_OUTPUT = 'data_respirasi.csv'
# --------------------

# Inisialisasi MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Inisialisasi kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Tidak bisa membuka kamera.")
    exit()

# Mengatur FPS kamera (tidak semua kamera mendukung ini)
cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
fps_aktual = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS kamera diatur ke: {fps_aktual} FPS")

# Mendapatkan resolusi frame
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Inisialisasi VideoWriter untuk menyimpan video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(FILE_VIDEO_OUTPUT, fourcc, fps_aktual, (frame_width, frame_height))

# Persiapan file CSV
csv_file = open(FILE_CSV_OUTPUT, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['timestamp', 'left_shoulder_y', 'right_shoulder_y', 'avg_shoulder_y'])

print("Bersiap... Posisikan bahu Anda di depan kamera.")

# Countdown sebelum mulai
for i in range(5, 0, -1):
    ret, frame = cap.read()
    if not ret:
        break
    
    # Tampilkan countdown di tengah layar
    cv2.putText(frame, str(i), (frame_width // 2 - 50, frame_height // 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5, cv2.LINE_AA)
    cv2.imshow('Kamera Respirasi', frame)
    cv2.waitKey(500)

print("Mulai merekam...")
start_time = time.time()
data_respirasi = []

while (time.time() - start_time) < DURASI_REKAMAN:
    ret, frame = cap.read()
    if not ret:
        break

    # Balik frame secara horizontal agar seperti cermin
    frame = cv2.flip(frame, 1)

    # Konversi warna BGR ke RGB untuk MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    # Gambar ulang ke BGR untuk ditampilkan
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Titik 11: Bahu Kiri (LEFT_SHOULDER)
        # Titik 12: Bahu Kanan (RIGHT_SHOULDER)
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        # Cek apakah kedua bahu terdeteksi
        if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
            # Ambil koordinat Y (posisi vertikal)
            # Dikalikan tinggi frame karena koordinat dari mediapipe dinormalisasi (0.0 - 1.0)
            y_left = left_shoulder.y * frame_height
            y_right = right_shoulder.y * frame_height
            y_avg = (y_left + y_right) / 2
            
            # Waktu saat ini relatif terhadap awal rekaman
            current_time = time.time() - start_time
            
            # Simpan data
            data_respirasi.append([current_time, y_left, y_right, y_avg])

            # Gambar titik hijau di bahu
            cv2.circle(frame, (int(left_shoulder.x * frame_width), int(y_left)), 
                       8, (0, 255, 0), -1)
            cv2.circle(frame, (int(right_shoulder.x * frame_width), int(y_right)), 
                       8, (0, 255, 0), -1)

    # Tampilkan sisa waktu
    sisa_waktu = DURASI_REKAMAN - (time.time() - start_time)
    if sisa_waktu < 0:
        sisa_waktu = 0
    cv2.putText(frame, f"Sisa Waktu: {sisa_waktu:.2f}s", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Tampilkan frame
    cv2.imshow('Kamera Respirasi', frame)
    
    # Tulis frame ke file video
    out.write(frame)

    # Keluar jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tulis semua data yang terkumpul ke CSV
csv_writer.writerows(data_respirasi)

print("Perekaman selesai.")
print(f"Video disimpan di: {FILE_VIDEO_OUTPUT}")
print(f"Data CSV disimpan di: {FILE_CSV_OUTPUT}")

# Lepaskan semua resource
cap.release()
out.release()
pose.close()
csv_file.close()
cv2.destroyAllWindows()