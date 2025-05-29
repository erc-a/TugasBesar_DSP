import os
import cv2
import mediapipe as mp
import time
import csv
import numpy as np

# --- PENGATURAN ---
DURASI_REKAMAN = 30  # Detik
FPS_TARGET = 30      # Frames per second yang diinginkan
OUTPUT_FOLDER = 'penggabungan_output'  # Folder output untuk semua file
# --------------------

# Pastikan folder output ada
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Inisialisasi MediaPipe Pose dan Face Detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Inisialisasi kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Tidak bisa membuka kamera.")
    exit()

# Mengatur FPS kamera
cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
fps_aktual = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS kamera diatur ke: {fps_aktual:.2f} FPS")

# Mendapatkan resolusi frame
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Inisialisasi VideoWriter dan file CSV
file_video_output = os.path.join(OUTPUT_FOLDER, 'rekaman_vitals.mp4')
file_csv_output = os.path.join(OUTPUT_FOLDER, 'data_vitals.csv')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(file_video_output, fourcc, fps_aktual, (frame_width, frame_height))

csv_file = open(file_csv_output, 'w', newline='')
csv_writer = csv.writer(csv_file)
# Header CSV yang sudah digabung
csv_writer.writerow(['timestamp', 'left_shoulder_y', 'right_shoulder_y', 'avg_shoulder_y', 'mean_r', 'mean_g', 'mean_b'])

print("Bersiap... Posisikan wajah dan bahu Anda di depan kamera.")

# Countdown sebelum mulai
for i in range(5, 0, -1):
    ret, frame = cap.read()
    if not ret:
        break
    cv2.putText(frame, str(i), (frame_width // 2 - 50, frame_height // 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5, cv2.LINE_AA)
    cv2.imshow('Kamera Vitals', frame)
    cv2.waitKey(500)

print("Mulai merekam...")
start_time = time.time()
data_vitals = []

while (time.time() - start_time) < DURASI_REKAMAN:
    ret, frame = cap.read()
    if not ret:
        break

    # Balik frame secara horizontal (efek cermin)
    frame = cv2.flip(frame, 1)
    
    # Konversi BGR ke RGB untuk diproses MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Inisialisasi nilai default untuk data frame ini
    y_left, y_right, y_avg = 0.0, 0.0, 0.0
    mean_r, mean_g, mean_b = 0.0, 0.0, 0.0

    # --- 1. PROSES RESPIRASI (POSE DETECTION) ---
    results_pose = pose.process(image_rgb)
    if results_pose.pose_landmarks:
        landmarks = results_pose.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
            y_left = left_shoulder.y * frame_height
            y_right = right_shoulder.y * frame_height
            y_avg = (y_left + y_right) / 2
            
            # Gambar titik hijau di bahu
            cv2.circle(frame, (int(left_shoulder.x * frame_width), int(y_left)), 8, (0, 255, 0), -1)
            cv2.circle(frame, (int(right_shoulder.x * frame_width), int(y_right)), 8, (0, 255, 0), -1)

    # --- 2. PROSES RPPG (FACE DETECTION) ---
    results_face = face_detection.process(image_rgb)
    if results_face.detections:
        detection = results_face.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        x, y = int(bboxC.xmin * frame_width), int(bboxC.ymin * frame_height)
        w, h = int(bboxC.width * frame_width), int(bboxC.height * frame_height)
        
        # Pastikan ROI valid
        if w > 0 and h > 0:
            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size > 0:
                # Ekstrak nilai rata-rata RGB dari ROI wajah
                mean_bgr = np.mean(face_roi, axis=(0, 1))
                mean_r, mean_g, mean_b = mean_bgr[2], mean_bgr[1], mean_bgr[0]
            
            # Gambar kotak di wajah
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Simpan semua data yang terkumpul untuk frame ini
    current_time = time.time() - start_time
    data_vitals.append([current_time, y_left, y_right, y_avg, mean_r, mean_g, mean_b])

    # Tampilkan sisa waktu
    sisa_waktu = DURASI_REKAMAN - (time.time() - start_time)
    sisa_waktu = max(0, sisa_waktu)
    cv2.putText(frame, f"Sisa Waktu: {sisa_waktu:.2f}s", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Tampilkan frame hasil
    cv2.imshow('Kamera Vitals', frame)
    out_video.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tulis semua data ke file CSV
csv_writer.writerows(data_vitals)

print("\nPerekaman selesai.")
print(f"Video disimpan di: {file_video_output}")
print(f"Data CSV disimpan di: {file_csv_output}")

# Lepaskan semua resource
cap.release()
out_video.release()
pose.close()
face_detection.close()
csv_file.close()
cv2.destroyAllWindows()