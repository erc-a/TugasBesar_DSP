import numpy as np
import mediapipe as mp
import cv2
import os
import csv
import time
from datetime import datetime


def detect_faces(frame, face_detection):
    # Process the frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    # Default values if no face is detected
    x, y, width, height = 0, 0, 0, 0

    # Extract face bounding box if a face is detected
    if results.detections:
        detection = results.detections[0]  # Get the first detected face
        bboxC = detection.location_data.relative_bounding_box
        
        ih, iw, _ = frame.shape
        x = int(bboxC.xmin * iw)
        y = int(bboxC.ymin * ih)
        width = int(bboxC.width * iw)
        height = int(bboxC.height * ih)
    
    return x, y, width, height

def extract_mean_rgb_from_roi(image, x, y, width, height):
    # Extract the ROI using the bounding box
    roi = image[y:y+height, x:x+width]
    if roi.size == 0:
        return 0.0, 0.0, 0.0
    
    # Calculate mean RGB values
    mean_bgr = np.mean(roi, axis=(0,1))
    mean_r, mean_g, mean_b = mean_bgr[2], mean_bgr[1], mean_bgr[0]
    return mean_r, mean_g, mean_b

def main():
    Duration = 10
    FPS = 30
    OUTPUT_DIR = "rppg_data"
    COUNTDOWN = 3

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_file = os.path.join(OUTPUT_DIR, f"video_{time_stamp}.mp4")
    csv_file = os.path.join(OUTPUT_DIR, f"rppg_data_{time_stamp}.csv")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_file, fourcc, FPS, (frame_width, frame_height))

    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    # Add countdown before starting recording
    print("Get ready! Recording will start in 3 seconds...")
    while COUNTDOWN > 0:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            return

        # Display countdown on frame
        cv2.putText(frame, f"Starting in {COUNTDOWN}...", 
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("rPPG Recorder", frame)
        
        if cv2.waitKey(1000) & 0xFF == ord('q'):  # Wait 1 second between counts
            break
        COUNTDOWN -= 1

    with open(csv_file, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['frame_number', 'timestamp', 'R', 'G', 'B'])

        print(f"Recording video and extracting RGB values for {Duration} seconds...")
        start_time = time.time()
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            elapsed_time = time.time() - start_time
            if elapsed_time > Duration:
                break

            # Detect face and get bounding box
            x, y, w, h = detect_faces(frame, face_detection)

            mean_r, mean_g, mean_b = 0.0, 0.0, 0.0
            if w > 0 and h > 0:  # If face was detected
                mean_r, mean_g, mean_b = extract_mean_rgb_from_roi(frame, x, y, w, h)
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            out.write(frame)
            csv_writer.writerow([frame_count, f"{elapsed_time:.3f}", f"{mean_r:.2f}", f"{mean_g:.2f}", f"{mean_b:.2f}"])

            cv2.putText(frame, f"Recording... {int(Duration - elapsed_time)}s left",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("rPPG Recorder", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved to {video_file}")
    print(f"RGB data saved to {csv_file}")

if __name__ == "__main__":
    main()


