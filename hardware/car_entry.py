import platform
import cv2
from ultralytics import YOLO
import pytesseract
import os
import time
import serial
import serial.tools.list_ports
import csv
from collections import Counter

# Load YOLOv8 model
model = YOLO('../model_dev/runs/detect/train/weights/best.pt')

# Plate save directory and CSV log file
SAVE_DIR = 'plates'
CSV_FILE = 'plates_log.csv'
os.makedirs(SAVE_DIR, exist_ok=True)

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['no', 'entry_time', 'exit_time', 'car_plate', 'due payment', 'payment status'])

# Auto-detect Arduino Serial Port
def detect_arduino_port():
    ports = serial.tools.list_ports.comports()
    system = platform.system()
    for port in ports:
        dev = port.device
        if system == 'Linux' and 'ttyACM' in dev:
            return dev
        if system == 'Darwin' and ('usbmodem' in dev or 'usbserial' in dev):
            return dev
        if system == 'Windows' and 'COM' in dev:
            return dev
    return None

# Read distance from Arduino
def read_distance(arduino):
    if not arduino or arduino.in_waiting == 0:
        return None
    try:
        line = arduino.readline().decode('utf-8').strip()
        return float(line)
    except (UnicodeDecodeError, ValueError):
        return None

# Initialize Arduino connection
audruino_port = detect_arduino_port()
arduino = None
if audruino_port:
    print(f"[CONNECTED] Arduino on {audruino_port}")
    arduino = serial.Serial(audruino_port, 9600, timeout=1)
    time.sleep(2)
else:
    print("[ERROR] Arduino not detected.")

# Setup OpenCV windows as resizable
def init_windows():
    cv2.namedWindow('Webcam Feed', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Plate', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Processed', cv2.WINDOW_NORMAL)
    # Optionally set initial sizes
    cv2.resizeWindow('Webcam Feed', 800, 600)

# Initialize webcam
print("[DEBUG] Opening camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open the camera.")
    exit(1)

init_windows()

plate_buffer = []
entry_cooldown = 300  # seconds
last_saved_plate = None
last_entry_time = 0
entry_count = sum(1 for _ in open(CSV_FILE)) - 1

print("[SYSTEM] Ready. Press 'q' to exit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read from camera.")
            break

        # Read ultrasonic distance
        distance = read_distance(arduino) or 30
        annotated_frame = frame.copy()

        if distance <= 50:
            results = model(frame)[0]
            annotated_frame = results.plot()

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_img = frame[y1:y2, x1:x2]

                # Preprocess for OCR
                gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                thresh = cv2.threshold(blur, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

                # OCR extraction
                plate_text = pytesseract.image_to_string(
                    thresh,
                    config='--psm 8 --oem 3 '
                           '-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                ).strip().replace(' ', '')

                # Validate Rwandan plate pattern
                if plate_text.startswith('RA') and len(plate_text) >= 7:
                    candidate = plate_text[:7]
                    prefix, digits, suffix = candidate[:3], candidate[3:6], candidate[6]
                    if prefix.isalpha() and digits.isdigit() and suffix.isalpha():
                        plate_buffer.append(candidate)

                # After 3 captures, pick most common
                if len(plate_buffer) >= 3:
                    most_common = Counter(plate_buffer).most_common(1)[0][0]
                    now = time.time()
                    if (most_common != last_saved_plate or
                        (now - last_entry_time) > entry_cooldown):
                        with open(CSV_FILE, 'a', newline='') as f:
                            writer = csv.writer(f)
                            entry_count += 1
                            writer.writerow([
                                entry_count,
                                time.strftime('%Y-%m-%d %H:%M:%S'),
                                '', most_common, '', 0
                            ])
                        if arduino:
                            arduino.write(b'1')
                            time.sleep(15)
                            arduino.write(b'0')
                        last_saved_plate = most_common
                        last_entry_time = now
                    plate_buffer.clear()

                # Display plate and processed images
                cv2.imshow('Plate', plate_img)
                cv2.imshow('Processed', thresh)
                time.sleep(0.5)

        # Show annotated or raw feed
        cv2.imshow('Webcam Feed', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    if arduino:
        arduino.close()
    cv2.destroyAllWindows()
