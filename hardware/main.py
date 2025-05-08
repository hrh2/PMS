# === MODIFIED entrance.py ===
import cv2
from ultralytics import YOLO
import pytesseract
import os
import time
import serial
import serial.tools.list_ports
import csv
from collections import Counter
import platform

model = YOLO('../model_dev/runs/detect/train/weights/best.pt')
save_dir = 'plates'
os.makedirs(save_dir, exist_ok=True)

csv_file = 'plates_log.csv'
if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['no', 'entry_time', 'exit_time', 'car_plate', 'due payment', 'payment status'])

def detect_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    system = platform.system()
    for port in ports:
        if system == "Linux" and ("ttyUSB" in port.device or "ttyACM" in port.device):
            return port.device
        elif system == "Darwin" and ("usbmodem" in port.device or "usbserial" in port.device):
            return port.device
        elif system == "Windows" and "COM" in port.device:
            return port.device
    return None

# ===== Ultrasonic Sensor Reading from Arduino =====
def read_distance(arduino):
    """
    Reads a distance (float) value from the Arduino via serial.
    Returns the float if valid, or None if invalid/empty.
    """
    if arduino and arduino.in_waiting > 0:
        try:
            line = arduino.readline().decode('utf-8').strip()
            return float(line)
        except ValueError:
            return None
    return None


arduino_port = detect_arduino_port()
arduino = serial.Serial(arduino_port, 9600, timeout=1) if arduino_port else None
if arduino:
    time.sleep(2)

cap = cv2.VideoCapture(0)
plate_buffer = []
entry_cooldown = 300
last_saved_plate = None
last_entry_time = 0

print("[SYSTEM] Ready. Press 'q' to exit.")

entry_count = sum(1 for _ in open(csv_file)) - 1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    distance = read_distance(arduino)
    if distance is None:
        continue

    results = model(frame)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_img = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            plate_text = pytesseract.image_to_string(
                thresh, config='--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            ).strip().replace(" ", "")

            if "RA" in plate_text:
                plate_candidate = plate_text[plate_text.find("RA"):][:7]
                if (len(plate_candidate) == 7 and plate_candidate[:3].isalpha() and
                    plate_candidate[3:6].isdigit() and plate_candidate[6].isalpha()):

                    print(f"[VALID] Plate Detected: {plate_candidate}")
                    plate_buffer.append(plate_candidate)

                    if len(plate_buffer) >= 3:
                        most_common = Counter(plate_buffer).most_common(1)[0][0]
                        current_time = time.time()

                        if (most_common != last_saved_plate or
                            (current_time - last_entry_time) > entry_cooldown):

                            with open(csv_file, 'a', newline='') as f:
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
                            last_entry_time = current_time
                        else:
                            print("[SKIPPED] Duplicate within cooldown.")

                        plate_buffer.clear()

    annotated_frame = results[0].plot() if distance <= 50 else frame
    cv2.imshow('Webcam Feed', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if arduino:
    arduino.close()
cv2.destroyAllWindows()
