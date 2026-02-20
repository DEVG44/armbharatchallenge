import cv2
import numpy as np
import tensorflow as tf
import time
import os
import csv
import threading
import queue 
import RPi.GPIO as GPIO
from ultralytics import YOLO
from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import ssd1306
from PIL import ImageFont

# --- DIRECTORY SETUP ---
os.makedirs("logs/images", exist_ok=True)

# --- ASYNC SAVERS ---
save_queue = queue.Queue()
csv_queue = queue.Queue()

def image_saver_worker():
    while True:
        item = save_queue.get()
        if item:
            path, img_data = item
            cv2.imwrite(path, img_data)
        save_queue.task_done()

def csv_worker():
    log_path = "logs/road_data.csv"
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow(["Timestamp", "Object", "Conf", "Temp", "Image"])
    while True:
        data = csv_queue.get()
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow(data)
        csv_queue.task_done()

threading.Thread(target=image_saver_worker, daemon=True).start()
threading.Thread(target=csv_worker, daemon=True).start()

# --- ENHANCEMENT ---
def enhance_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) < 80:
        img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return frame

# --- GPIO SETUP ---
CLASS_MAP = {
    4: {"label": "POTHOLE", "color": (0, 0, 255), "pin": 17},
    3: {"label": "PERSON",  "color": (255, 0, 0), "pin": 27},
    0: {"label": "ANIMAL",  "color": (0, 255, 0), "pin": 22},
    5: {"label": "VEHICLE", "color": (0, 165, 255), "pin": 23},
    2: {"label": "OBSTACLE","color": (255, 255, 255), "pin": 24},
    1: {"label": "CRACK",   "color": (0, 255, 255), "pin": 25}
}
BUZZER_PIN = 26
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO.setup(BUZZER_PIN, GPIO.OUT, initial=GPIO.LOW)
for cls in CLASS_MAP.values(): 
    GPIO.setup(cls["pin"], GPIO.OUT, initial=GPIO.LOW)

# --- CONFIG ---
MODEL_PATH = "best.tflite" 
IMG_SIZE = 640 
SMOOTH_BOX = 0.15   
SMOOTH_CONF = 0.05  
LOG_COOLDOWN = 5 
last_log_time = {cid: 0 for cid in CLASS_MAP.keys()}

class State:
    label, conf, running = "SCANNING", 0.0, True
    smoothed_boxes = {}
    smoothed_confs = {}
state = State()

def get_temp_raw():
    return os.popen("vcgencmd measure_temp").readline().replace("temp=","").replace("'C\n","").strip()

def oled_worker():
    try:
        serial = i2c(port=1, address=0x3C)
        device = ssd1306(serial)
        f_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        f_m = ImageFont.truetype(f_path, 18); f_s = ImageFont.truetype(f_path, 10); f_h = ImageFont.truetype(f_path, 28)
        while state.running:
            with canvas(device) as draw:
                draw.text((64, 5), state.label.upper(), font=f_m, fill="white", anchor="mt")
                draw.text((64, 25), "Confidence:", font=f_s, fill="white", anchor="mt")
                draw.text((64, 38), f"{int(state.conf)}%", font=f_h, fill="white", anchor="mt")
            time.sleep(0.1)
    except: pass

threading.Thread(target=oled_worker, daemon=True).start()

model = YOLO(MODEL_PATH, task="detect")
cap = cv2.VideoCapture(0)
cap.set(3, IMG_SIZE); cap.set(4, IMG_SIZE)
prev_time, buzzer_end = 0, 0

try:
    while True:
        ret, frame = cap.read()
        if not ret: break

        proc_frame = enhance_frame(frame)
        results = model.predict(source=proc_frame, imgsz=IMG_SIZE, conf=0.15, iou=0.2, agnostic_nms=True, verbose=False)
        
        for cls in CLASS_MAP.values(): GPIO.output(cls["pin"], GPIO.LOW)
        
        new_boxes, new_confs = {}, {}
        should_save = False
        save_path = ""
        log_ts = ""

        if len(results[0].boxes) > 0:
            top_idx = results[0].boxes.conf.argmax()

            for i, box in enumerate(results[0].boxes):
                cid = int(box.cls[0]); raw_c = float(box.conf[0])
                if cid in CLASS_MAP:
                    # Robust Smoothing
                    val_to_smooth = raw_c * 100
                    p_c = state.smoothed_confs.get(i, val_to_smooth)
                    s_c = (SMOOTH_CONF * val_to_smooth) + ((1 - SMOOTH_CONF) * p_c)
                    new_confs[i] = s_c

                    if i == top_idx:
                        state.conf = s_c
                        state.label = CLASS_MAP[cid]["label"]
                        GPIO.output(CLASS_MAP[cid]["pin"], GPIO.HIGH)
                        if cid == 4 and raw_c > 0.5: 
                            GPIO.output(BUZZER_PIN, GPIO.HIGH)
                            buzzer_end = time.time() + 0.5

                    # Clamped Coordinate Smoothing
                    raw_pts = box.xyxy[0].cpu().numpy()
                    if i in state.smoothed_boxes:
                        raw_pts = (SMOOTH_BOX * raw_pts + (1 - SMOOTH_BOX) * state.smoothed_boxes[i])
                    new_boxes[i] = raw_pts

                    x1, y1, x2, y2 = np.clip(raw_pts, 0, IMG_SIZE).astype(int)

                    if raw_c > 0.20:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), CLASS_MAP[cid]["color"], 3)
                        cv2.putText(frame, f"{CLASS_MAP[cid]['label']} {int(s_c)}%", (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLASS_MAP[cid]["color"], 2)

                    if raw_c > 0.25:
                        if (time.time() - last_log_time[cid]) > LOG_COOLDOWN:
                            last_log_time[cid] = time.time()
                            cur_t = get_temp_raw()
                            display_ts = time.strftime("%Y-%m-%d %H:%M:%S")
                            ts_file = time.strftime("%Y-%m-%d_%H-%M-%S")
                            f_path = f"logs/images/{CLASS_MAP[cid]['label']}_{ts_file}.jpg"
                            csv_queue.put([display_ts, CLASS_MAP[cid]['label'], round(raw_c, 2), f"{cur_t}°C", f_path if raw_c > 0.5 else "N/A"])
                            
                            if raw_c > 0.5:
                                should_save = True
                                save_path = f_path
                                log_ts = display_ts

            state.smoothed_boxes = new_boxes
            state.smoothed_confs = new_confs
        else:
            state.label, state.conf = "SCANNING", 0
            state.smoothed_boxes.clear(); state.smoothed_confs.clear()

        # --- CAPTURE CLEAN LOG (With boxes, no FPS) ---
        if should_save:
            log_img = frame.copy() 
            cv2.putText(log_img, log_ts, (log_img.shape[1]-420, log_img.shape[0]-30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
            save_queue.put((save_path, log_img))

        # --- MONITOR DIAGNOSTICS (Drawn after snapshot) ---
        curr_t = time.time()
        fps = 1 / (curr_t - prev_time) if prev_time != 0 else 0
        prev_time = curr_t
        t_val = get_temp_raw()
        hud_main = f"FPS: {int(fps)} | CPU: {t_val}"
        cv2.putText(frame, hud_main, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        text_sz = cv2.getTextSize(hud_main, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        deg_x = 20 + text_sz[0] + 4
        cv2.circle(frame, (deg_x, 30), 3, (0, 255, 0), 2)
        cv2.putText(frame, "C", (deg_x + 8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if time.time() > buzzer_end: GPIO.output(BUZZER_PIN, GPIO.LOW)
        cv2.imshow("Road AI Diagnostic", frame)
        if cv2.waitKey(1) != -1: break
finally:
    state.running = False
    GPIO.cleanup(); cap.release(); cv2.destroyAllWindows()