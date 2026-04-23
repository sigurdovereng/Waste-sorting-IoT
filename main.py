from gpiozero import MotionSensor, LED
from time import sleep
import os
import time
import cv2
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter
import subprocess
import threading
## SIGURD DASHBOARD
import json
import shutil
from datetime import datetime

# =========================
# GPIO SETUP
# =========================
pir = MotionSensor(17)

plast = LED(22)
organic = LED(23)
glass = LED(24)
paper = LED(25)

leds = [plast, organic, glass, paper]


# =========================
# PATHS
# =========================
SAVE_FOLDER = "/home/pi4/Desktop/SF"

LOG_FOLDER = "/home/pi4/Desktop/SF/logs"
LOG_FILE = f"{LOG_FOLDER}/log.txt"

DETECTOR_MODEL_PATH = "/home/pi4/Desktop/SF/coco_detector.tflite"
DETECTOR_LABELS_PATH = "/home/pi4/Desktop/SF/coco_labels.txt"

CLASSIFIER_MODEL_PATH = "/home/pi4/Desktop/SF/waste_classifier.tflite"
CLASSIFIER_LABELS_PATH = "/home/pi4/Desktop/SF/labels.txt"

## SIGURD DASHBOARD
DASHBOARD_JSON = "/home/pi4/Desktop/SF/dashboard_data.json"
LATEST_FOLDER = "/home/pi4/Desktop/SF/static/latest"


# =========================
# CONFIG
# =========================
DETECTION_CONFIDENCE_THRESHOLD = 0.45
CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.65
CLASSIFICATION_MARGIN_THRESHOLD = 0.15

IGNORED_CLASSES = {
    "person",
    "dining table",
    "tv",
    "couch",
    "car",
    "bus",
    "train",
    "motorcycle",
    "airplane",
    "boat"
}

LABEL_MAP = {
    "plast": "plastic",
    "plastic": "plastic",

    "organisk": "organic",
    "organic": "organic",
    "bio": "organic",
    "food": "organic",

    "papir": "paper",
    "paper": "paper",
    "cardboard": "paper",

    "glass": "glass_metall",
    "glass_metall": "glass_metall",
    "glass_metal": "glass_metall",
    "metal": "glass_metall"
}


# =========================
# HELPERS
# =========================
def load_labels(path):
    labels = {}
    with open(path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            # supports both "0 person" and plain "person"
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                labels[int(parts[0])] = parts[1]
            else:
                labels[i] = line
    return labels

# Her begynner logging delen
def write_log(text):
    os.makedirs(LOG_FOLDER, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text + "\n")


def all_off():
    plast.off()
    organic.off()
    glass.off()
    paper.off()


def loading_animation(stop_event):
    while not stop_event.is_set():
        for led in leds:
            if stop_event.is_set():
                break
            all_off()
            led.on()
            sleep(0.2)

    all_off()
    
def start_loading_animation():
    stop_event = threading.Event()
    thread = threading.Thread(
        target=loading_animation,
        args=(stop_event,),
        daemon=True
    )
    thread.start()
    return stop_event, thread


def stop_loading_animation(stop_event, thread):
    stop_event.set()
    thread.join(timeout=1)
    all_off()


def show_result(result):
    all_off()

    if result == "plastic":
        plast.on()
    elif result == "organic":
        organic.on()
    elif result == "glass":
        glass.on()
    elif result == "paper":
        paper.on()

    sleep(3)
    all_off()


# =========================
# TFLITE OBJECT DETECTOR
# =========================
class CocoDetector:
    def __init__(self, model_path, labels_path):
        self.labels = load_labels(labels_path)

        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = self.input_details[0]["shape"]
        self.input_height = int(self.input_shape[1])
        self.input_width = int(self.input_shape[2])
        self.input_dtype = self.input_details[0]["dtype"]
        

        print(f"Detector loaded: {model_path}")
        print(f"Detector input size: {self.input_width}x{self.input_height}")
        print(f"Detector input dtype: {self.input_dtype}")

    def preprocess(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_width, self.input_height))

        if self.input_dtype == np.float32:
            input_data = np.expand_dims(resized.astype(np.float32), axis=0)
            input_data = (input_data / 127.5) - 1.0
        else:
            input_data = np.expand_dims(resized.astype(np.uint8), axis=0)

        return input_data

    def detect(self, frame):
        h, w = frame.shape[:2]

        input_data = self.preprocess(frame)
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        boxes_detail = self.output_details[0]
        classes_detail = self.output_details[1]
        scores_detail = self.output_details[2]
        num_detail = self.output_details[3]

        boxes = self.interpreter.get_tensor(boxes_detail["index"])[0]
        classes = self.interpreter.get_tensor(classes_detail["index"])[0]
        scores = self.interpreter.get_tensor(scores_detail["index"])[0]
        num = int(self.interpreter.get_tensor(num_detail["index"])[0])

        print(f"[DEBUG] boxes tensor: {boxes_detail['name']} {boxes.shape}")
        print(f"[DEBUG] classes tensor: {classes_detail['name']} {classes.shape}")
        print(f"[DEBUG] scores tensor: {scores_detail['name']} {scores.shape}")
        print(f"[DEBUG] num tensor: {num_detail['name']} {num}")

        results = []

        for i in range(num):
            score = float(scores[i])
            if score < DETECTION_CONFIDENCE_THRESHOLD:
                continue

            class_id = int(classes[i])
            class_name = self.labels.get(class_id, f"id_{class_id}")

            if class_name in IGNORED_CLASSES:
                continue

            ymin, xmin, ymax, xmax = boxes[i]

            x1 = max(0, int(xmin * w))
            y1 = max(0, int(ymin * h))
            x2 = min(w, int(xmax * w))
            y2 = min(h, int(ymax * h))

            if x2 <= x1 or y2 <= y1:
                continue

            area = (x2 - x1) * (y2 - y1)
            if area < 2000:
                continue

            results.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": score,
                "box": (x1, y1, x2, y2),
                "area": area
            })

        return results


# =========================
# TFLITE CLASSIFIER
# =========================
class WasteClassifier:
    def __init__(self, model_path, labels_path):
        with open(labels_path, "r") as f:
            self.class_names = [line.strip() for line in f if line.strip()]

        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = self.input_details[0]["shape"]
        self.input_dtype = self.input_details[0]["dtype"]

        self.img_height = int(self.input_shape[1])
        self.img_width = int(self.input_shape[2])

        print(f"Classifier loaded: {model_path}")
        print(f"Classifier labels: {self.class_names}")

    def preprocess_image(self, image_path):
        img = Image.open(image_path).convert("RGB")
        img = img.resize((self.img_width, self.img_height))
        img_array = np.array(img).astype(np.float32)

        return np.expand_dims(img_array, axis=0)

    def classify(self, image_path):
        input_data = self.preprocess_image(image_path)

        start_time = time.time()
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])
        inference_time = (time.time() - start_time) * 1000

        probs = output_data[0]
        idx = int(np.argmax(probs))
        predicted_class = self.class_names[idx]
        confidence = float(probs[idx])
        
        print("Raw output:", output_data[0])
        print("Sum output:", float(np.sum(output_data[0])))

        return {
            "class": predicted_class,
            "confidence": confidence,
            "all_predictions": {
                self.class_names[i]: float(probs[i])
                for i in range(len(self.class_names))
            },
            "inference_time_ms": round(inference_time, 2),
        }


# Load both models once
try:
    detector = CocoDetector(DETECTOR_MODEL_PATH, DETECTOR_LABELS_PATH)
    classifier = WasteClassifier(CLASSIFIER_MODEL_PATH, CLASSIFIER_LABELS_PATH)
except Exception as e:
    print(f"Failed to load models: {e}")
    all_off()
    raise


# =========================
# DETECT + CROP
# =========================
def detect_and_crop_object(image_path):
    original = cv2.imread(image_path)

    if original is None:
        print("Could not read image.")
        return None

    original_h, original_w = original.shape[:2]

    # resized copy for detection
    frame = cv2.resize(original, (640, 480))
    h, w = frame.shape[:2]

    detections = detector.detect(frame)

    if not detections:
        print("No valid detected objects found.")
        return None

    center_x = w / 2
    center_y = h / 2
    max_distance = (center_x ** 2 + center_y ** 2) ** 0.5
    image_area = w * h

    def score_box(item):
        x1, y1, x2, y2 = item["box"]
        box_center_x = (x1 + x2) / 2
        box_center_y = (y1 + y2) / 2

        distance = ((box_center_x - center_x) ** 2 + (box_center_y - center_y) ** 2) ** 0.5
        center_score = 1 - (distance / max_distance)
        area_score = item["area"] / image_area
        confidence_score = item["confidence"]

        return (0.45 * confidence_score) + (0.35 * area_score) + (0.20 * center_score)

    best = max(detections, key=score_box)
    x1, y1, x2, y2 = best["box"]

    # add padding on resized image coordinates
    padding = 30
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    # map resized coords back to original image
    scale_x = original_w / w
    scale_y = original_h / h

    ox1 = int(x1 * scale_x)
    oy1 = int(y1 * scale_y)
    ox2 = int(x2 * scale_x)
    oy2 = int(y2 * scale_y)

    cropped = original[oy1:oy2, ox1:ox2]

    cropped_path = image_path.replace(".jpg", "_cropped.jpg")
    cv2.imwrite(cropped_path, cropped)

    debug_image = frame.copy()
    for item in detections:
        dx1, dy1, dx2, dy2 = item["box"]
        label = f"{item['class_name']} {item['confidence']:.2f}"
        cv2.rectangle(debug_image, (dx1, dy1), (dx2, dy2), (255, 0, 0), 2)
        cv2.putText(debug_image, label, (dx1, max(20, dy1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    debug_path = image_path.replace(".jpg", "_debug.jpg")
    cv2.imwrite(debug_path, debug_image)

    print(f"Selected detected object: {best['class_name']} ({best['confidence']:.2f})")
    print(f"Cropped image saved: {cropped_path}")
    print(f"Debug image saved: {debug_path}")

    return cropped_path



def classify_waste(cropped_image_path):
    if cropped_image_path is None:
        return None

    result = classifier.classify(cropped_image_path)

    sorted_predictions = sorted(
        result["all_predictions"].items(),
        key=lambda x: x[1],
        reverse=True
    )

    top1_class, top1_conf = sorted_predictions[0]
    top2_class, top2_conf = sorted_predictions[1]
    margin = top1_conf - top2_conf

    predicted_raw = top1_class.strip().lower()
    mapped_result = LABEL_MAP.get(predicted_raw)

    print(f"Raw model prediction: {top1_class}")
    print(f"Mapped prediction: {mapped_result}")
    print(f"Confidence: {top1_conf:.2%}")
    print(f"Second best: {top2_class} ({top2_conf:.2%})")
    print(f"Margin: {margin:.2%}")
    print(f"Inference time: {result['inference_time_ms']} ms")

    print("Top predictions:")
    for cls, prob in sorted_predictions:
        print(f"  {cls}: {prob:.2%}")

    if mapped_result is None:
        print("Prediction could not be mapped to a valid waste category.")
        return None

    if top1_conf < CLASSIFICATION_CONFIDENCE_THRESHOLD:
        print("Prediction rejected: confidence too low.")
        return None

    if margin < CLASSIFICATION_MARGIN_THRESHOLD:
        print("Prediction rejected: top two classes are too close.")
        return None

    return mapped_result

## SIGURD DASHBOARD -- DENNE BYTTER UT classify_waste SENERE
def classify_waste_with_details(cropped_image_path):
    if cropped_image_path is None:
        return {
            "status": "no_cropped_image",
            "predicted_class": None,
            "raw_prediction": None,
            "confidence": None,
            "top2_class": None,
            "top2_confidence": None,
            "inference_time_ms": None
        }

    result = classifier.classify(cropped_image_path)

    sorted_predictions = sorted(
        result["all_predictions"].items(),
        key=lambda x: x[1],
        reverse=True
    )

    top1_class, top1_conf = sorted_predictions[0]
    top2_class, top2_conf = sorted_predictions[1]
    margin = top1_conf - top2_conf

    predicted_raw = top1_class.strip().lower()
    mapped_result = LABEL_MAP.get(predicted_raw)

    if mapped_result == "glass_metall":
        dashboard_class = "glass"
    else:
        dashboard_class = mapped_result

    if mapped_result is None:
        return {
            "status": "invalid_mapping",
            "predicted_class": None,
            "raw_prediction": top1_class,
            "confidence": round(top1_conf, 4),
            "top2_class": top2_class,
            "top2_confidence": round(top2_conf, 4),
            "inference_time_ms": result["inference_time_ms"]
        }

    if top1_conf < CLASSIFICATION_CONFIDENCE_THRESHOLD:
        return {
            "status": "rejected_low_confidence",
            "predicted_class": dashboard_class,
            "raw_prediction": top1_class,
            "confidence": round(top1_conf, 4),
            "top2_class": top2_class,
            "top2_confidence": round(top2_conf, 4),
            "inference_time_ms": result["inference_time_ms"]
        }

    if margin < CLASSIFICATION_MARGIN_THRESHOLD:
        return {
            "status": "rejected_low_margin",
            "predicted_class": dashboard_class,
            "raw_prediction": top1_class,
            "confidence": round(top1_conf, 4),
            "top2_class": top2_class,
            "top2_confidence": round(top2_conf, 4),
            "inference_time_ms": result["inference_time_ms"]
        }

    return {
        "status": "accepted",
        "predicted_class": dashboard_class,
        "raw_prediction": top1_class,
        "confidence": round(top1_conf, 4),
        "top2_class": top2_class,
        "top2_confidence": round(top2_conf, 4),
        "inference_time_ms": result["inference_time_ms"]
    }

def capture_image(image_path):
    try:
        subprocess.run(
            ["rpicam-still", "-o", image_path],
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Camera command failed: {e}")
        return False
    except Exception as e:
        print(f"Unexpected camera error: {e}")
        return False

## SIGURD DASHBOARD
def update_dashboard(original_path=None, cropped_path=None, debug_path=None, result=None):
    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "original_image": "/static/latest/latest.jpg" if original_path else None,
        "cropped_image": "/static/latest/latest_cropped.jpg" if cropped_path else None,
        "debug_image": "/static/latest/latest_debug.jpg" if debug_path else None,
        "result": result or {}
    }

    if original_path and os.path.exists(original_path):
        shutil.copy(original_path, f"{LATEST_FOLDER}/latest.jpg")

    if cropped_path and os.path.exists(cropped_path):
        shutil.copy(cropped_path, f"{LATEST_FOLDER}/latest_cropped.jpg")

    if debug_path and os.path.exists(debug_path):
        shutil.copy(debug_path, f"{LATEST_FOLDER}/latest_debug.jpg")

    try:
        with open(DASHBOARD_JSON, "r") as f:
            existing = json.load(f)
    except:
        existing = {"latest": {"result": {}}, "history": []}

    history = existing.get("history", [])
    history.insert(0, {
        "timestamp": data["timestamp"],
        "status": result.get("status") if result else None,
        "predicted_class": result.get("predicted_class") if result else None,
        "confidence": result.get("confidence") if result else None
    })
    history = history[:10]

    final_data = {
        "latest": data,
        "history": history
    }

    with open(DASHBOARD_JSON, "w") as f:
        json.dump(final_data, f, indent=2)

## SIGURD DASHBOARD - OPPDATERT VERSJON
def process_motion_event():
    timestamp = int(time.time())
    image_path = f"{SAVE_FOLDER}/image_{timestamp}.jpg"

## Logging
    log_text = ""
    log_text += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    log_text += f"Image: image_{timestamp}.jpg\n"
    log_text += "Motion detected: yes\n"

    

    print("Capturing image...")
    success = capture_image(image_path)

    if not success:
        print("Image capture failed.")
        log_text += "Image captured: no\n"
        log_text += "Status: camera_failed\n"
        log_text += "LED: none\n"
        write_log(log_text)
        update_dashboard(
            result={"status": "camera_failed", "predicted_class": None, "confidence": None}
        )
        return

    print(f"Image captured and saved: {image_path}")
    log_text += "Image captured: yes\n"
    log_text += f"File: {image_path}\n"

    print("Detecting and cropping object...")
    cropped_path = detect_and_crop_object(image_path)
    debug_path = image_path.replace(".jpg", "_debug.jpg")

    if cropped_path is None:
        print("No valid object found. Skipping classification.")
        log_text += "Object detected: none\n"
        log_text += "Status: no_object_detected\n"
        log_text += "LED: none\n"
        write_log(log_text)
        update_dashboard(
            original_path=image_path,
            debug_path=debug_path if os.path.exists(debug_path) else None,
            result={"status": "no_object_detected", "predicted_class": None, "confidence": None}
        )
        return

    log_text += f"Cropped image: {cropped_path}\n"

    print("Sending cropped image to AI...")

    stop_event, animation_thread = start_loading_animation()

    try:
        result = classify_waste_with_details(cropped_path)
    finally:
        stop_loading_animation(stop_event, animation_thread)

    print("Final result:", result)

    #Logging
    log_text += f"AI raw prediction: {result['raw_prediction']}\n"
    log_text += f"AI mapped to: {result['predicted_class']}\n"
    log_text += f"Top 1: {result['raw_prediction']} ({int(result['confidence']*100)}%)\n"
    log_text += f"Top 2: {result['top2_class']} ({int(result['top2_confidence']*100)}%)\n"
    log_text += f"Margin: {int((result['confidence'] - result['top2_confidence'])*100)}%\n"
    log_text += f"Status: {result['status']}\n"

    update_dashboard(
        original_path=image_path,
        cropped_path=cropped_path,
        debug_path=debug_path if os.path.exists(debug_path) else None,
        result=result
    )

    if result["status"] == "accepted" and result["predicted_class"] in ["plastic", "organic", "glass", "paper"]:
        log_text += f"LED: {result['predicted_class']}\n"
        write_log(log_text)
        show_result(result["predicted_class"])
    else:
        print("No confident or valid prediction.")
        log_text += "LED: none\n"
        write_log(log_text)
        all_off()

print("System ready")

try:
    while True:
        pir.wait_for_motion()
        print("Motion detected")

        try:
            process_motion_event()

        except Exception as e:
            print(f"Runtime error during processing: {e}")
            all_off()

        finally:
            try:
                pir.wait_for_no_motion()
                print("Motion stopped")
            except Exception as e:
                print(f"Error while waiting for motion to stop: {e}")
                all_off()

except KeyboardInterrupt:
    print("Program stopped by user.")
    all_off()

except Exception as e:
    print(f"Fatal system error: {e}")
    all_off()