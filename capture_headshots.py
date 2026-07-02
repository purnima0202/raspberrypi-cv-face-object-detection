import os
import time
import pickle
from threading import Thread
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import tflite_runtime.interpreter as tflite  # TFLite runtime used in Lab-5 for efficient Pi inference (Vignesh, Lab-5, Sec. 3)
import face_recognition  # HOG-based face pipeline from Lab-3 (Vignesh, Lab-3, Sec. 4-6)
from picamera2 import Picamera2  # PiCamera2 interface from Lab-2 (Vignesh, Lab-2, Sec. 4)

PERSON_NAME = 'Jacob'  # Default person name for headshot capture (Lab-3 style dataset naming)
DATASET_DIR = 'dataset'  # Root folder for per-person subdirectories (Vignesh, Lab-3, Sec. 5)
ENC_FILE = 'encodings.pickle'  # Trained encodings file produced by model_training script (Lab-3)
MODEL = 'ssd_mobilenet_v1.tflite'  # TFLite model chosen in Lab-5 for speed on Raspberry Pi (Vignesh, Lab-5, Sec. 7)
LABELS = 'models/labels.txt'  # COCO labels file used with SSD MobileNet (Lab-5, Sec. 4)
MIN_CONFID = 0.5  # Minimum confidence threshold for object detection, as in Lab-5 examples
RUN_FOR = 20  # Run duration (seconds) to match project requirement of a 20s observation window
LOG_INTERVAL = 0.25  # Log detection data every 0.25s for fine-grained temporal analysis (more detail than 1s logs)

# ---------------- camera thread ----------------
class CameraThread:
    def __init__(self, width=640, height=480):
        self.picam2 = Picamera2()
        config = self.picam2.create_still_configuration(main={"format": "RGB888", "size": (width, height)})
        # RGB888 + 640x480: same resolution as labs, balances speed and image quality (Vignesh, Lab-2, Sec. 6)
        self.picam2.configure(config)
        self.picam2.start()
        self.stopped = False
        self.frame = None
        self.thread = Thread(target=self.update, daemon=True)  # Separate thread so detection loop is not blocked by capture

    def start(self):
        self.thread.start()
        return self

    def update(self):
        # Continuously grab the latest frame in the background for smoother real-time processing
        while not self.stopped:
            self.frame = self.picam2.capture_array()

    def read(self):
        # Main thread calls this to fetch the most recent frame
        return self.frame

    def stop(self):
        self.stopped = True
        self.picam2.close()  # Cleanly close the camera as recommended in Lab-2

# ---------------- face & object helpers ----------------
class FaceRecC:
    def __init__(self, encfile=ENC_FILE):
        if not os.path.exists(encfile):
            # Force user to train encodings beforehand (same assumption as Lab-3 face_rec scripts)
            raise FileNotFoundError("Run model_training to create encodings.pickle")
        with open(encfile, 'rb') as f:
            d = pickle.load(f)
        self.known_enc = d["encodings"]  # 128-D face embeddings
        self.known_names = d["names"]    # Corresponding labels

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # face_recognition expects RGB input (Vignesh, Lab-3, Sec. 4)
        locs = face_recognition.face_locations(rgb)   # HOG-based face detector
        encs = face_recognition.face_encodings(rgb, locs)  # Compute encodings for each detected face
        out = []
        for loc, enc in zip(locs, encs):
            if len(self.known_enc) == 0:
                # Safety path if no encodings are loaded
                out.append(("Unknown", 0.0, loc))
                continue
            dists = face_recognition.face_distance(self.known_enc, enc)  # Smaller distance = closer match (Lab-4, Sec. 3.3)
            idx = int(np.argmin(dists))
            conf = float(1.0 - dists[idx])  # Simple confidence proxy: 1 - distance (used in Lab-4 analysis)
            # 0.48 threshold chosen from experiments and Lab-3 guidance to separate matches/non-matches
            name = self.known_names[idx] if conf > 0.48 else "Unknown"
            out.append((name, conf, loc))
        return out

class ObjectDetC:
    def __init__(self, model=MODEL, labels=LABELS):
        self.interp = tflite.Interpreter(model_path=model)  # Lightweight TFLite interpreter for Pi (Lab-5, Sec. 3)
        self.interp.allocate_tensors()
        self.id_in = self.interp.get_input_details()
        self.id_out = self.interp.get_output_details()
        self.h = self.id_in[0]["shape"][1]  # Model input height
        self.w = self.id_in[0]["shape"][2]  # Model input width
        with open(labels, "r") as f:
            self.labels = [l.strip() for l in f.readlines()]  # Load COCO class names

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.w, self.h))  # Resize frame to model input size (Lab-5, Sec. 5)
        inpt = np.expand_dims(resized, axis=0)
        if self.id_in[0]["dtype"] == np.float32:
            # Normalization to [-1, 1] as in TFLite SSD example (Vignesh, Lab-5, Sec. 6.3)
            inpt = (np.float32(inpt) - 127.5) / 127.5
        self.interp.set_tensor(self.id_in[0]["index"], inpt)
        self.interp.invoke()  # Run forward pass on TFLite model
        boxes = self.interp.get_tensor(self.id_out[0]["index"])[0]
        classes = self.interp.get_tensor(self.id_out[1]["index"])[0]
        scores = self.interp.get_tensor(self.id_out[2]["index"])[0]
        dets = []
        h0, w0 = frame.shape[:2]
        for i, s in enumerate(scores):
            if s >= MIN_CONFID:
                # Convert normalized box coords back to original frame size
                ymin, xmin, ymax, xmax = boxes[i]
                dets.append(
                    (
                        self.labels[int(classes[i])],  # Class label
                        float(s),                      # Confidence score
                        (int(xmin * w0), int(ymin * h0), int(xmax * w0), int(ymax * h0)),
                    )
                )
        return dets

# ---------------- headshot capture ----------------
def capture_headshots_c(name=PERSON_NAME, target=20):
    # Simple headshot capture tool following Lab-3 dataset collection approach (Vignesh, Lab-3, Sec. 5)
    folder = os.path.join(DATASET_DIR, name)
    os.makedirs(folder, exist_ok=True)
    cam = Picamera2()
    cam.configure(cam.create_video_configuration(main={"format": "RGB888", "size": (640, 480)}))
    cam.start()
    count = 0
    print("[C] Capture headshots. SPACE to save, q to quit.")
    while True:
        frame = cam.capture_array()
        cv2.imshow("C Headshots", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" "):
            # Each press saves one labelled training image for this person
            count += 1
            fname = f"{name}_{count:03d}.jpg"
            cv2.imwrite(os.path.join(folder, fname), frame)
            print(f"Saved {fname}")
            if count >= target:  # Stop at target images (20) for a balanced per-person dataset
                break
        elif k == ord("q"):
            # Allow early exit if needed
            break
    cam.close()
    cv2.destroyAllWindows()

# ---------------- main session ----------------
def run_session_c():
    face = FaceRecC()  # Load trained face encodings (person identities)
    obj = ObjectDetC()  # Load SSD MobileNet object detector (for non-person objects)
    cam = CameraThread(width=640, height=480)  # Use threaded camera to avoid I/O bottlenecks
    cam.start()
    time.sleep(0.5)  # Small delay to let camera warm up and start streaming frames

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # Very low FPS here (0.2) was used during experimentation to get ~2025s videos for repeated testing
    out = cv2.VideoWriter("output_video.mp4", fourcc, 0.2, (640, 480))

    start = time.time()
    frame_count = 30  # Start from 30: helps track total frames written during experiments
    last_log_time = 0

    face_conf = defaultdict(list)  # Store (time, confidence) per person for later plotting and tables
    obj_conf = defaultdict(list)   # Store (time, confidence) per object
    presence = defaultdict(list)   # Raw timestamps where each person was seen (for presence intervals)

    try:
        while time.time() - start < RUN_FOR:
            # Main loop runs for exactly RUN_FOR seconds (20s project window)
            frame = cam.read()
            if frame is None:
                # If camera hasnt produced a frame yet, skip this iteration
                continue

            current_time = time.time()
            tnow = current_time - start  # Time since session start

            if current_time - last_log_time >= LOG_INTERVAL:
                # Log only every LOG_INTERVAL seconds to control log size and avoid redundant entries
                last_log_time = current_time

                faces = face.detect(frame)
                for name, conf, loc in faces:
                    face_conf[name].append((tnow, conf))  # Log temporal confidence trace
                    presence[name].append(tnow)           # Mark that this person is present at time tnow

                dets = obj.detect(frame)
                for label, score, _ in dets:
                    if label.lower() == "person":
                        # Skip 'person' from object detector to avoid double-counting with face_recognition
                        continue
                    obj_conf[label].append((tnow, score))  # Log non-person objects (e.g., bottle, cell phone)

            # Still draw detections every frame so the saved video has full overlays
            faces = face.detect(frame)
            for name, conf, loc in faces:
                top, right, bottom, left = loc
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 128, 255), 2)
                # Overlay label + confidence for qualitative inspection in the video (Lab-3 visualisation style)
                cv2.putText(
                    frame,
                    f"{name} {conf:.2f}",
                    (left, top - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 128, 255),
                    2,
                )

            dets = obj.detect(frame)
            for label, score, b in dets:
                if label.lower() == "person":
                    continue
                x1, y1, x2, y2 = b
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw object label and score as in Lab-5 examples
                cv2.putText(
                    frame,
                    f"{label} {score:.2f}",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            # Always show current session time on the frame for easier analysis in the saved video
            cv2.putText(
                frame,
                f"t={tnow:.2f}s",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            out.write(frame)  # Save frame to MP4 file for later review
            frame_count += 1

            cv2.imshow("Script C", frame)  # Live preview window
            if cv2.waitKey(1) & 0xFF == ord("q"):
                # Allow manual early stop during testing
                break

    finally:
        # Print how many frames we actually wrote to compare against expected values
        print(f"Total frames written: {frame_count} (should be ~200 for 20s at 10 FPS)")
        cam.stop()
        out.release()
        cv2.destroyAllWindows()

    # Build presence intervals
    intervals = {}
    for name, ts in presence.items():
        if not ts:
            intervals[name] = []
            continue
        ts_sorted = sorted(ts)
        merged = []
        s = ts_sorted[0]
        e = ts_sorted[0]
        for t in ts_sorted[1:]:
            if t - e <= 0.7:
                # If next detection is within 0.7s, treat it as the same continuous presence interval
                # This hysteresis avoids splitting presence due to brief missed frames
                e = t
            else:
                merged.append((s, e))
                s = t
                e = t
        merged.append((s, e))
        intervals[name] = merged  # Final list of (start, end) intervals per person/object

    # Save logs in Markdown table format for direct inclusion in the report (copy-paste friendly)
    out_log = f"log_scriptC_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(out_log, "w") as f:
        f.write("### Face Confidences\n")
        f.write("| Person Name | Time (s) | Confidence |\n")
        f.write("|-------------|----------|------------|\n")
        for person, records in face_conf.items():
            for t, c in records:
                f.write(f"| {person} | {t:.2f} | {c:.2f} |\n")

        f.write("\n### Object Confidences\n")
        f.write("| Object | Time (s) | Confidence |\n")
        f.write("|--------|----------|------------|\n")
        for obj, records in obj_conf.items():
            for t, c in records:
                f.write(f"| {obj} | {t:.2f} | {c:.2f} |\n")

        f.write("\n### Presence Intervals\n")
        f.write("| Person/Object | Start Time (s) | End Time (s) |\n")
        f.write("|---------------|----------------|--------------|\n")
        for entity, ranges in intervals.items():
            for start_t, end_t in ranges:
                f.write(f"| {entity} | {start_t:.2f} | {end_t:.2f} |\n")
    print(f"[C] Log saved to {out_log}")

    # Print a small sample of logs to the terminal for quick sanity-check
    print("\nFace Confidence Samples:")
    print("| Person Name | Time (s) | Confidence |")
    print("|-------------|----------|------------|")
    for person in sorted(face_conf.keys()):
        sample = face_conf[person][:5]
        for t, c in sample:
            print(f"| {person} | {t:.2f} | {c:.2f} |")

    print("\nObject Confidence Samples:")
    print("| Object | Time (s) | Confidence |")
    print("|--------|----------|------------|")
    for obj in sorted(obj_conf.keys()):
        sample = obj_conf[obj][:5]
        for t, c in sample:
            print(f"| {obj} | {t:.2f} | {c:.2f} |")

    print("\nPresence Intervals:")
    print("| Person/Object | Start (s) | End (s) |")
    print("|---------------|-----------|---------|")
    for entity, ranges in intervals.items():
        for start_t, end_t in ranges:
            print(f"| {entity} | {start_t:.2f} | {end_t:.2f} |")

    # Plotting confidence and presence intervals as required for analysis in the report (Vignesh, Lab-6, Sec. 5)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    top_names = sorted(face_conf.keys(), key=lambda n: -len(face_conf[n]))[:4]
    for name in top_names:
        times = [t for t, c in face_conf[name]]
        confs = [c for t, c in face_conf[name]]
        if times:
            ax[0].scatter(times, confs, s=15, label=name)  # Scatter plot of confidence vs time for main people
    ax[0].set_title("Face confidence scatter (Script C)")
    ax[0].set_ylabel("Confidence")
    ax[0].legend()
    ax[0].grid(True)

    for i, (name, ints) in enumerate(intervals.items()):
        for s, e in ints:
            # Draw thick horizontal bars for each presence interval
            ax[1].plot([s, e], [i, i], linewidth=10, alpha=0.6)
            ax[1].scatter([s, e], [i, i], color="red", s=30)  # Mark start and end points
    ax[1].set_yticks(list(range(len(intervals))))
    ax[1].set_yticklabels(list(intervals.keys()))
    ax[1].set_title("Presence intervals (Script C)")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_xlim(0, RUN_FOR)
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Script C - Threaded camera + different plotting")  # Entry point label
    print("1) Capture Headshots (dataset/PERSON_NAME)")      # Option to build dataset (Lab-3 workflow)
    print("2) Run detection & generate plots (20s)")         # Option to run full experiment
    choice = input("Select 1 or 2: ").strip()
    if choice == "1":
        name = input("Name (enter for default): ").strip() or PERSON_NAME  # Allow custom dataset name
        capture_headshots_c(name)
    elif choice == "2":
        run_session_c()
    else:
        print("Exiting")  # Graceful exit if input is not 1 or 2
