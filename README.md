# Real-Time Multi-Person & Object Detection on Raspberry Pi 4

Real-time computer vision system built on a Raspberry Pi 4 that detects and identifies multiple people (via face recognition) and objects (via TensorFlow Lite) simultaneously, evaluated across three lighting conditions: ambient indoor, flashlight-lit dark room, and low-light outdoor (~9 lux).

Built as a module project for **CS6461 – Computer Vision Systems**, University of Limerick.

## Overview

- **Face recognition:** HOG-based 128D face encodings (`face_recognition` library), trained on custom headshot dataset
- **Object detection:** TensorFlow Lite, COCO-labelled classes
- **Hardware:** Raspberry Pi 4, Pi Camera Module (OV5647/IMX219)
- **Dual virtual environments** to isolate face-rec and TFLite dependencies
- Frame rate tuned to 3.5 FPS to manage Pi's CPU-only inference load

## Results Summary

| Scenario | Lighting | Avg. Confidence | Notes |
|---|---|---|---|
| Ambient indoor | 200–250 lux | 49–50% | No false negatives; best overall performance |
| Flashlight (dark room) | ~35 lux (beam) | 58–63% | Reliable within beam; one person misidentified as "unknown" on swap |
| Outdoor evening | ~9 lux | Lower, variable | Only 2–3 of 4 people consistently identified; ID card misclassified as cell phone |

Full methodology, confidence-score graphs, and discussion of a detected bias against darker complexions in low light (and the retraining done to address it) are in the [project report](./CS6461_Project_Report.pdf).

## Repo Contents

```
├── capture_headshots.py     # Script to build the training dataset (15-20 images/person)
├── CS6461_Project_Report.pdf # Full write-up: methodology, results, ethics discussion
└── videos/                   # Demo footage from all three lighting scenarios
    ├── ambientlight2.mp4
    ├── flashlight indarkroom.mp4
    └── outdoorlighting.mp4
```

> Videos include footage of course participants who have consented to this footage being shared publicly.

## Key Findings

- High ambient lighting gave the most stable detection with no false negatives
- Harsh directional lighting (flashlight) introduced shadows/specular highlights that disrupted HOG feature extraction, causing occasional misidentification
- Low-light outdoor conditions were the hardest case — confirming a known weakness of HOG-based face recognition in low signal-to-noise conditions
- Retraining with darker-complexion samples reduced (but did not eliminate) a detection bias observed in early low-light testing

## Ethical Considerations

This is an academic prototype, not production software. Any real-world deployment of similar continuous face/object tracking would need explicit consent, GDPR-compliant data handling, and rigorous bias auditing — see the report's Ethical Implications section for full discussion.

## Author

**Purnima** — MEng Computer Vision & AI, University of Limerick
