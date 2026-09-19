import sys
import cv2
import time
from ultralytics import YOLO

sys.path.insert(0, "src")
from speech.tts import Speaker

#model = YOLO("models/yolov8n.pt")
#model = YOLO("models/yolov8s-oiv7.pt")
model = YOLO("models/yolo26n.pt")
speaker = Speaker()

camera = cv2.VideoCapture(0)

FRAMES_BEFORE_GONE = 50

currently_present = set()
missing_counts = {}

previous_time = 0

while True:
    success, frame = camera.read()
    if not success:
        break

    results = model.predict(source=frame, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()

    detected_now = set()
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        label = results[0].names[class_id]
        detected_now.add(label)

    new_objects = detected_now - currently_present

    for label in new_objects:
        currently_present.add(label)
        print(f"ANNOUNCING: {label}")
        speaker.speak(label)

    for label in detected_now:
        missing_counts[label] = 0

    for label in list(currently_present):
        if label not in detected_now:
            missing_counts[label] = missing_counts.get(label, 0) + 1
            if missing_counts[label] >= FRAMES_BEFORE_GONE:
                currently_present.remove(label)
                missing_counts.pop(label, None)

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    for _ in range(5):
        speaker.update()
   
    cv2.imshow("Detection with Speech", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
speaker.shutdown()