import cv2
import time
from ultralytics import YOLO

model = YOLO("models/yolov8n.pt")

camera = cv2.VideoCapture(0)

previous_time = 0

while True:
    success, frame = camera.read()
    if not success:
        break

    results = model.predict(source=frame, conf=0.7, verbose=False)
    annotated_frame = results[0].plot()

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Detection Test", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()