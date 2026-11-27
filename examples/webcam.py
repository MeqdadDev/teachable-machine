"""Classify webcam frames and show the result on the video. Press Esc to quit."""
import cv2 as cv

from teachable_machine import TeachableMachine

cap = cv.VideoCapture(0)
model = TeachableMachine(model_path="keras_model.h5", labels_file_path="labels.txt")

image_path = "screenshot.jpg"

while True:
    ok, frame = cap.read()
    if not ok:
        break

    cv.imwrite(image_path, frame)  # the package reads images from files

    result, annotated = model.classify_and_show(image_path)

    print("class_index:", result["class_index"])
    print("class_name:", result["class_name"])
    print("class_confidence:", result["class_confidence"])
    print("predictions:", result["predictions"])

    cv.imshow("Camera", annotated)
    if cv.waitKey(1) == 27:  # Esc
        break

cap.release()
cv.destroyAllWindows()
