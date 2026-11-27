"""Classify and draw in two separate steps. Press Esc to quit."""
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

    result = model.classify_image(image_path)
    annotated = model.show_prediction_on_image(image_path, result)

    print(f"{result['class_name']}: {result['class_confidence']:.0%}")

    cv.imshow("Camera", annotated)
    if cv.waitKey(1) == 27:  # Esc
        break

cap.release()
cv.destroyAllWindows()
