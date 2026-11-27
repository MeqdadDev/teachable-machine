"""Classify webcam frames with any exported model, chosen on the command line."""
import argparse

import cv2 as cv

from teachable_machine import TeachableMachine

parser = argparse.ArgumentParser(
    description="Classify webcam frames with a Teachable Machine model.")
parser.add_argument("--model", default="keras_model.h5", help="path to the .h5 model file")
parser.add_argument("--labels", default="labels.txt", help="path to the labels.txt file")
parser.add_argument("--camera", type=int, default=0,
                    help="camera index (try 1 for an external webcam)")
args = parser.parse_args()

model = TeachableMachine(model_path=args.model, labels_file_path=args.labels)

cap = cv.VideoCapture(args.camera)
if not cap.isOpened():
    raise SystemExit(f"Could not open camera {args.camera}.")

image_path = "screenshot.jpg"

while True:
    ok, frame = cap.read()
    if not ok:
        break

    cv.imwrite(image_path, frame)  # the package reads images from files

    result, annotated = model.classify_and_show(image_path)
    print(f"{result['class_name']}: {result['class_confidence']:.0%}")

    cv.imshow("Camera", annotated)
    if cv.waitKey(1) == 27:  # Esc
        break

cap.release()
cv.destroyAllWindows()
