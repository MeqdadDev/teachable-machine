"""Classify one image file and act only when the model is confident enough."""
from teachable_machine import TeachableMachine

model = TeachableMachine(model_path="keras_model.h5", labels_file_path="labels.txt")

result = model.classify_image("photo.jpg")

# "0 Cat" -> "Cat"
label = result["class_name"].split(maxsplit=1)[-1]
confidence = float(result["class_confidence"])  # from 0 to 1

if confidence > 0.8:
    print("Detected:", label)
else:
    print(f"Not sure. Top guess: {label} ({confidence:.0%})")
