# Teachable Machine

[![Downloads](https://static.pepy.tech/badge/teachable-machine)](https://pepy.tech/project/teachable-machine)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

A Python package to simplify the deployment process of exported [Teachable Machine](https://teachablemachine.withgoogle.com/) models into Python projects.

## Supported Classifiers

**Image Classification**: use exported keras model from Teachable Machine platfrom.

## Requirements

``` Python >= 3.7 ```

## How to install package

```bash
pip install teachable-machine
```

## Dependencies

```bash
numpy
Pillow
tensorflow
```

## Example

An example for teachable machine package with OpenCV:

```python
from teachable_machine import TeachableMachine
import cv2 as cv

cap = cv.VideoCapture(0)
model = TeachableMachine(model_path="keras_model.h5",
                         labels_file_path="labels.txt")

image_path = "screenshot.jpg"

while True:
    _, img = cap.read()
    cv.imwrite(image_path, img)

    result = model.classify_image(image_path)

    print("class_index", result["class_index"])

    print("class_name:::", result["class_name"])

    print("class_confidence:", result["class_confidence"])

    print("predictions:", result["predictions"])

    cv.imshow("Video Stream", img)

    cv.waitKey(1)
```

`class_index` and `class_name`  are assigned based on the content of labels.txt file.

Links:

[PyPI](https://pypi.org/project/teachable-machine/)

[Source Code](https://github.com/MeqdadDev/teachable-machine)
