from teachable_machine import TeachableMachine


__author__ = "MeqdadDev"
__version__ = "1.2"

__doc__ = """
Teachable Machine
================
Description
-----------
A Python package designed to simplify the integration of exported models from Google's Teachable Machine platform into various environments.
This tool was specifically crafted to work seamlessly with Teachable Machine, making it easier to implement and use your trained models.

Developed by: Meqdad Darwish

Source Code is published on GitHub:
https://github.com/MeqdadDev/teachable-machine

Example
-------
>>> from teachable_machine import TeachableMachine
>>> import cv2 as cv

>>> cap = cv.VideoCapture(0)
>>> model = TeachableMachine(model_path="keras_model.h5",
>>>                          labels_file_path="labels.txt")

>>> image_path = "screenshot.jpg"

>>> while True:
>>>     _, img = cap.read()
>>>     cv.imwrite(image_path, img)

>>>     result = model.classify_image(image_path)

>>>     print("class_index", result["class_index"])

>>>     print("class_name:::", result["class_name"])

>>>     print("class_confidence:", result["class_confidence"])

>>>     print("predictions:", result["predictions"])

>>>     cv.imshow("Video Stream", img)

>>>     cv.waitKey(1)

References
----------
* https://github.com/MeqdadDev
"""
