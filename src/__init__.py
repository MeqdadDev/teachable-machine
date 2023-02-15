from teachable_machine import TeachableMachine


__author__ = 'MeqdadDev'
__version__ = '1.1'

__doc__ = """
Teachable Machine
================
Description
-----------
A Python package to simplify the deployment process of exported Teachable Machine models into different environments.
Teahchable Machine is a tool that was built for Teachable Machine Platform from Google.
Teahchable Machine developed by Eng. Meqdad Darwish.

Source Code is published on GitHub:
https://github.com/MeqdadDev/teachable-machine

Example
-------
>>> from detectors_world import DetectorCreator
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
