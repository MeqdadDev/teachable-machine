## Code Examples

Before running any of the code examples, ensure that the package is installed successfully.
You can find the installation instructions [here](https://meqdaddev.github.io/teachable-machine/how-to-guide/#how-to-install-the-package).

You also need to have an exported model (with `.h5` file extension) with an associated labels text file for the package to use in annotations.

Expected structure before running the code examples:

```
test-directory/
├── keras_model.h5
├── labels.txt
└── app.py (your code example file)
```

### Example 1

In this example for the teachable machine package with OpenCV, we will classify frames coming from the camera view and display 
the classification results on the camera view itself. We use `classify_and_show` to achieve this. See the code example below:

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

    result, resultImage = model.classify_and_show(image_path)

    print("class_index", result["class_index"])

    print("class_name:::", result["class_name"])

    print("class_confidence:", result["class_confidence"])

    print("predictions:", result["predictions"])

    cv.imshow("Video Stream", resultImage)

    k = cv.waitKey(1)
    if k == 27:  # Press ESC to close the camera view
        break
    
cap.release()
cv.destroyAllWindows()
```

The values of `result` are assigned based on the content of the `labels.txt` file.

### Example 2

In this example for the teachable machine package with OpenCV, we will classify frames coming from the camera view and display 
the classification results on the camera view itself, but with separated methods for each task.

We use `classify_image` to classify the captured image but without showing the results on the camera view. Also, we use `show_prediction_on_image` to
get a frame with previous classification results.


Note that the `classify_image` method returns only the prediction results (a Python `dict`), not an image (numpy array or PIL.Image).

See the code example below:

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
    
    resultImage = model.show_prediction_on_image(image_path, result)

    print("class_index", result["class_index"])

    print("class_name:::", result["class_name"])

    print("class_confidence:", result["class_confidence"])

    print("predictions:", result["predictions"])

    cv.imshow("Video Stream", resultImage)

    k = cv.waitKey(1)
    if k == 27:  # Press ESC to close the camera view
        break
    
cap.release()
cv.destroyAllWindows()
```
