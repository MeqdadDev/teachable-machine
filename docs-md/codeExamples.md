## Code Examples

### Example 1

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

`result` values are assigned based on the content of `labels.txt` file.
