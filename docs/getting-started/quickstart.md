# Quick Start

This page takes you from an exported model to live classification on a webcam.

## Before you start

- [Install the package](installation.md).
- [Export your model](export-model.md) in Keras format.
- For the webcam program, install OpenCV: `pip install opencv-python`.

Your folder should look like this:

```text
my-project/
├── keras_model.h5
├── labels.txt
└── app.py
```

## Your first classifier

Save this as `app.py`:

```python
--8<-- "examples/webcam.py"
```

Run it:

```bash
python app.py
```

Show your camera one of the things you trained the model on. The class name and confidence appear
at the bottom of the video. To quit, click the video window and press ++esc++.

## What each part does

| Code | Purpose |
|---|---|
| `TeachableMachine(...)` | Loads the model and its labels once. |
| `cv.VideoCapture(0)` | Opens the first camera. |
| `cv.imwrite(image_path, frame)` | Saves the frame, because the package reads images from files. |
| `model.classify_and_show(image_path)` | Classifies the image and returns the result and an annotated copy. |
| `cv.imshow(...)` | Shows the annotated image. |

## Act on the result

`class_confidence` goes from 0 to 1, so 80% is `0.8`:

```python
result = model.classify_image("photo.jpg")
label = result["class_name"].split(maxsplit=1)[-1]

if label == "Cat" and result["class_confidence"] > 0.8:
    print("There's a cat!")
```

## Next steps

- See every value in [Results](../reference/results.md).
- Learn about [Images and Camera Frames](../guides/images-and-frames.md).
- Try the other [Examples](../examples/index.md).
