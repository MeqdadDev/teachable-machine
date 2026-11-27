# Images and Camera Frames

## What you can pass in

`classify_image`, `show_prediction_on_image` and `classify_and_show` take the **path of an image
file**:

```python
result = model.classify_image("photo.jpg")
```

They don't accept NumPy arrays (such as OpenCV frames) or PIL images: passing one raises
`TypeError: Unsupported image type`.

## Classify camera frames

Save each frame to a file, then pass its path:

```python
ok, frame = cap.read()
if ok:
    cv.imwrite("screenshot.jpg", frame)
    result = model.classify_image("screenshot.jpg")
```

!!! tip

    [Teachable Machine Lite](https://meqdaddev.github.io/teachable-machine-lite/) takes camera
    frames directly, without saving them. It uses the TensorFlow Lite export of your model.

## How images are prepared

The package converts the image to RGB, crops it around the centre to a square, and resizes it to
224 × 224 pixels. Keep the object you want to classify near the middle of the picture.

## Get the annotated image

`classify_and_show` returns the result and the image with the class name and confidence drawn in
the bottom left corner. The original file isn't changed.

```python
result, annotated = model.classify_and_show("photo.jpg")
```

The annotated image is a **BGR NumPy array**, ready for OpenCV:

```python
cv.imwrite("labelled.jpg", annotated)
```

To get a PIL image in RGB instead, pass `convert_to_bgr=False`:

```python
result, picture = model.classify_and_show("photo.jpg", convert_to_bgr=False)
picture.save("labelled.jpg")
```

If you already have a result, `show_prediction_on_image` draws it without classifying again:

```python
result = model.classify_image("photo.jpg")
annotated = model.show_prediction_on_image("photo.jpg", result)
```
