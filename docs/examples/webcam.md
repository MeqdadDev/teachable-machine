# Webcam Classifier

Classifies frames from a webcam and shows the result on the video, using `classify_and_show`.

## Code

```python title="examples/webcam.py"
--8<-- "examples/webcam.py"
```

## How it works

- The model is loaded once, before the loop.
- Each frame is saved as `screenshot.jpg`, because the package reads images from files.
- `classify_and_show` returns the [result](../reference/results.md) and an annotated copy of the image.
