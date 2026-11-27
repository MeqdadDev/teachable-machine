# Export Your Model

The package runs **image classification** models exported from Google's Teachable Machine in
Keras format.

## Steps

1. Train your image model on [Teachable Machine](https://teachablemachine.withgoogle.com/).
2. Click **Export Model**, then open the **TensorFlow** tab.
3. Choose **Keras** as the model conversion type.
4. Download and unzip. Put `keras_model.h5` and `labels.txt` next to your code.

!!! note

    Check the file names in your download. If they differ, pass them as `model_path` and
    `labels_file_path`.

## The labels file

Each line of `labels.txt` holds a class number and a class name:

```text
0 Cat
1 Dog
```

The package keeps the whole line, so `result["class_name"]` is `"0 Cat"`. To get only the name:

```python
label = result["class_name"].split(maxsplit=1)[-1]  # "Cat"
```

Always use the `labels.txt` that came in the same download as the model, with its lines in their
original order.

!!! warning "Class names that aren't in English, on Windows"

    Names in Arabic and other non-Latin scripts can come out garbled on Windows. See
    [Troubleshooting](../guides/troubleshooting.md#garbled-class-names-on-windows) for the fix.
