# Troubleshooting

## `OSError` or `FileNotFoundError` when loading

The package raises `Error loading model`, `Model file not found` or `Error loading labels`.

- Check `model_path` and `labels_file_path`. Relative paths start from the folder you run the
  program in, which isn't always the folder the script is in.
- Use the `.h5` file from the **TensorFlow** → **Keras** export, and make sure the download is
  complete. A `.tflite` file needs
  [Teachable Machine Lite](https://meqdaddev.github.io/teachable-machine-lite/) instead.
- [Upgrade the package](compatibility.md#check-your-version).

To handle all of these in code, catch `OSError`.

## `DepthwiseConv2D: {'groups': 1}`

You have a version older than 1.3.1. [Upgrade the package](compatibility.md#check-your-version).

## `ValueError: Unsupported model type`

`model_type` must be `"h5"` (the default) or `"keras"`.

## `TypeError: Unsupported image type`

You passed something other than the path of an image the package can open. For a camera frame,
[save it to a file first](images-and-frames.md#classify-camera-frames).

## `FileNotFoundError: Image file not found`

Check the image path.

## `class_name` starts with a number

That's expected: the package keeps the whole line from `labels.txt`. See
[The labels file](../getting-started/export-model.md#the-labels-file).

## Garbled class names on Windows

This is fixed in version 1.3.3, which always reads `labels.txt` as UTF-8.
[Upgrade the package](compatibility.md#check-your-version).

With an earlier version, turn on Python's UTF-8 mode before running your program.

=== "PowerShell"

    ```powershell
    $env:PYTHONUTF8 = "1"
    python app.py
    ```

=== "Command Prompt"

    ```bat
    set PYTHONUTF8=1
    python app.py
    ```

## `Labels file not found`, but the file is there

Since version 1.3.3, the labels file must be UTF-8. The `labels.txt` exported by Teachable Machine
already is. If you edited it and saved it in another encoding, save it again as UTF-8.

## `json.dumps` fails on the result

The result contains NumPy values. [Convert them first](../reference/results.md#convert-to-json).

## Progress bars fill the console

Keras prints a progress bar each time the model classifies an image. To turn them off, add this
before your loop:

```python
import keras

keras.config.disable_interactive_logging()
```

## Messages about GPU support or oneDNN

These come from TensorFlow, not from this package.

## The camera doesn't open

Try another camera index, such as `cv.VideoCapture(1)`, and close other programs that use the
camera.

## Still stuck?

[Open an issue](https://github.com/MeqdadDev/teachable-machine/issues) with the full error
message, your system, and your Python and TensorFlow versions.
