# Command-Line Webcam

Reads the model and labels paths from the command line, so one script works with any exported model.

## Code

```python title="examples/command_line_webcam.py"
--8<-- "examples/command_line_webcam.py"
```

## Run it

From your project folder, with the script saved as `command_line_webcam.py`:

```bash
python command_line_webcam.py --model path/to/keras_model.h5 --labels path/to/labels.txt
```

To use an external webcam, add `--camera 1`. Put paths that contain spaces in quotes, such as
`--model "C:\My Models\keras_model.h5"`. Run `python command_line_webcam.py --help` to list all
the options.
