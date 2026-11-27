# Examples

Each example is a complete program. Their code lives in the
[`examples` folder](https://github.com/MeqdadDev/teachable-machine/tree/main/examples) of the
repository, and this site shows the same files.

## Before you run one

1. [Install the package](../getting-started/installation.md). Webcam examples also need OpenCV.
2. [Export your model](../getting-started/export-model.md) in Keras format.
3. Put the files together:

```text
my-project/
├── keras_model.h5
├── labels.txt
└── app.py        (the example code)
```

Run the program from this folder. In the webcam examples, click the video window and press ++esc++
to quit.

## All examples

| Example | What it shows |
|---|---|
| [Webcam Classifier](webcam.md) | Classify camera frames and show the result on the video |
| [Separate Drawing](separate-drawing.md) | Classify and draw in two steps |
| [Image File](image-file.md) | Classify a photo and act only when confident |
| [Command-Line Webcam](command-line.md) | Choose the model and camera on the command line |
