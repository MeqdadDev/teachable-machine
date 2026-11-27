# Installation

## Install the package { #how-to-install-the-package }

```bash
pip install teachable-machine
```

This also installs TensorFlow, which is large, so the first installation takes a while.

!!! tip "Use a virtual environment"

    It keeps TensorFlow and its dependencies separate from your other projects:

    === "Windows"

        ```powershell
        python -m venv .venv
        .venv\Scripts\activate
        pip install teachable-machine
        ```

    === "Linux and macOS"

        ```bash
        python -m venv .venv
        source .venv/bin/activate
        pip install teachable-machine
        ```

The webcam examples also need OpenCV:

```bash
pip install opencv-python
```

## Check the installation

```bash
python -c "from teachable_machine import TeachableMachine; print('Ready')"
```

TensorFlow may print some informational messages first.

## Next step

[Export your model](export-model.md), then follow the [Quick Start](quickstart.md).
