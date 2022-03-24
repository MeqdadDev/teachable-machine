# Teachable Machine

A Python package to simplify the deployment of exported Teachable Machine models into different environments like PC, Raspberry Pi and so on.

Link on PyPI: <https://pypi.org/project/teachable-machine/>

## Requirements

Python >= 3.8

## How to install package

```bash
pip install teachable-machine
```

## Dependencies

```numpy, Pillow, tensorflow```

## How to use teachable machine package

```py
from teachable_machine import TeachableMachine

my_model = TeachableMachine(model_path='keras_model.h5', deployment_env='pc')

img_path = 'images/my_image.jpg'

highest_prediction, predictions = my_model.classify_on_pc(img_path)
```

More features are coming...
