# Teachable Machine

A Python package to simplify the deployment process of exported [Teachable Machine](https://teachablemachine.withgoogle.com/) models into different environments like Windows, Linux and MAC.

Links:

[PyPI](https://pypi.org/project/teachable-machine/)

[Source Code](https://github.com/MeqdadDev/teachable-machine)

## Supported Tools in Teachable Machine

Image Classification using exported keras model from Teachable Machine platfrom.

Next tool in the package: **Pose Classification**

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

my_model = TeachableMachine(model_path='keras_model.h5', model_type='h5')

img_path = 'images/my_image.jpg'

result = my_model.classify_image(img_path)

print('highest_class_id:', result['highest_class_id'])
print('all_predictions:', result['all_predictions'])
```

_highest_class_id_ is selected based on labels.txt file.

More features are coming soon...
