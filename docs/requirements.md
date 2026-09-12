## Requirements

### Python Version
``` Python >= 3.9 ```

### Dependencies

```bash
numpy
Pillow
tensorflow>=2.16
```

## Compatibility with recent TensorFlow/Keras releases

Teachable Machine's exported `.h5` models embed a legacy `DepthwiseConv2D` layer config that current Keras (Keras 3, bundled by default since TensorFlow 2.16) rejects with an error such as:

```
TypeError: Unrecognized keyword arguments passed to DepthwiseConv2D: {'groups': 1}
```

Since `v1.3.1`, this package patches the model loader so exported models load correctly on up-to-date TensorFlow/Keras installs, with no need to pin an old TensorFlow version. See [issue #2](https://github.com/MeqdadDev/teachable-machine/issues/2) for background.
