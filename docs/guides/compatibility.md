# TensorFlow Compatibility

TensorFlow 2.16 and later include Keras 3, which can't load some of Teachable Machine's exported
`.h5` models by itself. Since version 1.3.1, the package handles this for you, so there's no need
to pin an old TensorFlow version.

## The `groups` argument

Exported models contain an old `DepthwiseConv2D` layer setting that Keras 3 rejects:

```text
TypeError: Unrecognized keyword arguments passed to DepthwiseConv2D: {'groups': 1}
```

The package loads the layer without that unused setting. See
[issue #2](https://github.com/MeqdadDev/teachable-machine/issues/2) for the background.

## Nested models

Some exports save a `Sequential` model that wraps other `Sequential` or `Functional` models.
Keras 3's `.h5` loader rebuilds this shape incorrectly. Before 1.3.1 this showed up as a
misleading `FileNotFoundError: Model file not found`. The package now rebuilds such models and
restores their weights.

## Drawing on Windows and recent Pillow

Version 1.3.1 also fixed crashes in `show_prediction_on_image` on Windows and with recent
versions of Pillow.

## Check your version

```bash
pip show teachable-machine
pip install --upgrade teachable-machine
```

See the [Changelog](../project/changelog.md) for details.
