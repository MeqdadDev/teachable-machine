# Changelog

All notable changes to this project are documented in this file.

## [1.3.0]

### Fixed

- **TensorFlow/Keras 3 compatibility**: Teachable Machine's exported `.h5` models embed a legacy `DepthwiseConv2D` layer config (`'groups': 1`) that current Keras (Keras 3, bundled by default since TensorFlow 2.16) rejects during deserialization, causing model loading to fail with `TypeError`/`ValueError: Unrecognized keyword arguments passed to DepthwiseConv2D: {'groups': 1}`. `TeachableMachine` now loads models with a compatibility shim that discards the unused key, so exported models work on up-to-date TensorFlow/Keras installs without pinning an old TensorFlow version. Fixes [#2](https://github.com/MeqdadDev/teachable-machine/issues/2).

### Changed

- `install_requires` now specifies `tensorflow>=2.16` (previously unpinned, which could silently resolve to a broken combination).
- `python_requires` raised to `>=3.9` to match current TensorFlow's supported Python range; removed unsupported 3.7/3.8 classifiers.
- `requirements.txt` (dev/test) updated to current dependency versions and now includes `pytest`, `pytest-mock`, and `h5py`.

### Added

- Regression tests covering the legacy `DepthwiseConv2D` config shape, including an end-to-end test that saves a model, tampers its config to reproduce Teachable Machine's legacy export, and verifies it still loads and classifies correctly.
- CI workflow to run the test suite on pushes/PRs, and a release workflow to publish to PyPI on tagged GitHub releases.

## [1.2.1] - 2024-08-18

- Added `setup.py` for packaging.

## [1.2.0]

- Added package logo, expanded documentation (how-to guide, requirements, code examples), and unit tests.

## [1.1] and [1.0]

- Initial releases of the `TeachableMachine` class for image classification with exported Keras models.
