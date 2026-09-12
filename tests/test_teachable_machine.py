import json
from unittest import mock

import pytest
from src.teachable_machine import TeachableMachine, _CompatDepthwiseConv2D
from PIL import Image, ImageFont
import numpy as np


@pytest.fixture
def teachable_machine(mocker):
    """Return a TeachableMachine instance with load_model/open mocked out."""
    # Create a TeachableMachine instance
    mocker.patch("src.teachable_machine.load_model", return_value="mock_model")

    # Mock the open function to simulate reading from the labels.txt file with the specified content
    mocker.patch("builtins.open", mocker.mock_open(read_data="0 Class A\n1 Class B\n"))

    # Initialize TeachableMachine
    return TeachableMachine(
        model_path="dummy_model_path.h5", labels_file_path="dummy_labels.txt"
    )


def test_teachable_machine_initialization(mocker):
    """
    Test the initialization of the TeachableMachine class and loading of the model and labels.
    """
    # Mock the load_model and _load_labels methods to avoid actual file operations
    mocker.patch("src.teachable_machine.load_model", return_value="mock_model")

    # Mock the open function to simulate reading from the labels.txt file with the specified content
    mocker.patch("builtins.open", mocker.mock_open(read_data="0 Class A\n1 Class B\n"))

    # Initialize TeachableMachine
    tm = TeachableMachine(
        model_path="dummy_model_path.h5", labels_file_path="dummy_labels.txt"
    )

    # Check if the model is loaded correctly
    assert tm._model == "mock_model"

    # Check if labels are loaded and formatted correctly
    assert tm._labels == ["0 Class A\n", "1 Class B\n"]

    # Check if the model type is correctly set
    assert tm._model_type == "h5"

    # Test that the initialization does not raise any exceptions
    assert isinstance(tm, TeachableMachine)


def test_open_image(mocker):
    """
    Test the _open_image method of the TeachableMachine class.
    """
    # Create a TeachableMachine instance
    mocker.patch("src.teachable_machine.load_model", return_value="mock_model")

    # Mock the open function to simulate reading from the labels.txt file with the specified content
    mocker.patch("builtins.open", mocker.mock_open(read_data="0 Class A\n1 Class B\n"))

    # Initialize TeachableMachine
    tm = TeachableMachine(
        model_path="dummy_model_path.h5", labels_file_path="dummy_labels.txt"
    )

    # Mock PIL.Image.open and the convert method
    mock_image = mocker.Mock(spec=Image.Image)
    mock_image.convert.return_value = mock_image
    mock_open = mocker.patch("PIL.Image.open", return_value=mock_image)

    # Test successful image opening
    result = tm._open_image("dummy_image.jpg")
    assert result == mock_image
    mock_open.assert_called_once_with("dummy_image.jpg")
    mock_image.convert.assert_called_once_with("RGB")

    # Test FileNotFoundError
    mock_open.side_effect = FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Image file not found"):
        tm._open_image("non_existent_image.jpg")

    # Test other exceptions (simulating unsupported image type)
    mock_open.side_effect = Exception
    with pytest.raises(TypeError, match="Unsupported image type"):
        tm._open_image("invalid_image.txt")

    # Test that error messages are printed
    mock_print = mocker.patch("builtins.print")

    mock_open.side_effect = FileNotFoundError
    with pytest.raises(FileNotFoundError):
        tm._open_image("non_existent_image.jpg")
    mock_print.assert_called_with("ImageNotFound: Error in image file.")

    mock_open.side_effect = Exception
    with pytest.raises(TypeError):
        tm._open_image("invalid_image.txt")
    mock_print.assert_called_with(
        "ImageTypeError: Error while opening or converting image"
    )


def test_preprocess_image_output_shape(teachable_machine):
    """_preprocess_image should resize any input image to the model's fixed input shape."""
    # Create a sample image
    sample_image = Image.new("RGB", (300, 200))

    # Process the image
    processed = teachable_machine._preprocess_image(sample_image)

    # Check the output shape
    assert processed.shape == (1, 224, 224, 3)


def test_preprocess_image_normalization(teachable_machine):
    """_preprocess_image should scale pixel values into the [-1, 1] range."""
    # Create a sample image with known values
    sample_array = np.full((300, 200, 3), 127, dtype=np.uint8)
    sample_image = Image.fromarray(sample_array)

    # Process the image
    processed = teachable_machine._preprocess_image(sample_image)

    # Check if the values are normalized correctly (close to -0.00392, not exactly 0)
    expected_value = (127 / 127.5) - 1
    assert np.allclose(processed, expected_value, atol=1e-6)


def test_preprocess_image_different_sizes(teachable_machine):
    """_preprocess_image should normalize inputs of any size to the same output shape."""
    # Test with different image sizes
    sizes = [(100, 100), (500, 300), (224, 224)]

    for size in sizes:
        sample_image = Image.new("RGB", size)
        processed = teachable_machine._preprocess_image(sample_image)
        assert processed.shape == (1, 224, 224, 3)


def test_preprocess_image_content(teachable_machine):
    """_preprocess_image should preserve the relative spatial layout of image content."""
    # Create a sample image with a specific pattern
    sample_array = np.zeros((300, 200, 3), dtype=np.uint8)
    sample_array[:100, :100] = 255  # White square in top-left corner
    sample_image = Image.fromarray(sample_array)

    # Process the image
    processed = teachable_machine._preprocess_image(sample_image)

    # Check if the white square is still in the top-left corner (approximately)
    assert np.mean(processed[0, :50, :50]) > np.mean(processed[0, 50:, 50:])


def test_compat_depthwise_conv2d_ignores_stray_groups_kwarg():
    """
    `_CompatDepthwiseConv2D` must silently drop an unrecognized 'groups'
    kwarg, which Teachable Machine's exported .h5 configs always include.
    Plain `tf.keras.layers.DepthwiseConv2D` raises on this under Keras 3
    (TensorFlow >= 2.16); this is the direct unit-level check of the fix.
    """
    layer = _CompatDepthwiseConv2D(kernel_size=3, groups=1)
    assert layer.kernel_size == (3, 3)


def test_load_model_with_legacy_groups_config(tmp_path):
    """
    Regression test for https://github.com/MeqdadDev/teachable-machine/issues/2.

    Teachable Machine's Keras/.h5 export embeds a stray 'groups': 1 key
    in every saved DepthwiseConv2D layer config. Current Keras (bundled
    by default since TensorFlow 2.16) validates configs strictly and
    raises on that unrecognized key. This test builds a tiny model,
    tampers its saved .h5 config to reproduce that legacy shape, and
    verifies TeachableMachine still loads and classifies with it.
    """
    h5py = pytest.importorskip("h5py")
    tf = pytest.importorskip("tensorflow")

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, name="expanded_conv_depthwise"
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )

    model_path = tmp_path / "keras_model.h5"
    model.save(model_path)

    # Reproduce Teachable Machine's legacy DepthwiseConv2D config by
    # injecting the unused 'groups' key into the saved model config.
    with h5py.File(model_path, "r+") as f:
        model_config = json.loads(f.attrs["model_config"])
        layers = model_config["config"]["layers"]
        patched_layers = [
            layer for layer in layers if layer["class_name"] == "DepthwiseConv2D"
        ]
        assert patched_layers, "test setup: no DepthwiseConv2D layer to tamper"
        for layer in patched_layers:
            layer["config"]["groups"] = 1
        f.attrs["model_config"] = json.dumps(model_config)

    labels_path = tmp_path / "labels.txt"
    labels_path.write_text("0 Class A\n1 Class B\n")

    # Without the compatibility patch, this raises TypeError/ValueError
    # on Keras 3: "Unrecognized keyword arguments... {'groups': 1}".
    tm = TeachableMachine(
        model_path=str(model_path), labels_file_path=str(labels_path)
    )

    sample_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
    result = tm._get_image_classification(sample_image)

    assert result["class_name"] in {"0 Class A", "1 Class B"}
    assert result["predictions"].shape == (2,)


def test_flatten_sequential_layers():
    """
    _flatten_sequential_layers must inline nested Sequential layer lists
    (keeping only the first InputLayer seen overall) while leaving
    Functional submodels and ordinary layers intact -- this is the core
    of the nested-Sequential .h5 fix (see _load_nested_sequential_h5):
    some Teachable Machine exports save the model as a Sequential
    wrapping nested Sequential/Functional submodels (mirroring a
    MobileNet-based feature extractor + classification head), which
    Keras 3's legacy H5 loader mis-rebuilds.
    """
    nested_config = {
        "class_name": "Sequential",
        "config": {
            "layers": [
                {
                    "class_name": "InputLayer",
                    "config": {"batch_input_shape": [None, 224, 224, 3]},
                },
                {
                    "class_name": "Sequential",
                    "config": {
                        "layers": [
                            {
                                "class_name": "InputLayer",
                                "config": {"batch_input_shape": [None, 224, 224, 3]},
                            },
                            {"class_name": "Functional", "config": {"name": "backbone"}},
                            {
                                "class_name": "GlobalAveragePooling2D",
                                "config": {"name": "pool"},
                            },
                        ]
                    },
                },
                {
                    "class_name": "Sequential",
                    "config": {
                        "layers": [
                            {
                                "class_name": "InputLayer",
                                "config": {"batch_input_shape": [None, 1280]},
                            },
                            {"class_name": "Dense", "config": {"name": "dense"}},
                            {"class_name": "Dense", "config": {"name": "dense_1"}},
                        ]
                    },
                },
            ]
        },
    }

    flat = TeachableMachine._flatten_sequential_layers(nested_config)

    assert [layer["class_name"] for layer in flat] == [
        "InputLayer",
        "Functional",
        "GlobalAveragePooling2D",
        "Dense",
        "Dense",
    ]
    # Only the very first InputLayer survives; the nested submodels' own
    # InputLayers (which just restate the same input shape) are dropped.
    assert sum(1 for layer in flat if layer["class_name"] == "InputLayer") == 1


def test_restore_weights_by_leaf_name(tmp_path):
    """
    _restore_weights_by_leaf_name must match each leaf layer's name
    against the H5 file's saved weight paths -- indexed via each group's
    'weight_names' attribute -- regardless of what group they're nested
    under. This is what lets _load_nested_sequential_h5 restore weights
    after discarding the original nested-Sequential grouping.
    """
    h5py = pytest.importorskip("h5py")
    from tensorflow.keras import layers

    dense_a_weights = [
        np.ones((2, 3), dtype="float32"),
        np.full((3,), 2.0, dtype="float32"),
    ]
    dense_b_weights = [
        np.full((3, 1), 5.0, dtype="float32"),
        np.array([7.0], dtype="float32"),
    ]

    # Hand-built H5 layout: dense_a is nested two levels deep, as if
    # under a wrapper Sequential that _load_nested_sequential_h5 would
    # have discarded; dense_b sits at the top level. The lookup is by
    # leaf layer name alone, so nesting depth shouldn't matter.
    h5_path = tmp_path / "weights.h5"
    with h5py.File(h5_path, "w") as f:
        root = f.create_group("model_weights")
        wrapper = root.create_group("outer_wrapper")
        group_a = wrapper.create_group("dense_a")
        group_a.attrs["weight_names"] = [b"dense_a/kernel:0", b"dense_a/bias:0"]
        group_a.create_dataset("dense_a/kernel:0", data=dense_a_weights[0])
        group_a.create_dataset("dense_a/bias:0", data=dense_a_weights[1])

        group_b = root.create_group("dense_b")
        group_b.attrs["weight_names"] = [b"dense_b/kernel:0", b"dense_b/bias:0"]
        group_b.create_dataset("dense_b/kernel:0", data=dense_b_weights[0])
        group_b.create_dataset("dense_b/bias:0", data=dense_b_weights[1])

    # Freshly-built layers standing in for a rebuild via the Functional API.
    rebuilt_a = layers.Dense(3, name="dense_a")
    rebuilt_a.build((None, 2))
    rebuilt_b = layers.Dense(1, name="dense_b")
    rebuilt_b.build((None, 3))

    tm = object.__new__(TeachableMachine)
    tm._restore_weights_by_leaf_name(str(h5_path), [rebuilt_a, rebuilt_b])

    np.testing.assert_array_equal(rebuilt_a.get_weights()[0], dense_a_weights[0])
    np.testing.assert_array_equal(rebuilt_a.get_weights()[1], dense_a_weights[1])
    np.testing.assert_array_equal(rebuilt_b.get_weights()[0], dense_b_weights[0])
    np.testing.assert_array_equal(rebuilt_b.get_weights()[1], dense_b_weights[1])


def test_restore_weights_by_leaf_name_raises_on_missing_layer(tmp_path):
    """A leaf layer with no matching saved weights must raise, not fail silently."""
    h5py = pytest.importorskip("h5py")
    from tensorflow.keras import layers

    h5_path = tmp_path / "weights.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_group("model_weights")

    layer = layers.Dense(3, name="dense_a")
    layer.build((None, 2))

    tm = object.__new__(TeachableMachine)
    with pytest.raises(ValueError, match="dense_a"):
        tm._restore_weights_by_leaf_name(str(h5_path), [layer])


def test_show_prediction_on_image_font_fallback(tmp_path):
    """
    show_prediction_on_image() must fall back to ImageFont.load_default()
    when ImageFont.truetype can't resolve the named font by bare name
    (e.g. on Windows, where DejaVuSans-Bold.ttf isn't bundled and raises
    OSError), instead of crashing -- and, by running for real against
    whatever Pillow is installed, also exercises textbbox()-based sizing
    in place of ImageDraw.textsize(), removed in Pillow 10+.
    """
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (224, 224), color=(50, 50, 50)).save(image_path)

    real_truetype = ImageFont.truetype

    def fake_truetype(font, *args, **kwargs):
        # Only the bare-name lookup fails, matching real Windows behavior;
        # Pillow's own load_default(size=...) fallback internally calls
        # truetype() again on its embedded font bytes, which must still
        # succeed, or this test would trip on that instead of the fix.
        if font == "DejaVuSans-Bold.ttf":
            raise OSError("cannot open resource")
        return real_truetype(font, *args, **kwargs)

    tm = object.__new__(TeachableMachine)
    result = {"class_name": "Class A", "class_confidence": 0.987}

    with mock.patch.object(ImageFont, "truetype", side_effect=fake_truetype):
        annotated = tm.show_prediction_on_image(
            str(image_path), result, convert_to_bgr=False
        )

    assert isinstance(annotated, Image.Image)
    assert annotated.size == (224, 224)
