import json

import pytest
from src.teachable_machine import TeachableMachine, _CompatDepthwiseConv2D
from PIL import Image
import numpy as np


@pytest.fixture
def teachable_machine(mocker):
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
    # Create a sample image
    sample_image = Image.new("RGB", (300, 200))

    # Process the image
    processed = teachable_machine._preprocess_image(sample_image)

    # Check the output shape
    assert processed.shape == (1, 224, 224, 3)


def test_preprocess_image_normalization(teachable_machine):
    # Create a sample image with known values
    sample_array = np.full((300, 200, 3), 127, dtype=np.uint8)
    sample_image = Image.fromarray(sample_array)

    # Process the image
    processed = teachable_machine._preprocess_image(sample_image)

    # Check if the values are normalized correctly (close to -0.00392, not exactly 0)
    expected_value = (127 / 127.5) - 1
    assert np.allclose(processed, expected_value, atol=1e-6)


def test_preprocess_image_different_sizes(teachable_machine):
    # Test with different image sizes
    sizes = [(100, 100), (500, 300), (224, 224)]

    for size in sizes:
        sample_image = Image.new("RGB", size)
        processed = teachable_machine._preprocess_image(sample_image)
        assert processed.shape == (1, 224, 224, 3)


def test_preprocess_image_content(teachable_machine):
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
