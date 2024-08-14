import pytest
from teachable_machine import TeachableMachine
from PIL import Image, ImageDraw
import numpy as np


@pytest.fixture
def teachable_machine(mocker):
    # Create a TeachableMachine instance
    mocker.patch("teachable_machine.load_model", return_value="mock_model")

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
    mocker.patch("teachable_machine.load_model", return_value="mock_model")

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
    mocker.patch("teachable_machine.load_model", return_value="mock_model")

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
