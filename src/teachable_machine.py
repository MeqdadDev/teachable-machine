from tensorflow.keras.models import load_model
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np


class TeachableMachine(object):
    """
    Create a TeachableMachine object to run pre-trained AI models.
    """

    SUPPORTED_TYPES = {"keras", "h5"}
    IMAGE_SIZE = (224, 224)

    def __init__(
        self,
        model_path="keras_model.h5",
        labels_file_path="labels.txt",
        model_type="h5",
    ) -> None:
        self._model_type = model_type.lower()
        if self._model_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported model type: {self._model_type}. Use 'keras' or 'h5'."
            )
        np.set_printoptions(suppress=True)

        self._load_model(model_path)
        self._load_labels(labels_file_path)
        print("Teachable Machine Object is created successfully.")

    def _load_model(self, model_path: str):
        try:
            self._model = load_model(model_path, compile=False)
        except IOError as e:
            print("LoadingModelError: Error while loading Teachable Machine model")
            raise IOError("Error loading model") from e
        except Exception as e:
            print("LoadingModelError: Error while loading Teachable Machine model")
            raise FileNotFoundError("Model file not found") from e

    def _load_labels(self, labels_file_path):
        try:
            with open(labels_file_path, "r") as file:
                self._labels = file.readlines()
        except IOError as e:
            print("LoadingLabelsError: Error while loading labels.txt file")
            raise IOError("Error loading labels") from e
        except Exception as e:
            print("LoadingLabelsError: Error while loading labels.txt file")
            raise FileNotFoundError("Labels file not found") from e

    def _open_image(self, image_path):
        """
        Open an image file and convert it to RGB mode.

        Parameters:
        image_path (str): Path to the image file.

        Returns:
        PIL.Image.Image: Opened image in RGB mode.
        """
        try:
            return Image.open(image_path).convert("RGB")
        except FileNotFoundError as e:
            print("ImageNotFound: Error in image file.")
            raise FileNotFoundError("Image file not found") from e
        except Exception as e:
            print("ImageTypeError: Error while opening or converting image")
            raise TypeError("Unsupported image type") from e

    def classify_image(self, image_path: str):
        """
        Classify an image using the pre-trained model.

        Parameters:
        image_path (str): Path of the image to be classified.

        Returns:
        dict: Classification results including class name, index, confidence and predictions.
        """
        image = self._open_image(image_path)
        return self._get_image_classification(image)

    def _get_image_classification(self, image):
        data = self._preprocess_image(image)
        prediction = self._model.predict(data)
        class_index = np.argmax(prediction)
        class_name = self._labels[class_index].strip()
        class_confidence = prediction[0][class_index]

        return {
            "class_name": class_name,
            "highest_class_name": class_name,
            "highest_class_id": class_index,
            "class_index": class_index,
            "class_id": class_index,
            "predictions": prediction[0],
            "all_predictions": prediction[0],
            "class_confidence": class_confidence,
            "highest_class_confidence": class_confidence,
        }

    def _preprocess_image(self, image):
        image = ImageOps.fit(image, self.IMAGE_SIZE, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        return np.expand_dims(normalized_image_array, axis=0)

    def classify_and_show(self, image_path: str, convert_to_bgr=True):
        """
        Classify an image and show the prediction results on the image.

        Parameters:
        image_path (str): Path of the input image to be classified.
        convert_to_bgr (bool, optional): Whether to convert the image to BGR format for OpenCV.
            If False, the image will be returned in RGB format. Default is True.

        Returns:
        tuple: (classification_result, image_with_prediction)
            classification_result (dict): Classification results including class name, index, confidence and predictions.
            image_with_prediction (np.ndarray or PIL.Image.Image): The image with prediction results drawn on it.
                Returns a NumPy array in BGR format if convert_to_bgr is True.
                Otherwise, returns a PIL.Image.Image in RGB format.
        """
        classification_result = self.classify_image(image_path)
        image_with_prediction = self.show_prediction_on_image(
            image_path, classification_result, convert_to_bgr=convert_to_bgr
        )
        return classification_result, image_with_prediction

    def show_prediction_on_image(
        self, image_path: str, classification_result=None, convert_to_bgr=True
    ):
        """
        Show the prediction results on the image and return the modified image.

        Parameters:
        image_path (str): Path of the input image to be classified.
        classification_result (dict, optional): Pre-computed classification result.
            If not provided, the method will classify the image.
        convert_to_bgr (bool, optional): Whether to convert the image to BGR format for OpenCV.
            If False, the image will be returned in RGB format. Default is True.

        Returns:
        np.ndarray or PIL.Image.Image: The image with prediction results drawn on it.
            Returns a NumPy array in BGR format if convert_to_bgr is True.
            Otherwise, returns a PIL.Image.Image in RGB format.
        """
        image = self._open_image(image_path)

        if classification_result is None:
            classification_result = self._get_image_classification(image)

        class_name = classification_result["class_name"]
        confidence = classification_result["class_confidence"]
        confidence_percent = confidence * 100
        text = f"{class_name}: {confidence_percent:.2f}%"

        draw = ImageDraw.Draw(image)

        font_size = int(image.height * 0.04)  # 4% of the image height
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)

        text_width, text_height = draw.textsize(text, font=font)
        position = (10, image.height - text_height - 10)

        draw.rectangle(
            [
                position[0],
                position[1],
                position[0] + text_width,
                position[1] + text_height,
            ],
            fill=(0, 0, 0, 128),
        )

        draw.text(position, text, font=font, fill=(255, 255, 255))

        if convert_to_bgr:
            image_2_numpy_arr = np.array(image)

            # Convert RGB to BGR
            image_2_numpy_arr = image_2_numpy_arr[:, :, ::-1]
            return image_2_numpy_arr

        return image
