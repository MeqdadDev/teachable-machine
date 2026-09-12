import json

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np


class _CompatDepthwiseConv2D(DepthwiseConv2D):
    """
    Drop-in replacement for Keras' DepthwiseConv2D that tolerates the
    stray 'groups' argument found in models exported by Google's
    Teachable Machine platform.

    Teachable Machine's Keras/.h5 export embeds 'groups': 1 in every
    saved DepthwiseConv2D layer config. That key was never meaningful
    for a depthwise convolution, but older Keras ignored it. Keras 3
    (bundled by default since TensorFlow 2.16) validates layer configs
    strictly and raises:
        TypeError/ValueError: Unrecognized keyword arguments passed to
        DepthwiseConv2D: {'groups': 1}
    This subclass discards the unused key on construction so exported
    models keep loading on current TensorFlow/Keras installs, without
    requiring users to pin an old TensorFlow version.
    """

    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)


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
            self._model = load_model(
                model_path,
                compile=False,
                custom_objects={"DepthwiseConv2D": _CompatDepthwiseConv2D},
            )
            return
        except IOError as e:
            print("LoadingModelError: Error while loading Teachable Machine model")
            raise IOError("Error loading model") from e
        except Exception as e:
            # Some Teachable Machine .h5 exports save the model as a Sequential
            # that wraps nested Sequential/Functional submodels (Sequential >
            # [InputLayer, Sequential > [InputLayer, Functional backbone,
            # GlobalAveragePooling2D], Sequential > [InputLayer, Dense, Dense]]).
            # Keras 3's legacy H5 loader mis-rebuilds that shape: when the
            # Functional backbone is called as a plain layer inside
            # Sequential.build(), its single output comes back wrapped in a
            # length-1 list, which the next layer then treats as two separate
            # inputs, failing with e.g. "expects 1 input(s), but it received 2
            # input tensors". Fall back to a manual reconstruction for that case.
            try:
                self._model = self._load_nested_sequential_h5(model_path)
                return
            except Exception:
                pass
            print("LoadingModelError: Error while loading Teachable Machine model")
            raise FileNotFoundError("Model file not found") from e

    def _load_nested_sequential_h5(self, model_path: str):
        """
        Rebuild a Teachable Machine .h5 export whose Sequential wraps nested
        Sequential/Functional submodels, working around the Keras 3 legacy H5
        loader bug described in `_load_model`.

        Flattens the nested Sequential wrappers (keeping only the first
        InputLayer and leaving any Functional submodels intact), rebuilds the
        layers in order via the Functional API -- unwrapping the length-1
        list a Functional submodel's call returns -- and then restores
        weights by matching each leaf layer's name against the H5 file's
        saved weight paths, sidestepping Keras's built-in weight loader
        (which expects the now-discarded nested-Sequential grouping).
        """
        import h5py
        from keras import Model
        from keras.layers import Input
        from keras.src.legacy.saving import saving_utils as legacy_saving_utils

        custom_objects = {"DepthwiseConv2D": _CompatDepthwiseConv2D}

        with h5py.File(model_path, "r") as f:
            raw_config = f.attrs.get("model_config")
            if raw_config is None:
                raise ValueError("No model_config found in H5 file")
            if isinstance(raw_config, bytes):
                raw_config = raw_config.decode("utf-8")
            full_config = json.loads(raw_config)

        if full_config.get("class_name") != "Sequential":
            raise ValueError("Not a nested-Sequential export; nothing to flatten")

        flat_layers = self._flatten_sequential_layers(full_config)
        if not flat_layers or flat_layers[0]["class_name"] != "InputLayer":
            raise ValueError("Unexpected model shape after flattening")

        input_shape = flat_layers[0]["config"]["batch_input_shape"][1:]
        x = Input(shape=input_shape)
        model_input = x
        leaf_layers = []
        for layer_config in flat_layers[1:]:
            if layer_config["class_name"] == "Functional":
                sub_model = Model.from_config(
                    layer_config["config"], custom_objects=custom_objects
                )
                x = sub_model(x)
                if isinstance(x, (list, tuple)) and len(x) == 1:
                    x = x[0]
                leaf_layers.extend(sub_model.layers)
            else:
                layer = legacy_saving_utils.model_from_config(
                    layer_config, custom_objects=custom_objects
                )
                x = layer(x)
                leaf_layers.append(layer)

        model = Model(inputs=model_input, outputs=x)
        self._restore_weights_by_leaf_name(model_path, leaf_layers)
        return model

    @staticmethod
    def _flatten_sequential_layers(layer_config, _seen_input=None):
        """
        Inline nested Sequential layer lists, keeping only the first
        InputLayer seen overall and leaving Functional submodels intact.
        """
        if _seen_input is None:
            _seen_input = [False]
        if layer_config["class_name"] != "Sequential":
            return [layer_config]
        flattened = []
        for sub_layer_config in layer_config["config"]["layers"]:
            if sub_layer_config["class_name"] == "InputLayer":
                if not _seen_input[0]:
                    flattened.append(sub_layer_config)
                    _seen_input[0] = True
                continue
            flattened.extend(
                TeachableMachine._flatten_sequential_layers(
                    sub_layer_config, _seen_input
                )
            )
        return flattened

    @staticmethod
    def _restore_weights_by_leaf_name(model_path, leaf_layers):
        """
        Load weights for `leaf_layers` by matching each layer's name against
        the H5 file's saved weight paths, regardless of the (now-discarded)
        nested-Sequential grouping those paths were originally saved under.
        """
        import h5py

        with h5py.File(model_path, "r") as f:
            root = f["model_weights"]
            leaf_paths = {}

            def walk(group):
                weight_names = group.attrs.get("weight_names")
                if weight_names is not None:
                    for weight_name in weight_names:
                        if isinstance(weight_name, bytes):
                            weight_name = weight_name.decode("utf-8")
                        full_path = f"{group.name}/{weight_name}"
                        leaf_name = (
                            weight_name.rsplit("/", 1)[0]
                            if "/" in weight_name
                            else group.name.rsplit("/", 1)[-1]
                        )
                        leaf_paths.setdefault(leaf_name, []).append(full_path)
                for key in group:
                    item = group[key]
                    if isinstance(item, h5py.Group):
                        walk(item)

            walk(root)

            missing = []
            for layer in leaf_layers:
                if not layer.weights:
                    continue
                paths = leaf_paths.get(layer.name)
                if paths is None:
                    missing.append(layer.name)
                    continue
                layer.set_weights([np.asarray(f[path]) for path in paths])

            if missing:
                raise ValueError(f"Could not find saved weights for layers: {missing}")

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
        try:
            # Not bundled with Pillow on every platform (notably Windows),
            # where a bare name can't be resolved and raises OSError.
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        except OSError:
            font = ImageFont.load_default(size=font_size)

        # ImageDraw.textsize was removed in Pillow 10; use textbbox instead.
        # A first measurement at the origin gives the text's extent, used only
        # to decide where to place it.
        _, top, _, bottom = draw.textbbox((0, 0), text, font=font)
        text_height = bottom - top
        position = (10, image.height - text_height - 10)

        # textbbox's (left, top, right, bottom) are offsets from the point
        # passed in, and that offset (particularly `top`, from font ascent)
        # is usually not zero -- reusing the origin-measured box as if it
        # started exactly at `position` shifts it up and clips the text's
        # descenders. Re-measure at the actual draw position instead, and
        # pad it a little so the box comfortably covers the glyphs.
        padding = 4
        box = draw.textbbox(position, text, font=font)
        draw.rectangle(
            [box[0] - padding, box[1] - padding, box[2] + padding, box[3] + padding],
            fill=(0, 0, 0, 128),
        )

        draw.text(position, text, font=font, fill=(255, 255, 255))

        if convert_to_bgr:
            image_2_numpy_arr = np.array(image)

            # Convert RGB to BGR
            image_2_numpy_arr = image_2_numpy_arr[:, :, ::-1]
            return image_2_numpy_arr

        return image
