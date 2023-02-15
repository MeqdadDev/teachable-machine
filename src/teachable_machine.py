from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np


class TeachableMachine(object):
    '''
    Create your TeachableMachine object to run your trained AI models.
    '''

    def __init__(self, model_path='keras_model.h5', labels_file_path='labels.txt', model_type='h5') -> None:
        self._model_type = model_type.lower()
        self._labels_file_path = labels_file_path
        self._supported_types = ('keras', 'Keras', 'h5', 'h5py')

        np.set_printoptions(suppress=True)
        try:
            self._model = load_model(model_path, compile=False)
        except IOError as e:
            print('LoadingModelError: Error while loading Teachable Machine model')
            raise IOError from e
        except:
            print("LoadingModelError: Error while loading Teachable Machine model")
            raise FileNotFoundError
        try:
            self._labels_file = open(self._labels_file_path, "r").readlines()
        except IOError as e:
            print('LoadingLabelsError: Error while loading labels.txt file')
            raise IOError from e
        except:
            print("LoadingLabelsError: Error while loading labels.txt file")
            raise FileNotFoundError

        self._object_creation_status = self._model_type in self._supported_types
        if self._object_creation_status:
            print('Teachable Machine Object is created successfully.')
        else:
            raise 'NotSupportedType: Your model type is not supported, try to use types such as "keras" or "h5".'

    def classify_image(self, frame_path: str):
        '''To deploy your Teachable Machine Model on a computer/PC
            with .h5 extension using TensorFlow.

            Parameters:
            * (str) frame_path: Provide path of the image to be classified.

            Returns:
            * class_name: Name of the highest predicted class according to labels.txt file
            * class_index: Index or ID of the highest predicted class according to labels.txt file
            * predictions: All prediction values for all classes.
        '''
        try:
            frame = Image.open(frame_path)
            if frame.mode != "RGB":
                frame = frame.convert("RGB")
        except FileNotFoundError as e:
            print("ImageNotFound: Error in image file.")
            raise FileNotFoundError from e
        except TypeError as e:
            print(
                "ImageTypeError: Error while converting image to RGB format, image type is not supported")
        try:
            if self._object_creation_status:
                return self._get_image_classification(frame)
        except BaseException as e:
            print('Error in classification process, retrain your model.')
            raise e

    def _get_image_classification(self, image):
        data = self._form_image(image)
        prediction = self._model.predict(data)
        class_index = np.argmax(prediction)
        class_name = self._labels_file[class_index]
        class_confidence = prediction[0][class_index]

        return {
            "class_name": class_name[2:],
            "highest_class_name": class_name[2:],
            "highest_class_id": class_index,
            "class_index": class_index,
            "class_id": class_index,
            "predictions": prediction,
            "all_predictions": prediction,
            "class_confidence": class_confidence,
            "highest_class_confidence": class_confidence,
        }

    def _form_image(self, image):
        image_data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        crop_size = (224, 224)
        image = ImageOps.fit(image, crop_size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)

        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        image_data[0] = normalized_image_array
        return image_data
