from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np


class TeachableMachine(object):
    '''
    Create your TeachableMachine object to run your exported AI models.
    '''
    __supported_types = ('keras', 'Keras', 'h5')

    def __init__(self, model_path='keras_model.h5', labels_file_path='labels.txt', model_type='h5') -> None:
        self.model_type = model_type.lower()
        self.labels_file_path = labels_file_path
        try:
            self.model = load_model(model_path)
        except IOError as e:
            print('*****Wrong model path*****')
            raise IOError from e
        except:
            raise '*****Error while loading model.*****'
        self.object_creation_status = self.model_type in self.__supported_types
        if self.object_creation_status:
            print('Teachable Machine Object is created successfully.')
        else:
            raise '*****Not supported model type: Select your model like "keras" or "h5".*****'

    def classify_image(self, img_path: str):
        '''To deploy your Teachable Machine Model on a computer/PC
            with .h5 extension using TensorFlow.

            Parameters:
            * img: Provide your img before classification process.

            Returns:
            * highest_prediciton: The highest prediction from the classes.
            * all_predictions: All values for all classes.
        '''
        try:
            image = Image.open(img_path)
        except FileNotFoundError as e:
            raise '*****Error in image path.*****' from e
        try:
            if self.object_creation_status:
                data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                size = (224, 224)
                image = ImageOps.fit(image, size, Image.ANTIALIAS)
                image_array = np.asarray(image)
                normalized_image_array = (
                    image_array.astype(np.float32) / 127.0) - 1
                data[0] = normalized_image_array
                all_predictions = self.model.predict(data)
                highest_class_id = np.argmax(all_predictions)
                return {
                    'highest_class_id': highest_class_id,
                    'all_predictions': all_predictions,
                }
        except:
            raise 'Error in classification process, retrain your model.'
