from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np


class TeachableMachine(object):
    __environments = ('pc', 'computer', 'rpi', 'raspberrypi', 'raspberry pi')

    def __init__(self, model_path='keras_model.h5', labels_file_path='labels.txt', deployment_env='pc') -> None:
        self.environment = deployment_env.lower()
        self.labels_file_path = labels_file_path
        try:
            self.model = load_model(model_path)
        except IOError as e:
            print('*****Wrong model path*****')
            raise IOError from e
        except:
            raise '*****Error while loading model.*****'
        self.object_creation_status = self.environment in self.__environments
        if self.object_creation_status:
            print('Teachable Machine Object is created successfully.')
        else:
            raise '*****Wrong environment selection: select your environment ["pc", "rpi"] ...etc.*****'

    def classify_on_pc(self, img_path: str):
        '''To deploy your Teachable Machine Model on a computer/PC
            with .h5 extension using TensorFlow.

            Parameters:
            * img: Provide your img before classification process.

            Returns:
            * highest_prediciton
            * all_predictions
        '''
        try:
            image = Image.open(img_path)
        except FileNotFoundError as e:
            raise '*****Error in image path.*****' from e

        if self.object_creation_status:
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.ANTIALIAS)
            image_array = np.asarray(image)
            normalized_image_array = (
                image_array.astype(np.float32) / 127.0) - 1
            data[0] = normalized_image_array
            all_predictions = self.model.predict(data)
            highest_prediciton = np.argmax(all_predictions)
            return highest_prediciton, all_predictions

    def classify_on_rpi(self, img):
        '''To deploy your Teachable Machine Model on Raspberry Pi board.
            with .tflite extension using TensorFlowLite.
            Parameters:
            - model_path: str (To provide the path of your exported model with .h5 extension)
        '''
        pass
