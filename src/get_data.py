import os
from skimage.color import rgb2lab
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.constants import Constanst

class GetData(Constanst):
    
    def __init__(self) -> None:
        pass

    def get_train_data_generator(self):
        DATA_DIR = os.path.join(os.getcwd(), self.TRAIN_DIR)

        train_datagen = ImageDataGenerator(rescale=1./255, 
                                    shear_range=0.2, zoom_range=0.2,
                                    rotation_range=20, horizontal_flip=True)

        train_generator = train_datagen.flow_from_directory(DATA_DIR,
                                                            target_size=self.TARGET_SIZE,
                                                            batch_size=self.BATCH_SIZE,
                                                            class_mode=None)
        
        return train_generator
    
    # Convert RGB images to LAB format and separate L and AB channels
    def preprocess_lab(self, image_batch):
        lab_batch = rgb2lab(image_batch)
        X_batch = lab_batch[:, :, :, 0]  # L channel
        Y_batch = lab_batch[:, :, :, 1:] / 128  # AB channels between -1 and 1
        X_batch = X_batch.reshape(X_batch.shape + (1,))  # Add channel dimension for grayscale
        return X_batch, Y_batch
