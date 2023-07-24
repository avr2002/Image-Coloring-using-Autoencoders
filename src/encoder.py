import os
import numpy as np
from src.constants import Constanst
from src.get_data import GetData
from tensorflow.keras.applications.vgg16 import VGG16


class Encoder(Constanst):
    def __init__(self) -> None:
        # Defining the model
        self.base_model = VGG16(include_top=False, weights='imagenet', 
                                input_shape=self.INPUT_SHAPE)

        self.base_model.trainable = False
        

    # Process the data in batches and extract VGG features
    def extract_vgg_features(self, generator, model):
        vgg_features = []
        Y_targets = []  # Collect the target data for each batch
        for batch in generator:
            X_batch, Y_batch = GetData.preprocess_lab(batch)
            vgg_batch = model.predict(batch, verbose=0)  # Use the original 3-channel image for VGG
            vgg_features.append(vgg_batch)
            Y_targets.append(Y_batch)
            if len(vgg_features) * self.BATCH_SIZE >= len(generator.filenames):
                break
        return np.concatenate(vgg_features, axis=0), np.concatenate(Y_targets, axis=0)
    
    def fit(self):
        # Extract VGG features and target data from the training data
        vgg_features, Y = self.extract_vgg_features(generator= GetData.get_train_data_generator, 
                                                    model= self.base_model)
        return vgg_features, Y, self.base_model
