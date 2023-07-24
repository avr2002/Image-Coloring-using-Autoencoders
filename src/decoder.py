import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from src.constants import Constanst
from src.encoder import Encoder


class Decoder(Constanst):

    def __init__(self):
        # Define the decoder model
        self.model = models.Sequential(name='decoder_model')

    def decoder_model(self):
        # input_shape=(7,7,512) --> This is the o/p shape of encoder i.e. VGG16 model
        self.model.add(layers.InputLayer(input_shape=self.DECODER_INPUT_SHAPE))

        self.model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same'))
        self.model.add(layers.UpSampling2D((2, 2)))

        self.model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same'))
        self.model.add(layers.UpSampling2D((2, 2)))

        self.model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
        self.model.add(layers.UpSampling2D((2, 2)))

        self.model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same'))
        self.model.add(layers.UpSampling2D((2, 2)))

        self.model.add(layers.Conv2D(16, (3,3), activation='relu', padding='same'))
        self.model.add(layers.Conv2D(8, (3,3), activation='relu', padding='same'))
        # self.model.add(layers.UpSampling2D((2, 2)))

        self.model.add(layers.Conv2D(2, (3, 3), activation='tanh', padding='same'))
        self.model.add(layers.UpSampling2D((2, 2)))

        # Compile the model
        self.model.compile(optimizer=optimizers.RMSprop(), 
                    loss='mse', 
                    metrics=['accuracy'])
        
        return self.model
    
    def fit(self, vgg_features, Y):
        # Apply Early Stopping callback
        # early_stopping = EarlyStopping(monitor='loss', 
        #                                patience=50,
        #                                restore_best_weights=True)

        tensorboard = TensorBoard(log_dir="./output/callbacks")

        # Train the decoder
        history = self.model.fit(vgg_features, Y,
                                verbose=1,
                                epochs=self.EPOCHS, 
                                batch_size=self.BATCH_SIZE, 
                                steps_per_epoch=10,
                                callbacks=[tensorboard], #early_stopping
                                workers=10)
        return history, self.model
