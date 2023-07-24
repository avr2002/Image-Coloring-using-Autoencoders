from src.encoder import Encoder
from src.decoder import Decoder


def train():
    X, y, encoder_model = Encoder.fit()
    history, decoder_model = Decoder.fit(vgg_features=X, Y=y)
    