class Constanst:
    TRAIN_DIR = "input/train/"
    TEST_DIR = "input/test/"
    TARGET_SIZE = (224,224)
    BATCH_SIZE = 16  # You can adjust this based on your GPU memory
    INPUT_SHAPE = (224,224,3)
    DECODER_INPUT_SHAPE = (7,7,512)
    EPOCHS = 2000