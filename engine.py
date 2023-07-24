import os
from src.constants import TRAIN_DIR
from src.get_data import get_train_data_generator

if __name__=='__main__':
    print(get_train_data_generator())
    # print(os.listdir(TRAIN_DIR))