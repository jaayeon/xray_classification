import os
from shutil import copyfile
import utils.loader as L


if __name__ == '__main__':
    train_dir = ""
    train_label_dir= ""

    valid_dir = ""
    valid_label_dir = ""

    test_dir = ""
    test_label_dir = ""

    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)
        os.makedirs(valid_label_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        os.makedirs(test_label_dir)

     