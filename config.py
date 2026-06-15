import os

# Image settings
IMG_SIZE = 28
img_shape = (28, 28, 1)

# Folders
DATA_DIR = os.path.join("data", "raw_symbols")
MODEL_PATH = "math_model.h5"

# put 'useless' at the end.
LABELS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'add', 'sub', 'mul', 'div', 'left', 'right'
]

NUM_CLASSES = len(LABELS)