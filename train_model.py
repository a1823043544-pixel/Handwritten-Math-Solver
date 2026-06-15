import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
import config  # We need the LABELS list from your config file

# --- Settings ---
# Where are your + - * / pictures?
LOCAL_FOLDER = os.path.join("data", "raw_symbols")

# The order of our labels.
# It must be: 0, 1, 2... 9, add, sub, mul, div, (, ), useless
MY_LABELS = config.LABELS


# This function makes your drawing look like the MNIST dataset
# MNIST images are: 28x28 size, black background, white text, centered
def make_image_look_nice(image):
    # 1. Turn to gray if it is colored
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Make it black and white (Black background, White text)
    # Check the corner pixel. If it's bright, the background is white.
    if image[0, 0] > 127:
        # Flip colors: White -> Black
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    else:
        # Keep colors
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # 3. Cut the black borders (Crop the symbol)
    # Find all white pixels
    points = cv2.findNonZero(image)
    if points is None: return None  # If image is empty, skip it

    # Get a box around the white pixels
    x, y, w, h = cv2.boundingRect(points)
    cropped = image[y:y + h, x:x + w]

    # 4. Resize to 20x20 (Keep the shape, don't stretch!)
    # We want it to fit inside a 28x28 box, so 20 is a good size.
    max_side = max(w, h)
    ratio = 20.0 / max_side
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    cropped = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 5. Paste into a 28x28 black box
    black_box = np.zeros((28, 28), dtype=np.uint8)

    # Calculate start position to put it in the center
    start_x = (28 - new_w) // 2
    start_y = (28 - new_h) // 2

    black_box[start_y:start_y + new_h, start_x:start_x + new_w] = cropped

    # 6. Shift to Center of Mass (This is what MNIST does)
    # This moves the "ink" to the center, not just the box.
    rows, cols = black_box.shape
    # Calculate moments
    M = cv2.moments(black_box)
    if M["m00"] > 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # Calculate how much to shift
        shift_x = 14 - cX
        shift_y = 14 - cY

        # Move the image
        matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        black_box = cv2.warpAffine(black_box, matrix, (28, 28))

    return black_box


def prepare_data():
    all_images = []
    all_labels = []

    # --- Part 1: Get 0-9 from the Internet (MNIST) ---
    print("Step 1: Downloading numbers (0-9) from MNIST...")
    # This automatically downloads the standard dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Combine train and test to get more data
    mnist_x = np.concatenate([x_train, x_test])
    mnist_y = np.concatenate([y_train, y_test])

    print(f"   -> Got {len(mnist_x)} images of numbers.")

    # Add them to our list
    all_images.append(mnist_x)
    all_labels.append(mnist_y)

    # --- Part 2: Get symbols from your computer ---
    print(f"Step 2: Loading symbols from {LOCAL_FOLDER}...")

    my_symbol_images = []
    my_symbol_labels = []

    # We skip 0-9 (first 10 items) because we already have them from MNIST
    # We only want: add, sub, mul, div, (, ), useless
    start_index = 10

    for i in range(start_index, len(MY_LABELS)):
        folder_name = MY_LABELS[i]
        full_path = os.path.join(LOCAL_FOLDER, folder_name)

        # Check if folder exists
        if not os.path.exists(full_path):
            print(f"   Warning: Cannot find folder for '{folder_name}'")
            continue

        # Read files in the folder
        count = 0
        file_list = os.listdir(full_path)

        for filename in file_list:
            # Only read images
            if filename.endswith(".png") or filename.endswith(".jpg"):
                filepath = os.path.join(full_path, filename)

                try:
                    # Read image as gray
                    raw_img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

                    # Fix the image (Resize, Center, Black Background)
                    nice_img = make_image_look_nice(raw_img)

                    if nice_img is not None:
                        my_symbol_images.append(nice_img)
                        my_symbol_labels.append(i)  # Save the ID (e.g., 10 for add)
                        count += 1
                except:
                    # If an image is broken, just skip it
                    pass

        print(f"   -> Loaded {count} images for '{folder_name}'")

    # --- Part 3: Balance the data ---
    # MNIST has 7000 images per number. Our symbols might only have 500.
    # We need to copy our symbols so the model sees them enough times.

    if len(my_symbol_images) > 0:
        # Convert list to numpy array
        np_images = np.array(my_symbol_images)
        np_labels = np.array(my_symbol_labels)

        # We want about 4000 images per symbol
        current_amount = len(np_images) // (len(MY_LABELS) - 10)

        if current_amount < 4000:
            # Calculate how many times to copy
            times_to_copy = 4000 // (current_amount + 1)
            # Don't copy too much (max 20 times)
            if times_to_copy > 20: times_to_copy = 20

            if times_to_copy > 1:
                print(f"Step 3: Copying symbol data {times_to_copy} times to balance...")
                np_images = np.repeat(np_images, times_to_copy, axis=0)
                np_labels = np.repeat(np_labels, times_to_copy, axis=0)

        # Add symbols to the big list
        all_images.append(np_images)
        all_labels.append(np_labels)

    # --- Part 4: Final formatting ---
    # Merge everything into one big array
    X = np.concatenate(all_images, axis=0)
    y = np.concatenate(all_labels, axis=0)

    # Reshape for Keras: (Number of images, 28, 28, 1 channel)
    X = np.expand_dims(X, axis=-1)

    # Normalize: Make numbers between 0 and 1 (instead of 0-255)
    X = X.astype('float32') / 255.0

    # Convert labels to "One-Hot" format
    # Example: 2 -> [0, 0, 1, 0, 0...]
    y = to_categorical(y, num_classes=len(MY_LABELS))

    # Shuffle the data (Mix numbers and symbols)
    X, y = shuffle(X, y, random_state=42)

    return X, y


# --- Main Program ---
if __name__ == "__main__":

    # Fix for "Out of Memory" errors on some GPUs
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # 1. Load Data
    print("--- Starting Data Preparation ---")
    X_train, y_train = prepare_data()
    print(f"Total images for training: {X_train.shape[0]}")

    # 2. Build the Model (CNN)
    # This is like a brain with layers
    model = Sequential([
        # Layer 1: Look for small lines and curves
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
        BatchNormalization(),
        MaxPooling2D(2, 2),  # Reduce size

        # Layer 2: Look for shapes
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        # Layer 3: Look for complex patterns
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        # Flatten: Turn 2D image into 1D list


        Flatten(),

        # Decision Layer
        Dense(128, activation='relu'),
        Dropout(0.5),  # Forget 50% randomly to prevent memorizing

        # Output Layer: Give probability for each label
        Dense(len(MY_LABELS), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 3. Setup Data Generator (Augmentation)
    # This slightly rotates and zooms images to make the model smarter
    # IMPORTANT: rotation_range is small (5) so '+' doesn't look like 'x'
    datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )

    # 4. Start Training
    print("--- Starting Training (15 Rounds) ---")
    model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=15)

    # 5. Save the result
    model.save(config.MODEL_PATH)
    print(f"--- Done! Model saved to {config.MODEL_PATH} ---")