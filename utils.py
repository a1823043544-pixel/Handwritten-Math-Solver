import cv2
import numpy as np


def get_contours_and_images(image_bgr):
    """
    Main function to process the image:
    1. Detect if background is black or white.
    2. Cut out the numbers/symbols.
    3. Merge dots (for division symbol).
    4. Process each cut-out to match model requirements.
    5. Count holes (for logic checks).
    """

    # 1. Convert to grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Smart Thresholding (Fixes Black vs White background)
    # Check the top-left pixel.
    # If it is dark (<127), it's a black background (keep as is).
    # If it is bright (>127), it's a white background (invert it).

    mean_val = np.mean(gray)

    if mean_val < 127:
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    else:
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    # 3. Find Contours (The boxes around numbers)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rect_list = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Filter out tiny noise (dots that are too small)
        if w * h > 30:
            rect_list.append([x, y, w, h])

    # 4. Merge Rectangles (Fixes ÷ and =)
    rect_list = merge_rectangles(rect_list)

    # Sort from Left to Right
    rect_list.sort(key=lambda r: r[0])

    # 5. Process each character
    final_data = []

    for (x, y, w, h) in rect_list:
        # Cut the region of interest (ROI)
        roi = thresh[y:y + h, x:x + w]

        # Prepare image for the AI Model
        processed_img = process_one_character(roi)

        # Get raw image for hole counting (convert back to 0-255 int)
        # processed_img is (28, 28, 1) float. We need (28, 28) int.
        raw_img_for_holes = (processed_img.squeeze() * 255).astype(np.uint8)

        # Count holes (0, 1, or 2)
        holes = count_holes(raw_img_for_holes)
        pixel_density = np.mean(processed_img)
        # Return: [Image, Width, Height, Hole_Count]
        final_data.append((processed_img, w, h, holes, pixel_density, raw_img_for_holes))

    return final_data


def merge_rectangles(rects):
    """
    Merges separate parts of a symbol (like the dots of a division sign).
    """
    found_merge = True
    while found_merge:
        found_merge = False
        new_rects = []

        while len(rects) > 0:
            r1 = rects.pop(0)
            x1, y1, w1, h1 = r1
            was_merged = False

            for i in range(len(rects)):
                r2 = rects[i]
                x2, y2, w2, h2 = r2

                # Center points
                c1_x = x1 + w1 / 2
                c2_x = x2 + w2 / 2

                # If they are close horizontally (within 30px)
                if abs(c1_x - c2_x) < 30:
                    # Merge them
                    new_x = min(x1, x2)
                    new_y = min(y1, y2)
                    new_w = max(x1 + w1, x2 + w2) - new_x
                    new_h = max(y1 + h1, y2 + h2) - new_y

                    rects[i] = [new_x, new_y, new_w, new_h]
                    found_merge = True
                    was_merged = True
                    break

            if not was_merged:
                new_rects.append(r1)

        rects = new_rects
    return rects


def process_one_character(img):
    """
    Resizes and pads the image to 28x28 to match MNIST training data.
    """
    # 1. Thicken the lines (Crucial for thin handwriting!)
    # Using 3x3 kernel makes it bold.
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)

    # 2. Resize maintaining aspect ratio
    h, w = img.shape
    max_side = max(h, w)

    # Scale to 20px (leaving 4px padding)
    scale = 20.0 / max_side
    new_w = int(w * scale)
    new_h = int(h * scale)

    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 3. Paste into center of 28x28 black canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)

    start_x = (28 - new_w) // 2
    start_y = (28 - new_h) // 2

    canvas[start_y:start_y + new_h, start_x:start_x + new_w] = img_resized

    # Normalize (0.0 to 1.0) and add channel dimension
    final = canvas.astype('float32') / 255.0
    final = np.expand_dims(final, axis=-1)

    return final


def count_holes(img):
    """
    Counts white enclosed areas (holes) in the image.
    Used to distinguish '8' from 'x', or '0' from 'u'.
    """
    # Erode slightly to separate crossed lines (like in 'x')
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)

    # Find contours including inner holes (RETR_CCOMP)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        return 0

    holes = 0
    # hierarchy: [Next, Previous, First_Child, Parent]
    # If Parent != -1, it means this contour is inside another one (a hole)
    for i in range(len(contours)):
        if hierarchy[0][i][3] != -1:
            # Only count if the hole is big enough (ignore noise)
            area = cv2.contourArea(contours[i])
            if area > 5:
                holes += 1
    return holes