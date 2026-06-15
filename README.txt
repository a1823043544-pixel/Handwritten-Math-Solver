PROJECT TITLE: Handwritten Math Solver
GROUP NO:16
GROUP MEMBER:WEN GUANGCHEN         AIT2509038
                              WANG CHANGKUN        AIT2509035
                              QIAO DAN                       AIT2509030
                              FENG ZIXI                         AIT2509011
                              ZENG LINGTIAN              AIT2509053
                              ZHANG JIE                       AIT2509056
COURSE: AIT102 - Python and TensorFlow Programming


1. PROJECT DESCRIPTION
This project is a Python application that recognizes and solves handwritten mathematical equations. Unlike standard OCR tools, this program integrates a Convolutional Neural Network (CNN) with custom "Geometric Logic Rules"  to correct common AI misclassifications (e.g., distinguishing '4' vs '+' based on stroke spine position, or '8' vs '×' based on topological hole counting).

   It offers three interaction modes:
       1. Draw Board: A digital canvas with adjustable pen thickness for writing equations directly.
       2. Upload File: Recognizes equations from local image files (supports both black-on-white and white-on-black).
       3. Camera Mode: Captures and solves equations in real-time using the webcam.


2. REQUIREMENTS & DEPENDENCIES
The project relies on TensorFlow for the AI model and OpenCV for image processing.
Required libraries:
- python >= 3.8
- tensorflow
- opencv-python
- numpy
- pillow
- pandas

  Install command:
  pip install tensorflow opencv-python numpy pillow pandas


3. FILE STRUCTURE
  - main.py: The main entry point of the application (GUI).
  - config.py: Configuration settings for model paths and labels.
  - utils.py: Core image processing logic (Smart Thresholding, Hole Counting, Contour Splitting).
  - math_solver_model.h5: The pre-trained CNN model file.
  - /data: Folder containing raw training data.
  - /scripts: Contains data generation scripts used for training.

4. HOW TO RUN
  1. Ensure 'math_solver_model.h5' is in the root directory.
  2.Run main.py


5. USER GUIDE
  1. Select Mode: Choose Draw, Upload, or Camera from the left panel.
  2. Draw Mode: Use the slider to adjust pen size. Write an equation e.g., "2+3" and click SOLVE.
  3. Result: The recognized equation and the calculated result will appear on the right.

6. REFERENCES
- Kaggle Handwritten Dataset:https://www.kaggle.com/datasets/sarunpakkkkkk/handwritten-math-symbols-dataset?resource=download
- Kaggle HASYv2 Dataset: https://www.kaggle.com/datasets/greg115/hasyv2
- OpenCV Contours Tutorial: https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
- TensorFlow CNN Guide: https://www.tensorflow.org/tutorials/images/cnn