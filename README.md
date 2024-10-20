# Animal Hand Sign Model Training

## Overview

Each and every required files and folders while training the SVM Classifier with linear kernel and True probabilty are included in this repository for future uses and to share among the ML enthusiasts.


## Dataset

- "video_files" ------> ".mp4" -------> "open-cv && Mediapipe" -------> hand_landmarks
- Since it is for experiment, I have a smaller number of dataset that has sample videos taken in bright light.


## How I Trained the Model

You can navigate through the files and folders as mentioned in below guidelines:

                                 Video_files(tiger,cow,ok,bull)
                                             |
                                             |
                                             |
                                             +
                                    Captured_frames.py -----------------> /Captured_frames
                                             |
                                             |
                                             |
                                             +
                                        Mediapipe.py--------------------> /npy_files ----------------> Annonated_frames.py
                                                                              |                                  |
                                                                              |                                  |
                                                                              |                                  |
                                                                              +                                  +
                                                                        train_model.py                    /Annonated_frames
                                                                              |
                                                                              |
                                                                              |
                                                                              +
                                                                        /Trained_model



## Requirements

If you want to explore the working principles or my workflow to train the model. For that, ensure you have the following installed:

- Python 3.x
- OpenCV
- Mediapipe
- Scikit-learn
- Joblib
- NumPy

Virtualenv is recommended to use a virtual environment to manage dependencies.

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```
2. **Activate Virtual Envioronment**
- On Windows
  
   ```bash
   venv\Scripts\activate
   ```
- On MacOS/Linux
  
    ```bash
   venv\bin\activate
   ```

3. **Install the Required Packages**

   ```bash
   pip install -r requirements.txt
   ```


## Workflow


1. **Run "captured_frames.py"**
   
   ```bash
   Python captured_frames.py
   ```
   - video files from "/video_files" are converted to the frames in "Captured_frames"


3. **Run "Mediapipe.py"**
   
   ```bash
   Python Mediapipe.py
   ```
   - frames from "Captured_frames" are preprocessed with Mediapipe hand detection module, and landmarks are saved in "npy_files"


5. **Run "Annonated_frames.py"**
   
   ```bash
   Python Annonated_frames.py
   ```
   - "npy_files" are annonated with "Captured_frames" and saved in "Annonated_frames"


7. **Run "train_model.py"**
   
    ```bash
   Python train_model.py
   ```
    - SVM Classifier is trained with "npy_files" with proper labeling and saved in "Trained_model"

   


## Contributing

   
Contributions are welcome! If you have suggestions or improvements, feel free to create a pull request.
