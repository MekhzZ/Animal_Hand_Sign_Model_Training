# Animal Hand Sign Model Training

## Overview

Each and every required files and folders while training the SVM Classifier with linear kernel and True probabilty are included in this repository for future uses and to share among the ML enthusiasts.


## Dataset

- "video_files" ------> ".mp4" -------> "open-cv && Mediapipe" -------> hand_landmarks


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



## Requirements & Workflow

If you want to explore the working principles or my workflow to train the model. For that, ensure you have the following installed:

- Python 3.x
- OpenCV
- Mediapipe
- Scikit-learn
- Joblib
- NumPy

### Virtualenv + Required Packages

It is recommended to use a virtual environment to manage dependencies.

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

### Workflow

1. **Run "captured_frames.py"**
   ```bash
   Python captured_frames.py
   ```

2. **Run "Mediapipe.py"**
   ```bash
   Python Mediapipe.py
   ```

3. **Run "Annonated_frames.py"**
   ```bash
   Python Annonated_frames.py
   ```

4. **Run "train_model.py"**
    ```bash
   Python train_model.py
   ```



## Contributing

   
Contributions are welcome! If you have suggestions or improvements, feel free to create a pull request.
