# Animal Hand Sign Model Training

## Overview

Each and every required files and folders while training the SVM Classifier with linear kernel and True probabilty are included in this repository for future uses and to share among the ML enthusiasts.


## Dataset

- "video_files" ------> ".mp4" -------> "open-cv && Mediapipe" -------> hand_landmarks


## Requirements

It is recommended to use a virtual environment to manage dependencies. Follow these steps:

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



## How I Trained the Model

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



## Contributing

   
Contributions are welcome! If you have suggestions or improvements, feel free to create a pull request.
