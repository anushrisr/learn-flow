This repository contains the following contents.

Sample program
Hand sign recognition model(TFLite)
Finger gesture recognition model(TFLite)
Learning data for hand sign recognition and notebook for learning
Learning data for finger gesture recognition and notebook for learning

Requirements
mediapipe 0.8.1
OpenCV 3.4.2 or Later
Tensorflow 2.3.0 or Later
tf-nightly 2.5.0.dev or later (Only when creating a TFLite for an LSTM model)
scikit-learn 0.23.2 or Later (Only if you want to display the confusion matrix)
matplotlib 3.3.2 or Later (Only if you want to display the confusion matrix)
Demo
Here's how to run the demo using your webcam.

python app.py
The following options can be specified when running the demo.

--device
Specifying the camera device number (Default：0)
--width
Width at the time of camera capture (Default：960)
--height
Height at the time of camera capture (Default：540)
--use_static_image_mode
Whether to use static_image_mode option for MediaPipe inference (Default：Unspecified)
--min_detection_confidence
Detection confidence threshold (Default：0.5)
--min_tracking_confidence
Tracking confidence threshold (Default：0.5)
Directory
│  app.py
│  keypoint_classification.ipynb
│  point_history_classification.ipynb
│  
├─model
│  ├─keypoint_classifier
│  │  │  keypoint.csv
│  │  │  keypoint_classifier.hdf5
│  │  │  keypoint_classifier.py
│  │  │  keypoint_classifier.tflite
│  │  └─ keypoint_classifier_label.csv
│  │          
│  └─point_history_classifier
│      │  point_history.csv
│      │  point_history_classifier.hdf5
│      │  point_history_classifier.py
│      │  point_history_classifier.tflite
│      └─ point_history_classifier_label.csv
│          
└─utils
    └─cvfpscalc.py
app.py
This is a sample program for inference.
In addition, learning data (key points) for hand sign recognition,
You can also collect training data (index finger coordinate history) for finger gesture recognition.

keypoint_classification.ipynb
This is a model training script for hand sign recognition.

point_history_classification.ipynb
This is a model training script for finger gesture recognition.

model/keypoint_classifier
This directory stores files related to hand sign recognition.
The following files are stored.

Training data(keypoint.csv)
Trained model(keypoint_classifier.tflite)
Label data(keypoint_classifier_label.csv)
Inference module(keypoint_classifier.py)
model/point_history_classifier
This directory stores files related to finger gesture recognition.
The following files are stored.

Training data(point_history.csv)
Trained model(point_history_classifier.tflite)
Label data(point_history_classifier_label.csv)
Inference module(point_history_classifier.py)
utils/cvfpscalc.py
This is a module for FPS measurement.