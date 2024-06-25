# Real-Time Object Detection with OpenCV and SSD MobileNet v3

This project demonstrates real-time object detection using OpenCV and a pre-trained SSD MobileNet v3 deep learning model. It can process live video feeds from a webcam or analyze objects within a video file.

## Features

*   Real-time Object Detection: Quickly identifies and locates objects in video streams.
*   SSD MobileNet v3 Architecture: Employs a powerful and efficient deep learning model for object detection.
*   Customizable: Easily adjust parameters like detection threshold and non-maximum suppression.
*   Multiple Input Sources: Supports both webcam input and video files.

## Requirements

*   Python 3.x
*   OpenCV (cv2)
*   NumPy (np)
*   TensorFlow or other deep learning framework (if you need to fine-tune or retrain the model)

## Installation

1.  Clone the repository:

    ```bash
    git clone [https://github.com/your_username/your_repository_name.git](https://github.com/your_username/your_repository_name.git)
    ```

2.  Install dependencies:

    ```bash
    pip install opencv-python numpy
    ```

## Download Model Files

Ensure you have `frozen_inference_graph.pb` (model weights) and `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt` (model configuration) in the same directory as your script. You can download them from the TensorFlow Model Zoo or other sources.

## Usage

Run with Webcam:

```bash
python object_detector.py -source 1
```

Run with Video File:
```bash
python object_detector.py -source 0 -video_path /path/to/your/video.mp4
```

## Customizations

* Detection Threshold: Adjust the thresh variable in the ObjectDetection class to control the confidence level required for object detection.
* Non-Maximum Suppression Threshold: Modify the nms_threshold variable to adjust how aggressively overlapping bounding boxes are suppressed.
