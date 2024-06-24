import numpy as np
import argparse
import cv2


def initializeParser():
    parser = argparse.ArgumentParser(description="Object Detection Script")
    parser.add_argument("-source", type=int, choices=[0, 1],
                        help="0 for webcam 1 for video file")
    parser.add_argument("-video_path", type=str, help="Path to the video file (required if source is 0)")
    return parser.parse_args()

class ObjectDetection:
    def __init__(self):
        self.thresh = 0.5  # Threshold to detect object
        self.nms_threshold = 0.2  # (0.1 to 1) 1 means no suppress , 0.1 means high suppress
        self.weightsPath = "frozen_inference_graph.pb"
        self.configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
        self.Colors = []
        self.classNames = []
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.cap = None
        self.net = None
        
    def initialize(self, args):
        source = 0
        if args.source == 1:
            if args.video_path is not None:
                source = args.video_path
            else:
                source = "traffic.mp4"
            
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 280) # width 
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120) # height 
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 150) # brightness 

        with open('coco.names', 'r') as f:
            self.classNames = f.read().splitlines()
        print(self.classNames)
        
        self.Colors = np.random.uniform(0, 255, size=(len(self.classNames), 3))
        self.net = cv2.dnn.DetectionModel(self.weightsPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)
    
    def run(self):
        while True:
            success, img = self.cap.read()
            classIds, confs, bbox = self.net.detect(img, confThreshold=self.thresh)
            bbox = list(bbox)
            confs = list(np.array(confs).reshape(1, -1)[0])
            confs = list(map(float, confs))

            indices = cv2.dnn.NMSBoxes(bbox, confs, self.thresh, self.nms_threshold)
            if len(classIds) != 0:
                for i in indices:
                    box = bbox[i]
                    confidence = str(round(confs[i], 2))
                    color = self.Colors[classIds[i] - 1]
                    x, y, w, h = box[0], box[1], box[2], box[3]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness=2)
                    cv2.putText(img, self.classNames[classIds[i] - 1] + " " + confidence, (x + 10, y + 20),
                                self.font, 1, color, 2)

            cv2.imshow("Output", img)
            cv2.waitKey(1)

if __name__ == "__main__":
    objectDetection = ObjectDetection()
    objectDetection.initialize(initializeParser())
    objectDetection.run()