import cv2
import time
import numpy as np

from .utils import format_boxes

class OpenCVYOLO(object):
    """OpenCVYOLO class - inference class for YOLO model in OpenCV"""

    def __init__(self, model_path, cfg_file, input_size, iou_threshold, score_threshold, opencv_dnn_target='CPU'):
        self.cfg_file = cfg_file
        self.weights_file = model_path
        self.input_size = input_size
        self.confidence_threshold = score_threshold
        self.nms_threshold = iou_threshold
        self.opencv_dnn_target = opencv_dnn_target

        self.net = cv2.dnn.readNetFromDarknet(self.cfg_file, self.weights_file)

        if self.opencv_dnn_target == 'MYRIAD':
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
        elif self.opencv_dnn_target == 'FP16':
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        elif self.opencv_dnn_target == 'FP32':
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(self.input_size, self.input_size), scale=1/255)

    def __del__(self):
        """Free device memories."""
        del self.model

    def _preprocess(self, frame):
        pass

    def _postprocess(self, bboxes, classes, scores, original_h, original_w):
        pass
        

    def detect(self, img):
        """Detect objects in the input image."""
        batch_data = img

        start_time = time.time()

        classes, scores, bboxes = self.model.detect(batch_data, confThreshold=self.confidence_threshold, nmsThreshold=self.nms_threshold)

        stop_time = time.time()

        return bboxes, np.array(scores).flatten(), np.array(classes).flatten(), len(classes), stop_time-start_time
