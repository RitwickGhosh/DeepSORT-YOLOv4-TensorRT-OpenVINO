import cv2
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from .utils import format_boxes


class TfYOLO(object):
    """TfYOLO class - inference class for YOLO model in Tensorflow"""

    def __init__(self, model_path, input_size, iou_threshold, score_threshold):
        """Initialize TensorRT plugins, engine and conetxt."""
        self.model_path = model_path
        self.input_size = input_size
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

        self.model = tf.saved_model.load(self.model_path, tags=[tag_constants.SERVING])
        self.infer = self.model.signatures['serving_default']

    def __del__(self):
        """Free device memories."""
        del self.model

    def _preprocess(self, frame):
        image_data = cv2.resize(frame, (self.input_size, self.input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        
        batch_data = tf.constant(image_data)
        
        return batch_data

    def _postprocess(self, pred_bbox, original_h, original_w):
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        bboxes = format_boxes(bboxes, original_h, original_w)

        return bboxes, scores, classes, num_objects
        

    def detect(self, img):
        """Detect objects in the input image."""
        batch_data = self._preprocess(img)

        start_time = time.time()

        pred_bbox = self.infer(batch_data)

        stop_time = time.time()

        bboxes, scores, classes, num_objects = self._postprocess(pred_bbox, original_h=img.shape[0], original_w=img.shape[1])

        return bboxes, scores, classes, num_objects, stop_time-start_time
