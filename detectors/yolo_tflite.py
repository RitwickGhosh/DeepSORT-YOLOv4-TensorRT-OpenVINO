import cv2
import time
import numpy as np
import tensorflow as tf
# import tflite_runtime.interpreter as tflite

from .utils import format_boxes, filter_boxes


class TfliteYOLO(object):
    """TfliteYOLO class - inference class for YOLO model in Tensorflow Lite"""

    def __init__(self, model_path, input_size, iou_threshold, score_threshold, model_type='yolov4', tiny=False):
        """Initialize TensorRT plugins, engine and conetxt."""
        self.model_path = model_path
        self.input_size = input_size
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.model_type = model_type
        self.tiny = tiny

        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __del__(self):
        """Free device memories."""
        del self.interpreter

    def _preprocess(self, frame):
        image_data = cv2.resize(frame, (self.input_size, self.input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        
        return image_data

    def _postprocess(self, pred_bbox, original_h, original_w):
        # run detections using yolov3 if flag is set
        if self.model_type == 'yolov3' and self.tiny == True:
            boxes, pred_conf = filter_boxes(pred_bbox[1], pred_bbox[0], score_threshold=self.score_threshold, 
                                            input_shape=(self.input_size, self.input_size))
        else:
            boxes, pred_conf = filter_boxes(pred_bbox[0], pred_bbox[1], score_threshold=self.score_threshold, 
                                            input_shape=(self.input_size, self.input_size))

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

        # num_objects = len(boxes[0])
        # bboxes = boxes[0]
        # scores = np.ones((num_objects)) * 0.75
        # # classes = classes[0]
        # classes = np.ones((num_objects))

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        bboxes = format_boxes(bboxes, original_h, original_w)

        return bboxes, scores, classes, num_objects
        

    def detect(self, img):
        """Detect objects in the input image."""
        batch_data = self._preprocess(img)

        start_time = time.time()

        self.interpreter.set_tensor(self.input_details[0]['index'], batch_data)
        self.interpreter.invoke()
        pred_bbox = [self.interpreter.get_tensor(self.output_details[i]['index']) for i in range(len(self.output_details))]

        stop_time = time.time()

        bboxes, scores, classes, num_objects = self._postprocess(pred_bbox, original_h=img.shape[0], original_w=img.shape[1])

        return bboxes, scores, classes, num_objects, stop_time-start_time
