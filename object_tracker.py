import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from absl import app, flags, logging
from absl.flags import FLAGS

# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# tensorflow yolo core imports
from tf_yolo.core.utils import read_class_names
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
# YOLO detectors
from detectors import TfYOLO, TfliteYOLO, TrtYOLO


flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './data/tf/yolov4-416', 'path to weights file')
flags.DEFINE_string('names', './data/classes/coco.names', 'path to yolo names file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')


def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    video_path = FLAGS.video

    # read in all class names from config
    class_names = read_class_names(FLAGS.names)

    # by default allow all classes in .names file
    allowed_classes = list(class_names.values())

    # custom allowed classes (uncomment line below to customize tracker for only people)
    # allowed_classes = ['person']

    detector_config = {
        'model_path': FLAGS.weights,
        'iou_threshold': FLAGS.iou, 
        'score_threshold': FLAGS.score
    }

    # load tf saved model
    if FLAGS.framework == 'tf':
        detector_config['input_size'] = FLAGS.size

        detector = TfYOLO(**detector_config)
    # load tflite model if flag is set
    elif FLAGS.framework == 'tflite':
        detector_config['input_size'] = FLAGS.size
        detector_config['model_type'] = FLAGS.model
        detector_config['tiny'] = FLAGS.tiny

        detector = TfliteYOLO(**detector_config)
    # load tensorrt engine if flag is set
    elif FLAGS.framework == 'trt':
        detector = TrtYOLO(**detector_config)

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    all_time = 0

    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        print('Frame #: ', frame_num)
        
        # preprocess frame
        # TODO: move preprocessing from detectors

        # start frame inference timer
        start_time = time.time()

        # get detections from detector
        bboxes, scores, classes, num_objects, detection_time = detector.detect(frame)

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        # names = []
        # deleted_indx = []
        # for i in range(num_objects):
        #     class_indx = int(classes[i])
        #     class_name = class_names[class_indx]
        #     if class_name not in allowed_classes:
        #         deleted_indx.append(i)
        #     else:
        #         names.append(class_name)
        # names = np.array(names)
        # count = len(names)

        # if FLAGS.count:
        #     cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
        #     print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        # bboxes = np.delete(bboxes, deleted_indx, axis=0)
        # scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
            # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

            # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        # calculate frames per second of running detections
        inference_time = time.time() - start_time
        all_time += inference_time
        fps = 1.0 / inference_time
        print("FPS: %.2f" % fps)

        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()
    
    print("Inference total time: %.4f" % all_time)
    print("Frames: %.2f" % frame_num)
    mean_time = all_time / frame_num
    mean_fps = frame_num / all_time
    print("Mean inference time: %.4f" % mean_time)
    print("Mean FPS: %.4f" % mean_fps)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
