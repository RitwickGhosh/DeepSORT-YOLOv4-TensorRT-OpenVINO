import os
import cv2
import time
import click
import numpy as np
import matplotlib.pyplot as plt

# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
# YOLO detectors
from detectors import TfYOLO, TfliteYOLO, TrtYOLO


@click.command()
@click.option('-f', '--framework', default='tf', type=str, help='Inference framework: {tf, tflite, trt}')
@click.option('-m', '--model_path', default='./data/tf/yolov4-416', type=str, help='Path to detection model')
@click.option('-n', '--yolo_names', default='./data/classes/coco.names', type=str, help='Path to YOLO class names file')
@click.option('-s', '--size', default=416, type=int, help='Model input size')
@click.option('-t', '--tiny', default=False, type=bool, help='If YOLO tiny architecture')
@click.option('-t', '--model_type', default='yolov4', type=str, help='yolov3 or yolov4k')
@click.option('-v', '--video_path', default='./data/video/test.mp4', type=str, help='Path to input video')
@click.option('-o', '--output', default=None, type=str, help='Path to output, inferenced video')
@click.option('--output_format', default='XVID', type=str, help='Codec used in VideoWriter when saving video to file')
@click.option('--iou', default=0.45, type=float, help='IoU threshold')
@click.option('--score_threshold', default=0.5, type=float, help='Confidence score threshold')
@click.option('--dont_show', default=True, type=bool, help='Do not show video output')
@click.option('--info', default=False, type=bool, help='Show detailed info of tracked objects')
@click.option('--count', default=False, type=bool, help='Count objects being tracked on screen')
def main(framework, model_path, yolo_names, size, tiny, model_type, video_path, output, output_format, iou, score_threshold, dont_show, info, count):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'data/deep_sort_model/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # read in all class names from config
    with open(yolo_names, 'r') as f:
        class_names = f.read().split('\n')

    # by default allow all classes in .names file
    allowed_classes = class_names

    # custom allowed classes (uncomment line below to customize tracker for only people)
    # allowed_classes = ['person']

    detector_config = {
        'model_path': model_path,
        'iou_threshold': iou, 
        'score_threshold': score_threshold
    }

    # load tf saved model
    if framework == 'tf':
        detector_config['input_size'] = size

        detector = TfYOLO(**detector_config)
    # load tflite model if flag is set
    elif framework == 'tflite':
        detector_config['input_size'] = size
        detector_config['model_type'] = model_type
        detector_config['tiny'] =tiny

        detector = TfliteYOLO(**detector_config)
    # load tensorrt engine if flag is set
    elif framework == 'trt':
        detector = TrtYOLO(**detector_config)

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    # get video ready to save locally if flag is set
    if output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*output_format)
        out = cv2.VideoWriter(output, codec, fps, (width, height))

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

        # start frame inference timer
        start_time = time.time()

        # get detections from detector
        bboxes, scores, classes, num_objects, detection_time = detector.detect(frame)

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for class_indx in classes.astype(int):
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)

        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)
        names = np.array(names)

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

        # call the tracker
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
            if info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        # calculate frames per second of running detections
        inference_time = time.time() - start_time
        all_time += inference_time
        fps = 1.0 / inference_time
        print("FPS: %.2f" % fps)

        if count:
            object_count = len(names)
            cv2.putText(frame, "Objects being tracked: {}".format(object_count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(object_count))

        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()
    
    print("Inference total time: %.4f" % all_time)
    print("Frames: ", frame_num)
    mean_time = all_time / frame_num
    mean_fps = frame_num / all_time
    print("Mean inference time: %.4f" % mean_time)
    print("Mean FPS: %.4f" % mean_fps)


if __name__ == '__main__':
    main()
