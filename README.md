# DeepSort YOLOv4 based object tracking

The repository contains the implementation of [DeepSort](https://arxiv.org/abs/1703.07402) object tracking based on YOLOv4 detections. Detector inference class is implemented in several frameworks like TensorFlow, TensorFlow Lite, TensorRT, OpenCV, and OpenVINO to benchmark methods and use the best one for edge-tailored solutions.

## How to use

```bash
python3 object_tracker.py -f trt -m <PATH TO MODEL> -s <MODEL INPUT SIZE> -n <PATH TO FILE WITH CLASS NAMES> -v <PATH TO INPUT VIDEO> --dont_show True
```

### Command line arguments

```bash
Usage: object_tracker.py [OPTIONS]

Options:
  -f, --framework TEXT            Inference framework: {tf, tflite, trt,
                                  opencv, openvino}
  -m, --model_path TEXT           Path to detection model
  -n, --yolo_names TEXT           Path to YOLO class names file
  -s, --size INTEGER              Model input size
  -t, --tiny BOOLEAN              If YOLO tiny architecture
  -t, --model_type TEXT           yolov3 or yolov4
  -v, --video_path TEXT           Path to input video
  -o, --output TEXT               Path to output, inferenced video
  --output_format TEXT            Codec used in VideoWriter when saving video
                                  to file
  --iou FLOAT                     IoU threshold
  --score_threshold FLOAT         Confidence score threshold
  --opencv_cuda_target_precision TEXT
                                  Precision of OpenCV DNN model
  --dont_show BOOLEAN             Do not show video output
  --info BOOLEAN                  Show detailed info of tracked objects
  --count BOOLEAN                 Count objects being tracked on screen
  --help                          Show this message and exit.
```

## Performance tests

The benchmark tests were performed on [NVIDIA Jetson Xavier NX](https://developer.nvidia.com/embedded/jetson-xavier-nx) and [Intel Neural Compute Stick 2](https://software.intel.com/content/www/us/en/develop/hardware/neural-compute-stick.html). Jetson Xavier NX was in mode 2 (`sudo nvpmodel -m 0`) and clocks, fan were set to the maximum frequency with `sudo jetson clocks --fan` command. To evaluate Intel INCS 2 Raspberry Pi 4B was used.

The performance tests were done on the [VisDrone MOT dataset](http://aiskyeye.com/).

## Use cases

The possible use cases of multiple object tracking are ...

## References

Many thanks for great job for:

- [The AI Guy](https://github.com/theAIGuysCode): [yolov4-deepsort](https://github.com/theAIGuysCode/yolov4-deepsort), MIT License

- [nwojke](https://github.com/nwojke): [deep_sort](https://github.com/nwojke/deep_sort), MIT License

- [hunglc007](https://github.com/hunglc007): [tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite), MIT License

- [jkjung-avt](https://github.com/jkjung-avt): [tensorrt_demos](https://github.com/jkjung-avt/tensorrt_demos), MIT License
