# YOLOv4 conversion to TensorFlow Lite

## YOLOv4 -> ONNX
Script for conversion to ONNX format is available in [tensorrt](../tensorrt/yolo_to_onnx.py) directory. 

```bash
python yolo_to_onnx.py -c 80 -m ./yolov4 -o ./yolov4.onnx
```

## ONNX -> TensorFlow

The conversion between ONNX and TensorFlow SavedModel is available with the use of onnx (onnx-tensorflow package) command line interface:

```bash
onnx-tf convert -i ./yolov4.onnx -o ./yolov4
```

## TensorFlow -> TensorFlow Lite

```bash
# convert ONNX to tflite with float32 weights
python tf2tflite.py -m ./yolov4/ -q fp32 -o ./yolov4_fp32.tflite

# convert ONNX to tflite with float16 weights
python tf2tflite.py -m ./yolov4/ -q fp16 -o ./yolov4_fp16.tflite
```

<p align="center"><img src="https://www.tensorflow.org/lite/images/convert/convert.png"\></p>
