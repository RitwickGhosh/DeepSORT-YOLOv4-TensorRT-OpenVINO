# YOLOv4 conversion to ONNX and TensorRT

## References

Many thanks for great job for [jkjung-avt](https://github.com/jkjung-avt) and his [tensorrt_demos](https://github.com/jkjung-avt/tensorrt_demos) repository, originally published with MIT License.

## YOLOv4 -> ONNX

```bash
python yolo_to_onnx.py -c 80 -m ./yolov4 -o ./yolov4.onnx
```

## ONNX -> TensorRT

```bash
# convert ONNX to tensorrt engine with float32 weights
python onnx_to_tensorrt.py -v -c 80 -m ./yolov4.onnx -q fp32 -o ./yolov4-fp32.trt

# convert ONNX to tensorrt engine with float16 weights
python onnx_to_tensorrt.py -v -c 80 -m ./yolov4.onnx -q fp16 -o ./yolov4-fp16.trt

# convert ONNX to tensorrt engine with int8 weights
# needs path to calibration dataset (representative images from dataset), marked below as './calib_images' 
python onnx_to_tensorrt.py -v -c 80 -m ./yolov4.onnx -i ./calib_images -q int8 -o ./yolov4-int8.trt
```

<p align="center"><img src="https://developer.nvidia.com/sites/default/files/akamai/tensorrt/nvidia-tensorrt-infographic.svg"\></p>
