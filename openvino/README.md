# YOLOv4 conversion to OpenVINO (IR)

## YOLOv4 -> ONNX

Script for conversion to ONNX format is available in [tensorrt](../tensorrt/yolo_to_onnx.py) directory. 

```bash
python yolo_to_onnx.py -c 80 -m ./yolov4 -o ./yolov4.onnx
```

## ONNX -> OpenVINO (IR)

Conversion from ONNX to Intermediate Representation (IR) requires usage of OpenVINO toolkit with Model Optimizer package. It is a part of tool and is in `<OPENVINO PATH>/deployment_tools/model_optimizer/` directory. Before running `mo.py` script it is necessary to source toolkit with command `source <OPENVINO PATH>/bin/setupvars.sh `.

```bash
# convert ONNX to OpenVINO IR with float32 weights
python mo.py --input_model ./yolov4.onnx --model_name yolov4_fp32 --data_type FP32 --batch 1

# convert ONNX to OpenVINO IR with float16 weights
python mo.py --input_model ./yolov4.onnx --model_name yolov4_fp16 --data_type FP16 --batch 1
```

![OpenVINO](https://miro.medium.com/max/761/1*75Tym19aH6nKF-c9SZlxNQ.png)
