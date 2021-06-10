# YOLOv4 conversion to Tensorflow and Tensorflow Lite

## References

Many thanks for great job for [hunglc007](https://github.com/hunglc007) and his [tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite) repository, originally published with MIT License.

## Use cases

**NOTE** If your YOLO detector differ from COCO trained detector (different class number, strides, anchors, etc.) change values in [core/config.py](./core/config.py) 

### YOLOv4 -> TF

```bash
python save_model.py --weights ./yolov4.weights --output ./yolov4/ --input_size 416
```

### YOLOv4 -> tflite

```bash
# save YOLO weights for tflite model 
python save_model.py --weights ./yolov4.weights --output ./yolov4-tflite/ --input_size 416 --framework tflite

# convert tf to tflite float32 weights
python convert_tflite.py --weights ./yolov4-tflite/ --output ./yolov4-fp32.tflite --input_size 416 --quantize_mode float32

# convert tf to tflite float16 weights
python convert_tflite.py --weights ./yolov4-tflite/ --output ./yolov4-fp32.tflite --input_size 416 --quantize_mode float16

# convert tf to tflite int8 weights
# doesn't work now 
```
