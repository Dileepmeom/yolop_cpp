# YOLOP C++ Inference

## Note
The following steps are applicable only for Ubuntu machines.

## Steps to Build
```bash
cmake .. -DONNXRUNTIME_ROOT=/home/ridebuddy/Documents/libraries/onnxruntime-linux-x64-1.21.0
make
```

## How to Run
./yolop_infer \
  /home/ridebuddy/Documents/yolop_cpp/yolop_cpp_320/yolop-320-320.onnx \
  /home/ridebuddy/Documents/yolop_cpp/yolop_cpp_320/images \
  /home/ridebuddy/Documents/yolop_cpp/yolop_cpp_generic \
  320 320