# YOLOP C++ Inference for Pi ARM architecture ( Pi Device )

## Note
The following steps are applicable only for Ubuntu machines.
  ## Steps to Build
  ```bash
  cmake .. -DONNXRUNTIME_ROOT=/home/ridebuddy/Documents/libraries/onnxruntime-linux-x64-1.21.0
  make
  ```
  ## How to Run
  ./yolop_infer ../yolop-320-320.onnx ../Images ../ 320 320