### Export path needed to set before running the binmary to assign the path of the librart
#### mostly its needed for the opencv atleast in the ubuntu machine which I am using

###### export LD_LIBRARY_PATH=/home/ridebuddy/Documents/yolop_cpp/lite-ort1.17.1+ocv4.9.0+ffmpeg4.2.2-linux-x86_64/include:$LD_LIBRARY_PATH

##### export LD_LIBRARY_PATH=/home/ridebuddy/Documents/yolop_cpp/lite-ort1.17.1+ocv4.9.0+ffmpeg4.2.2-linux-x86_64/third_party/opencv/lib:$LD_LIBRARY_PATH

##### export LD_LIBRARY_PATH=/home/ridebuddy/Documents/yolop_cpp/lite-ort1.17.1+ocv4.9.0+ffmpeg4.2.2-linux-x86_64/third_party/onnxruntime:$LD_LIBRARY_PATH

| Image | 640x640 (ms) | 320 res (ms) |
|-------|--------------|--------------|
| 1     | 1584.74      | 272.254      |
| 2     | 1532.75      | 460.045      |
| 3     | 1405.35      | 313.608      |
| 4     | 1260.99      | 399.612      |
| 5     | 1430.67      | 405.268      |
| 6     | 1354.14      | 436.198      |
| 7     | 1568.18      | 349.926      |


