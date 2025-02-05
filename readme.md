### Export path needed to set before running the binmary to assign the path of the librart
#### mostly its needed for the opencv atleast in the ubuntu machine which I am using

###### export LD_LIBRARY_PATH=/home/ridebuddy/Documents/yolop_cpp/lite-ort1.17.1+ocv4.9.0+ffmpeg4.2.2-linux-x86_64/include:$LD_LIBRARY_PATH

##### export LD_LIBRARY_PATH=/home/ridebuddy/Documents/yolop_cpp/lite-ort1.17.1+ocv4.9.0+ffmpeg4.2.2-linux-x86_64/third_party/opencv/lib:$LD_LIBRARY_PATH

##### export LD_LIBRARY_PATH=/home/ridebuddy/Documents/yolop_cpp/lite-ort1.17.1+ocv4.9.0+ffmpeg4.2.2-linux-x86_64/third_party/onnxruntime:$LD_LIBRARY_PATH

DA Inference Times for 640*640 (ms):
Image 1: 1584.74 ms
Image 2: 1532.75 ms
Image 3: 1405.35 ms
Image 4: 1260.99 ms
Image 5: 1430.67 ms
Image 6: 1354.14 ms
Image 7: 1568.18 ms
