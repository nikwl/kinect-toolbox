# kinect-toolbox
## Overview
This is a set of helper functions to make using the Microsoft Kinect V2 with python easier. Libfreenect2 provides a robust interface for the kinect but it follows a more "C++ like" design paradigm. With this wrapper the kinect can be used more like a cv2 webcam. This package provides methods to get color, depth, registered color, registered depth, and ir images, record video, get point clouds (quickly), and makes the kinect interface all around more pythonic.

Credit for point cloud acceleration methods goes to stackoverflow user [Logic1](https://stackoverflow.com/questions/41241236/vectorizing-the-kinect-real-world-coordinate-processing-algorithm-for-speed).

## Installation
1) [Install libfreenect2](https://github.com/OpenKinect/libfreenect2) 
2) Update your paths:
    ```bash
    export LIBFREENECT2_INSTALL_PREFIX=~/freenect2
    export LD_LIBRARY_PATH=$HOME/freenect2/lib:$LD_LIBRARY_PATH
    ```
3) Install required python packages:
    ```bash
    pip install -r requirements.txt
    ``` 
4) Test installation:
    ```bash
    python test.py
    ```
This package has been tested on Ubuntu 18.04 with python3.6.

## Usage
```python
import cv2
import ktb

k = ktb.Kinect()
while True:
    # Specify as many types as you want here
    color_frame = k.get_frame(ktb.COLOR)

    cv2.imshow('frame', color_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

## Methods
#### `Kinect()`
The kinect class. Supports the following input arguments:
* params_file: Kinect parameters file or dict. Parameters file can contain the kinect intrinsic parameters and a 3D transformation (which is applied to point clouds).
* device_index: Use to interface with a specific device if more than one is connected. 
* headless: If true, will allow the kinect to collect and process data without a display. Connect the Kinect to a server and collect data remotely!
* pipeline: Optionally pass a pylibfreenect2 pipeline that will be used with the kinect. Note that this will override the headless argument - Headless requires CUDA or OpenCL pipeline. Possible types are as follows:
    * OpenGLPacketPipeline
    * CpuPacketPipeline
    * OpenCLPacketPipeline
    * CudaPacketPipeline
    * OpenCLKdePacketPipeline

#### `Kinect.get_frame()`
Get a video frame from the kinect. Optionally specify a list of image types and the function will return a corresponding list of images. Available types are:

| Frame Type | Description |
|-|-|
| RAW_COLOR | returns a 1920 x 1080 color image |
| RAW_DEPTH | returns a 512 x 424 depth image (units are mm) |
| COLOR     | returns a 512 x 424 color image, registered to depth image |
| DEPTH     | returns a 512 x 424 undistorted depth, image (units are mm) |
| IR        | returns a 512 x 424 ir image |

Note that depth images have units in millimeters. To display them in a window without clipping first rescale the values:
```python
frame = frame / 4500.
```

#### `Kinect.record()`
Records a video from the kinect. Halts execution and opens a cv2 window displaying a stream from the kinect while writing the stream to disk. Optionally specify any of the above frame types to record that specific stream.

#### `Kinect.get_ptcld()`
Get a point cloud from the kinect. Returns an point cloud of size [512, 424, 3], corresponding a grid of xyz points. Optionally return a color map derived from the registered color image.