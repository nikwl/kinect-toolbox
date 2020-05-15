# kinect-toolbox
## Overview
This is a set of helper fuctions that I wrote to make using the Microsoft Kinect V2 with python easier. In my opinion the "C++ like" terminology of the pylibfreenect2 library is cumbersome and diffult. With this wrapper the kinect can be used more like a cv2 webcam. Also adds an easy method to obtain and view point clouds. Based partially on the [stack overflow question](https://stackoverflow.com/questions/41241236/vectorizing-the-kinect-real-world-coordinate-processing-algorithm-for-speed) asked by user Logic1, which I credit for some of the optimization and visualization code.

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

## Usage
```python
import cv2

from Kinect.Kinect import Kinect
from Kinect.KinectIm import KinectIm

# Create the kinect object
k = Kinect()

while True:
    # Specify as many types as you need here
    color_frame = k.get_frame([KinectIm.COLOR])[0]

    cv2.imshow('frame', color_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

## Methods
#### get_frame
Get a frame from the kinect. Returns a list of opencv style images corresponding to the list of image types passed.

#### get_ptcld, get_color_ptcld
Get a point cloud from the kinect. Returns an image of size [width, height, 3], corresponding to xyz points. Color version returns two images, one of xyz points and the other of rgb colors. 

#### get_perspective_depth
Get a simulated depth map from somewhere in the scene by specifying a principal point, viewing vector, and fov.

#### PtcldViewer
Spawn a Qt viewer to visualize point clouds from the kinect in real time. 

#### Streamer
Use imagezmq to bounce an image stream between a computer and server for inference.