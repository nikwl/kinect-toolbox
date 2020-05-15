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

