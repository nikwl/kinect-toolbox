import cv2
from kinecttb import Kinect, KFrame

k = Kinect()
while True:
    # Specify as many types as you want here
    color_frame = k.get_frame(KFrame.COLOR)

    cv2.imshow('frame', color_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break