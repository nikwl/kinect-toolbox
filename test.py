import cv2
import ktb

k = ktb.Kinect()

while True:
    color_frame = k.get_frame()
    
    cv2.imshow('frame', color_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break