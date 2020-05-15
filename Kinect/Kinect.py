# To use, need to run this first
# export LIBFREENECT2_INSTALL_PREFIX=~/freenect2
# export LD_LIBRARY_PATH=$HOME/freenect2/lib:$LD_LIBRARY_PATH

import sys
import os
import cv2
import json
import math
import numpy as np

from pyquaternion import Quaternion

from scipy.interpolate import griddata

from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame

from .KinectIm import KinectIm

try:
    from pylibfreenect2 import OpenGLPacketPipeline
    pipeline = OpenGLPacketPipeline()
except:
    from pylibfreenect2 import CpuPacketPipeline
    pipeline = CpuPacketPipeline()
print("Packet pipeline:", type(pipeline).__name__)

DEPTH_NORMALIZATION = 4.5 # Only applies to distorted depth
IR_NORMALIZATION = 65535.
DEPTH_SHAPE = (int(512), 
               int(424), 
               int(4))
COLOR_SHAPE = (int(1920), 
               int(1080))

def get_image_center_point(image, return_coords=False):
	s = image.shape
	r, c = int(s[0]/2), int(s[1]/2)
	if return_coords:
		return image[r, c,...], (r,c)
	return image[r, c,...]

class Kinect():
    def __init__(self, conf_file=None):
        '''
            Kinect: A wrapper for the microsoft kinect using the pylibfreenect2 library.
                Will collect and return color, depth, ir, and point cloud images. 
            ARGUMENTS:
                conf_file: (str)
                    Path to a configuration file. This file should contain the kinect
                    camera's worldspace position. 
        '''

        self.fn = Freenect2()
        num_devices = self.fn.enumerateDevices()

        if num_devices == 0:
            print("No device connected!")
            sys.exit(1)

        self.serial = self.fn.getDeviceSerialNumber(0)
        self.device = self.fn.openDevice(self.serial, pipeline=pipeline)

        types = 0
        types |= (FrameType.Color | FrameType.Depth | FrameType.Ir)
        
        self.listener = SyncMultiFrameListener(types)

        self.device.setColorFrameListener(self.listener)
        self.device.setIrAndDepthFrameListener(self.listener)

        self.device.setColorFrameListener(self.listener)
        self.device.setIrAndDepthFrameListener(self.listener)
        self.device.start()

        self.registration = Registration(self.device.getIrCameraParams(),
                                         self.device.getColorCameraParams())
   
        # Kinects's intrinsic parameters based on v2 hardware
        self.CameraParams = {
            "cx":self.device.getIrCameraParams().cx,
            "cy":self.device.getIrCameraParams().cy,
            "fx":self.device.getIrCameraParams().fx,
            "fy":self.device.getIrCameraParams().fy
        }

        # Load the kinect position
        self.CameraPosition = self.load_camera_position(conf_file)

    def __del__(self):
        self.device.stop()
        self.device.close()

    def load_camera_position(self, conf_file):
        '''
            load_camera_position: Kinect needs an understanding of where it is in the 
                worldspace in order to rectify point clouds. This file should contain 
                this information. 
            ARGUMENTS:
                conf_file: (str)
                    Path to a configuration file. This file should contain the kinect
                    camera's worldspace position.    
        '''
        # Load kinect configuration (worldspace attributes) from file
        if (conf_file is None):
            conf_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'kinect.conf')
        if (os.path.exists(conf_file)): 
            with open(conf_file, 'r') as infile:
                save_dict = json.load(infile)
        else:
            print('Conf file not found: ' + conf_file)
            exit(-1)
        
        return save_dict

    def get_frame(self, frame_type=None):
        '''
            get_frame: Gets a list of images corresponding to the input list of types.
                KinectIm.COLOR     - returns a 512 x 424 color image, registered to depth image
                KinectIm.DEPTH     - returns a 512 x 424 undistorted depth image (units are mm)
                KinectIm.IR        - returns a 512 x 424 ir image
                KinectIm.RAW_COLOR - returns a 1920 x 1080 color image
                KinectIm.RAW_DEPTH - returns a 512 x 424 depth image (units are mm)
                Will return a list of images regardless of the number of arguments.
            ARGUMENTS:
                frame_type: ([KinectIm])
                    A list of frame types. Output corresponds directly to this list. The 
                    default argument will return a single rectified color image. 
        '''
        # Color is default frame type
        if (frame_type is None):
            frame_type = [KinectIm.COLOR]

        # Continually poll the listener for a non-none frame
        while True:
            if (self.listener.hasNewFrame()):

                # Get the new frame
                frames = self.listener.waitForNewFrame()

                # Get each frame 
                return_frames = []
                for ft in frame_type:
                    if (ft == KinectIm.COLOR):
                        f, _ = self.get_registered_frame(frames)
                        return_frames.append(f)
                    elif (ft == KinectIm.DEPTH):
                        f = self.get_depth_frame(frames)
                        return_frames.append(f)
                    elif (ft == KinectIm.IR):
                        return_frames.append(self.get_ir_frame(frames))
                    elif (ft == KinectIm.RAW_COLOR):
                        return_frames.append(self.get_raw_color_frame(frames))
                    elif (ft == KinectIm.RAW_DEPTH):
                        return_frames.append(self.get_raw_depth_frame(frames))

                # Release and return the frames
                self.listener.release(frames)
                return return_frames

    def get_color_ptcld(self, roi=None, scale=1):
        '''
            get_color_ptcld: Returns a point cloud with color, generated from registered
                color and depth images. Units are mm by default.
            ARGUMENTS:
                roi: ([x, y, w, h])
                    If specified, will crop the point cloud according to the input roi. 
                    Does not accelerate runtime. 
                scale: (int)
                    Scales the point cloud such that ptcl = ptcl (mm) / scale. 
                    ie scale = 10 returns point cloud in meters. 
        '''
        while True:
            if self.listener.hasNewFrame():
                # Get registered and undistorted frames
                frames = self.listener.waitForNewFrame()
                registered, undistorted = self.get_registered_frame(frames)
                self.listener.release(frames)

                # Get point cloud 
                ptcld = self._depthMatrixToPointCloudPos(undistorted, self.CameraParams, scale=scale)

                # Adujust point cloud based on real world coordinates
                ptcld = self._applyCameraMatrixOrientation(ptcld, self.CameraPosition)

                # Reshape to correct size
                ptcld = ptcld.reshape(DEPTH_SHAPE[1], DEPTH_SHAPE[0], 3)

                # Format the color registration map - To become the "color" input for the scatterplot's setData() function.
                colors = np.divide(registered, 255) # values must be between 0.0 - 1.0
                colors = colors.reshape(colors.shape[0] * colors.shape[1], 4) # From: Rows X Cols X RGB -to- [[r,g,b],[r,g,b]...]
                colors = colors[:, :3:]  # remove alpha (fourth index) from BGRA to BGR
                colors = colors[...,::-1] #BGR to RGB
                
                # If roi, extract
                if (roi is not None):
                    if (isinstance(roi, tuple) or isinstance(roi, list)):
                        [y, x, h, w] = roi
                        xmin, xmax = int(x), int(x+w)
                        ymin, ymax = int(y), int(y+h)
                        ptcld = ptcld[xmin:xmax, ymin:ymax, :]
                        registered = registered[xmin:xmax, ymin:ymax, :]
                    else:
                        roi = np.clip(roi, 0, 1)
                        for c in range(3):
                            ptcld[:,:,c] = np.multiply(ptcld[:,:,c], roi) 

                return ptcld, colors
        
    def get_ptcld(self, roi=None, scale=1):
        '''
            get_color_ptcld: Returns a point cloud, generated from undistorted
                depth image. Units are mm.
            ARGUMENTS:
                roi: ([x, y, w, h])
                    If specified, will crop the point cloud according to the input roi. 
                    Does not accelerate runtime. 
                scale: (int)
                    Scales the point cloud such that ptcl = ptcl (mm) / scale. 
                    ie scale = 10 returns point cloud in meters. 
        '''
        while True:
            if self.listener.hasNewFrame():
                # Get undistorted frame
                frames = self.listener.waitForNewFrame()
                undistorted = self.get_depth_frame(frames)
                self.listener.release(frames)
                
                # Get point cloud 
                ptcld = self._depthMatrixToPointCloudPos(undistorted, self.CameraParams, scale=scale)

                # Adujust point cloud based on real world coordinates
                ptcld = self._applyCameraMatrixOrientation(ptcld, self.CameraPosition)

                # Reshape to correct size
                ptcld = ptcld.reshape(DEPTH_SHAPE[1], DEPTH_SHAPE[0], 3)

                # If roi, extract
                if (roi is not None):
                    if (isinstance(roi, tuple) or isinstance(roi, list)):
                        [y, x, h, w] = roi
                        xmin, xmax = int(x), int(x+w)
                        ymin, ymax = int(y), int(y+h)
                        ptcld = ptcld[xmin:xmax, ymin:ymax, :]
                    else:
                        roi = np.clip(roi, 0, 1)
                        for c in range(3):
                            ptcld[:,:,c] = np.multiply(ptcld[:,:,c], roi) 

                return ptcld  

    def get_perspective_depth(self, principal_point, view_vector, fov, roi=None, auto_adjust=False):
        '''
            get_perspective_depth: Generates a simulated depth map from the input location, orientation 
                and fov. Currently the depth map uses orthographic perspective. Depth map will appear as
                if the camera was moved from the downward vector (0, 0, -1) to the target vector, ie the
                up and down orientation will follow the minimum rotation between the vectors.  
            ARGUMENTS:
                principal_point: (x, y, z)
                    Principal point of the virtual camera.
                view_vector: (x, y, z)
                    Normal vector of the camera viewing plane (sign matters).
                fov: (x, y, z)
                    Size of viewing plane (mm).
                roi: (x, y, z)
                    If enabled, will crop the point cloud before rendering the depth image. 
                auto_adjust: (x, y, z)
                    If enabled, will change the pricipal point and fov to give an ideal view of the roi. 
        '''

        # View vector has to be normalized
        view_vector = list(np.asarray(view_vector) / np.linalg.norm(np.asarray(view_vector)))

        # Get the source point cloud
        ptcld = self.get_ptcld(roi=roi)
        ptcld_shape = ptcld.shape

        # Auto adjust places the principal point 50 mm above the center of the point cloud. 
        if (auto_adjust[0] and (roi is not None)):
            principal_point = [(abs(vv) * 50) + cp for vv, cp in zip(view_vector, get_image_center_point(ptcld))]

        # Extract size of resulting image
        (yfov, xfov) = fov
        ymin, ymax = -int(yfov/2), yfov - int(yfov/2)
        xmin, xmax = -int(xfov/2), xfov - int(xfov/2)

        # Vectorize point cloud
        ptcld = ptcld.reshape(ptcld_shape[1] * ptcld_shape[0], 3)

        # List of vectors from the principal point to each point in the cloud
        vector_cloud = np.repeat(np.expand_dims(np.asarray(principal_point), axis=0), (ptcld_shape[1] * ptcld_shape[0]), axis=0)
        vector_cloud = np.subtract(ptcld, vector_cloud)

        # In order to get the vector projections in x and y, align the view vector (and points) with the z axis 
        new_view_vector = [0, 0, -1]
        d = np.dot(new_view_vector, view_vector)

        # Sometimes the vectors are already aligned
        if (d == -1):
            valid_points = vector_cloud[:, 2] >= 0.0
            rotm = jnp.identity(3)
        elif (d == 1):
            valid_points = vector_cloud[:, 2] <= 0.0
            rotm = np.identity(3)
        else:
            # Sometimes they aren't and we need to rotate all the points
            angle = math.acos(np.dot(view_vector, new_view_vector))
            axis = np.cross(view_vector, new_view_vector)
            q = Quaternion(axis=axis, angle=angle)
            rotm = q.rotation_matrix
            vector_cloud = np.transpose(np.matmul(np.array(rotm), np.transpose(vector_cloud)))
            valid_points = vector_cloud[:, 2] <= 0.0
        # Discard points on the wrong side of the camera
        vector_cloud = vector_cloud[valid_points, :]

        # This will resize the fov, if desired
        if (auto_adjust[1] and (roi is not None)):
            min_pt = principal_point + vector_cloud[0,:]
            (yfov, xfov) = int(2*abs(principal_point[0] - min_pt[0])), int(2*abs(principal_point[1] - min_pt[1]))
            fov = [yfov, xfov]
            ymin, ymax = -int(yfov/2), yfov - int(yfov/2)
            xmin, xmax = -int(xfov/2), xfov - int(xfov/2)

        # Did this get rid of all the points?
        if (vector_cloud.shape[0] == 0):
            return np.matrix(np.zeros((fov))), np.linalg.inv(rotm), principal_point

        # Get the corresponding vector distances
        vector_dists = np.apply_along_axis(lambda v: math.sqrt(np.dot(v, v)), 1, vector_cloud)

        # Create the image
        points = np.zeros((yfov*xfov,2))
        values = np.zeros((yfov*xfov,1))
        for i, v in enumerate(vector_cloud):
            [y, x] = [int(-q) for q in v[:-1]]
            if (x >= xmin) and (x < xmax) and (y >= ymin) and (y < ymax):
                [y, x] = (y) - ymin,  (x) - xmin

                pt_idx = np.ravel_multi_index((y, x), fov)
                points[pt_idx, :] = [int(y), int(x)]
                if (values[pt_idx, :] > 0):
                    values[pt_idx, :] = min(vector_dists[i], values[pt_idx, :])
                else:
                    values[pt_idx, :] = vector_dists[i]
        points = points[values[:,0] > 0.0, :]
        values = values[values[:,0] > 0.0, :]

        # Generates the sparse depth map
        #bins = np.zeros((fov))
        #for i,p in enumerate(points):
        #    bins[int(p[0]), int(p[1])] = values[i]
                    
        if (points.shape[0] == 0):
            return np.matrix(np.zeros((fov))), np.linalg.inv(rotm), principal_point
        
        #values = values
    
        grid_y, grid_x  = np.mgrid[0:xfov, 0:yfov]
        grid = griddata(points, values, (grid_x, grid_y), method='nearest', fill_value=np.max(values))
        grid = np.reshape(np.nan_to_num(grid.T), (xfov, yfov, 1))

        # Apply a blur to simulate a depth map
        #blur_amnt = int(math.ceil((min(fov) / 50)))
        blur_amnt = 5
        if (blur_amnt > 1):
            grid = cv2.blur(grid, (blur_amnt, blur_amnt))
            grid = cv2.GaussianBlur(grid, (blur_amnt, blur_amnt),0)
        
        return grid, np.linalg.inv(rotm), principal_point

    def get_roi(self, frame_type=None):
        '''
            get_roi: When called, opens a cv2 select window and allows the user to 
                select an roi on the input frame type. Can be called with multiple 
                frame types but will only allow the user to select an roi on the 
                first one.
            ARGUMENTS:
                frame_type: ([KinectIm])
                    A list of frame types.
        '''
        frame = self.get_frame(frame_type)[0]
        [x, y, w, h] = cv2.selectROI(frame, True, False)
        cv2.destroyAllWindows()
        return [x, y, w, h]

    def record(self, filename=None):
        '''
            record: Records a video from the raw color kinect video. Video
                is saved as a 1080p avi. 
            ARGUMENTS:
                filename: (str)
                    Name to save the video with. 
        '''
        from datetime import datetime

        # Create the filename 
        now = datetime.now()
        if (filename is None):
            filename = 'video'+now.strftime("%m_%d_%Y_%H_%M_%S")+'.avi'

        # Make sure the file extension is correct
        _, ext = os.path.splitext(filename)
        if (ext != '.avi'):
            print('Only supported recording type is avi.')
            exit(-1)

        # Create the video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(filename, fourcc, 25, COLOR_SHAPE)

        # Record. On keyboard interrupt close and save. 
        try:
            while True:
                frame = self.get_frame([KinectIm.RAW_COLOR])[0]
                frame = frame[0][:,:,:3]
                cv2.imshow("KinectVideo", frame)
                out.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            out.release()
            cv2.destroyAllWindows()
        except KeyboardInterrupt:
            out.release()
            cv2.destroyAllWindows()
            
    def new_frame(self, shape):
        '''
            new_frame: Return a pylibfreenect2 frame with the dimensions specified.
                Note: Frames are pointers, and as such returning images stored in
                frames directly results in strange behavior. 
        '''
        return Frame(shape[0], shape[1], shape[2])

    def get_raw_color_frame(self, frames):
        '''
            get_raw_color_frame: Return the current rgb image as a cv2 image.
        '''
        return cv2.resize(frames["color"].asarray(), COLOR_SHAPE)

    def get_raw_depth_frame(self, frames):
        '''
            get_raw_depth_frame: Return the current depth image as a cv2 image.
        '''
        return cv2.resize(frames["depth"].asarray() / DEPTH_NORMALIZATION, COLOR_SHAPE)

    def get_ir_frame(self, frames):
        '''
            get_ir_depth_frame: Return the current ir image as a cv2 image.
        '''
        return cv2.resize(frames["ir"].asarray() / IR_NORMALIZATION, COLOR_SHAPE)
    
    def get_depth_frame(self, frames):
        '''
            get_depth_frame: Return the current undistorted depth image.
        '''
        undistorted = self.new_frame(DEPTH_SHAPE)
        self.registration.undistortDepth(frames["depth"], undistorted)
        undistorted = np.copy(undistorted.asarray(np.float32))
        return undistorted

    def get_registered_frame(self, frames):
        '''
            get_registered: Return registered color and undistorted depth image.
        '''
        registered = self.new_frame(DEPTH_SHAPE)
        undistorted = self.new_frame(DEPTH_SHAPE)
        self.registration.undistortDepth(frames["depth"], undistorted)
        self.registration.apply(frames["color"], frames["depth"], undistorted, registered)
        undistorted = np.copy(undistorted.asarray(np.float32))
        registered = np.copy(registered.asarray(np.uint8))
        return registered, undistorted

    def _depthMatrixToPointCloudPos(self, z, camera_params, scale=1000):
        '''
            Credit to: Logic1
            https://stackoverflow.com/questions/41241236/vectorizing-the-kinect-real-world-coordinate-processing-algorithm-for-speed
        '''
        # bacically this is a vectorized version of depthToPointCloudPos()
        # calculate the real-world xyz vertex coordinates from the raw depth data matrix.
        C, R = np.indices(z.shape)

        R = np.subtract(R, camera_params['cx'])
        R = np.multiply(R, z)
        R = np.divide(R, camera_params['fx'] * scale)

        C = np.subtract(C, camera_params['cy'])
        C = np.multiply(C, z)
        C = np.divide(C, camera_params['fy'] * scale)

        return np.column_stack((z.ravel() / scale, R.ravel(), -C.ravel()))

    def _applyCameraMatrixOrientation(self, pt, camera_position):
        '''
            Credit to: Logic1
            https://stackoverflow.com/questions/41241236/vectorizing-the-kinect-real-world-coordinate-processing-algorithm-for-speed
        '''
        # Kinect Sensor Orientation Compensation
        # bacically this is a vectorized version of applyCameraOrientation()
        # uses same trig to rotate a vertex around a gimbal.
        def rotatePoints(ax1, ax2, deg):
            # math to rotate vertexes around a center point on a plane.
            hyp = np.sqrt(pt[:, ax1] ** 2 + pt[:, ax2] ** 2) # Get the length of the hypotenuse of the real-world coordinate from center of rotation, this is the radius!
            d_tan = np.arctan2(pt[:, ax2], pt[:, ax1]) # Calculate the vertexes current angle (returns radians that go from -180 to 180)

            cur_angle = np.degrees(d_tan) % 360 # Convert radians to degrees and use modulo to adjust range from 0 to 360.
            new_angle = np.radians((cur_angle + deg) % 360) # The new angle (in radians) of the vertexes after being rotated by the value of deg.

            pt[:, ax1] = hyp * np.cos(new_angle) # Calculate the rotated coordinate for this axis.
            pt[:, ax2] = hyp * np.sin(new_angle) # Calculate the rotated coordinate for this axis.

        rotatePoints(1, 2, camera_position['roll']) #rotate on the Y&Z plane # Usually disabled because most tripods don't roll. If an Inertial Nav Unit is available this could be used)
        rotatePoints(0, 2, camera_position['elevation']) #rotate on the X&Z plane
        rotatePoints(0, 1, camera_position['azimuth']) #rotate on the X&Y

        # Apply offsets for height and linear position of the sensor (from viewport's center)
        pt[:] += np.float_([camera_position['x'], camera_position['y'], camera_position['z']])

        return pt

if __name__ == '__main__':
    # Create the kinect object
    k = Kinect()

    while True:
        # Specify as many types as you need here
        color_frame = k.get_frame([KinectIm.COLOR])[0]

        cv2.imshow('frame', color_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break