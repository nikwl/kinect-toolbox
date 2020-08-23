import sys
import os
import json
import math
from enum import Enum

import cv2
import numpy as np
from pyquaternion import Quaternion
from scipy.interpolate import griddata

from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame

DEPTH_SHAPE = (int(512), int(424), int(4))
COLOR_SHAPE = (int(1920), int(1080))

class KFrame(Enum):
    ''' KFrame: Specify kinect frame types '''
    COLOR = 0
    DEPTH = 1
    IR = 2
    RAW_COLOR = 3
    RAW_DEPTH = 4

class Kinect():
    def __init__(self, params_file=None, device_index=0, headless=False, pipeline=None):
        '''
            Kinect:  Kinect interface using the libfreenect library. Will query
                libfreenect for available devices, if none are found will throw
                a RuntimeError. Otherwise will open the device for collecting
                color, depth, and ir data. 

                Use kinect.get_frame([KFrame.<>]) within a typical opencv 
                capture loop to collect data from the kinect.

                When instantiating, optionally pass a parameters file with 
                the kinect's position and orientation in the worldspace and/or
                the kinect's intrinsic parameters. See _load_camera_params for
                formatting. 
            ARGUMENTS:
                params_file: str
                    Path to a kinect parameters file (.json).
                device_index: int
                    Use to interface with a specific device if more than one is
                    connected. 
                headless: bool
                    If true, will allow the kinect to collect and process data
                    without a display. Usefull if the kinect is plugged into a 
                    server. 
                pipeline: PacketPipeline
                    Optionally pass a pylibfreenect2 pipeline that will be used
                    with the kinect. Possible types are as follows:
                        OpenGLPacketPipeline
                        CpuPacketPipeline
                        OpenCLPacketPipeline
                        CudaPacketPipeline
                        OpenCLKdePacketPipeline
                    Note that this will override the headless argument - 
                    Headless requires CUDA or OpenCL pipeline. 
        '''

        self.fn = Freenect2()
        num_devices = self.fn.enumerateDevices()
        if (num_devices == 0):
            raise RuntimeError('No device connected!')
        if (device_index >= num_devices):
            raise RuntimeError('Device index not availible!')

        # Import pipeline depending on headless state
        if (pipeline is None):
            pipeline = self._import_pipeline(headless)
        print("Packet pipeline:", type(pipeline).__name__)
        self.device = self.fn.openDevice(self.fn.getDeviceSerialNumber(device_index), 
            pipeline=pipeline)

        # We want all the types
        types = (FrameType.Color | FrameType.Depth | FrameType.Ir)
        self.listener = SyncMultiFrameListener(types)

        self.device.setColorFrameListener(self.listener)
        self.device.setIrAndDepthFrameListener(self.listener)
        self.device.start()

        # We'll use this to generate registered depth images
        self.registration = Registration(self.device.getIrCameraParams(),
                                         self.device.getColorCameraParams())
   
        # Try an load parameters
        position, intrinsic_params = self._load_camera_params(params_file)
        if (position is not None):
            self.CameraPosition = position
        if (intrinsic_params is not None):
            self.CameraParams = intrinsic_params
        else:
            # Kinects's intrinsic parameters based on v2 hardware
            self.CameraParams = {
                "cx":self.device.getIrCameraParams().cx,
                "cy":self.device.getIrCameraParams().cy,
                "fx":self.device.getIrCameraParams().fx,
                "fy":self.device.getIrCameraParams().fy
            }

    def __del__(self):
        self.device.stop()

    @staticmethod
    def _import_pipeline(headless=False):
        '''
            _import_pipeline: Imports the pylibfreenect2 pipeline based on
                whether or not headless mode is enabled. Unfortunately 
                more intelligent importing cannot be implemented (contrary
                to the example scripts) because the import raises a C
                exception, not a python one. As a result the only pipelines
                this function will load are OpenGL or OpenCL.
            ARGUMENTS:
                headless: bool
                    whether or not to run kinect in headless mode.
        '''
        if headless:
            from pylibfreenect2 import OpenCLPacketPipeline
            pipeline = OpenCLPacketPipeline()
        else:
            from pylibfreenect2 import OpenGLPacketPipeline
            pipeline = OpenGLPacketPipeline()
        return pipeline

    @staticmethod
    def _load_camera_params(params_file=None):
        '''
            _load_camera_params: Optionally give the kinect's position and 
                orientation in the worldspace and/or the kinect's intrinsic 
                parameters by passing a json file. An example dictionary is 
                shown below:

                    kinect_params = {
                        "transform": {
                            "x": 0,
                            "y": 0,
                            "z": 0.810,
                            "roll": 0,
                            "azimuth": 0,
                            "elevation": -34
                        }
                        "intrinsic_parameters": {
                            "cx": 256.684,
                            "cy": 207.085,
                            "fx": 366.193,
                            "fy": 366.193
                        }
                    }

                Specifically, the dict may contain the "transform" or 
                "intrinsic_parameters" keys, with the above fields.
            ARGUMENTS:
                params_file: str
                    Path to a kinect parameters file (.json). 
        '''

        if (params_file is None):
            # If the file wasn't passed then use the default file, if it exists
            params_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'kinect.conf')
            if (not os.path.exists(params_file)):
                print('Default kinect parameters file at {} not found, running with automatically generated parameters'.format(params_file))
                return None, None
        else:
            # If the file was passed but isn't there, warn
            if (not os.path.exists(params_file)): 
                print('Kienct parameters file at {} not found, running with automatically generated parameters'.format(params_file))
                return None, None
        
        with open(params_file, 'r') as infile:
            params_dict = json.load(infile)

        # If the keys do not exist, return None
        position, intrinsic_params = params_dict.get('transform', None), params_dict.get('intrinsic_parameters', None)
        return position, intrinsic_params

    def get_frame(self, frame_type=KFrame.COLOR):
        '''
            get_frame: Returns singleton or list of frames corresponding to 
                input. Frames can be any of the following types:
                    KFrame.COLOR     - returns a 512 x 424 color image, 
                        registered to depth image
                    KFrame.DEPTH     - returns a 512 x 424 undistorted depth 
                        image (units are m)
                    KFrame.IR        - returns a 512 x 424 ir image
                    KFrame.RAW_COLOR - returns a 1920 x 1080 color image
                    KFrame.RAW_DEPTH - returns a 512 x 424 depth image 
                        (units are mm)
            ARGUMENTS:
                frame_type: [KFrame] or KFrame
                    A list of frame types. Output corresponds directly to this 
                    list. The default argument will return a single registered 
                    color image. 
        '''
        
        def get_single_frame(ft):
            if (ft == KFrame.COLOR):
                f, _ = self._get_registered_frame(frames)
                return f
            elif (ft == KFrame.DEPTH):
                return self._get_depth_frame(frames)
            elif (ft == KFrame.RAW_COLOR):
                return self._get_raw_color_frame(frames)
            elif (ft == KFrame.RAW_DEPTH):
                return self._get_raw_depth_frame(frames)
            elif (ft == KFrame.IR):
                return self._get_ir_frame(frames)
            else:
                raise RuntimeError('Unknown frame type {}'.format(ft))

        # If just one frame type was passed
        frames = self.listener.waitForNewFrame()
        if isinstance(frame_type, KFrame):
            return_frames = get_single_frame(frame_type)
        else:
            return_frames = [None] * len(frame_type)
            for i, ft in enumerate(frame_type):
                return_frames[i] = get_single_frame(ft)
        self.listener.release(frames)

        return return_frames

    def get_ptcld(self, roi=None, scale=1000, colorized=False):
        '''
            get_ptcld: Returns a point cloud, generated from depth image. Units 
                are mm by default.
            ARGUMENTS:
                roi: [x, y, w, h]
                    If specified, will crop the point cloud according to the 
                    input roi. Does not accelerate runtime. 
                scale: int
                    Scales the point cloud such that ptcl = ptcl (m) / scale. 
                    ie scale = 1000 returns point cloud in mm. 
                colorized: bool
                    If True, returns color matrix along with point cloud such 
                    that if pt = ptcld[x,y,:], the color of that point is color 
                    = color[x,y,:]
        '''
        # Get undistorted (and registered) frames
        frames = self.listener.waitForNewFrame()
        if (colorized):
            registered, undistorted = self._get_registered_frame(frames)
        else:
            undistorted = self._get_depth_frame(frames)
        self.listener.release(frames)

        # Get point cloud 
        ptcld = self._depthMatrixToPointCloudPos(undistorted, self.CameraParams, scale=scale)

        # Adjust point cloud based on real world coordinates
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
                if (colorized):
                    registered = registered[xmin:xmax, ymin:ymax, :]
            else:
                roi = np.clip(roi, 0, 1)
                for c in range(3):
                    ptcld[:,:,c] = np.multiply(ptcld[:,:,c], roi) 

        if (colorized):
            # Format the color registration map - To become the "color" input for the scatterplot's setData() function.
            colors = np.divide(registered, 255) # values must be between 0.0 - 1.0
            colors = colors.reshape(colors.shape[0] * colors.shape[1], 4) # From: Rows X Cols X RGB -to- [[r,g,b],[r,g,b]...]
            colors = colors[:, :3:]  # remove alpha (fourth index) from BGRA to BGR
            colors = colors[...,::-1] #BGR to RGB
            return ptcld, colors
        
        return ptcld 

    def get_perspective_depth(self, principal_point, view_vector, fov, roi=None, auto_adjust=False, blur_amnt=5):
        '''
            get_perspective_depth: Generates a simulated orthographic depth map 
                from the input location and orientation and with the specified 
                fov. The depth map is generated by tilting a point cloud so 
                that the viewing vector is coincident with the z axis, 
                identifying those points that are within the specified fov,
                and filling in pixels of the depth map based on whether or not 
                a point appears there. Then scipy.gridata fills in the holes, 
                and the result is smoothed using a combination "average" and 
                gaussian blur. 
            ARGUMENTS:
                principal_point: (x, y, z)
                    Principal point of the virtual camera.
                view_vector: (x, y, z)
                    Signed normal vector of the virtual camera viewing plane.
                fov: (x, y)
                    Size of viewing plane (mm).
                roi: [x, y, w, h]
                    Optional, will crop the point cloud before rendering the 
                    depth image.
                auto_adjust: bool
                    Optional, will change the principal point and fov to give 
                    good view of the roi. 
                blur_amnt: int
                    How much blur to apply to the resulting depth map. Will 
                    help fill in much of the blank space. 
        '''

        # View vector has to be normalized
        view_vector = list(np.asarray(view_vector) / np.linalg.norm(np.asarray(view_vector)))

        # Get the source point cloud
        ptcld = self.get_ptcld(roi=roi)
        ptcld_shape = ptcld.shape

        # Auto adjust places the principal point 50 mm above the center of the point cloud. 
        if (auto_adjust[0] and (roi is not None)):
            def get_image_center_point(image, return_coords=False):
                s = image.shape
                r, c = int(s[0]/2), int(s[1]/2)
                if return_coords:
                    return image[r, c,...], (r,c)
                return image[r, c,...]
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
            rotm = np.identity(3)
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

        if (points.shape[0] == 0):
            return np.matrix(np.zeros((fov))), np.linalg.inv(rotm), principal_point
    
        # Fill in the holes
        grid_y, grid_x  = np.mgrid[0:xfov, 0:yfov]
        grid = griddata(points, values, (grid_x, grid_y), method='nearest', fill_value=np.max(values))
        grid = np.reshape(np.nan_to_num(grid.T), (xfov, yfov, 1))

        # Apply a blur to simulate a depth map
        if (blur_amnt > 1):
            grid = cv2.blur(grid, (blur_amnt, blur_amnt))
            grid = cv2.GaussianBlur(grid, (blur_amnt, blur_amnt),0)
        
        return grid, np.linalg.inv(rotm), principal_point

    def get_roi(self, frame_type=None):
        '''
            get_roi: Simply, this functions as a wrapper for the opencv method
                selectROI. When called, collects a new frame of the type
                specified, opens a cv2 window and allows the user to select a
                region.
            ARGUMENTS:
                frame_type: KFrame
                    A single frame type.
        '''
        frame = self.get_frame(frame_type)
        [x, y, w, h] = cv2.selectROI('roi window', frame, True, False)
        cv2.destroyWindow('roi window')
        return frame[x:x+w, y:y+h], [x, y, w, h]

    def record(self, frame_type=KFrame.COLOR, video_codec='XVID', filename=None):
        '''
            record: Records a video of the type specified. If no filename is 
                given, records as a .avi. Do not call this in conjunction with
                a typical cv2 display loop. 
            ARGUMENTS:
                filename: (str)
                    Name to save the video with. 
        '''
        from datetime import datetime

        # Create the filename 
        if (filename is None):
            filename = 'kinect_{}.avi'.format(datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))

        # Create the video writer
        if (frame_type == KFrame.RAW_COLOR):
            shape = COLOR_SHAPE
        else:
            shape = DEPTH_SHAPE[:2]
        fourcc = cv2.VideoWriter_fourcc(*video_codec)
        out = cv2.VideoWriter(filename, fourcc, 25, shape)

        # Record. On keyboard interrupt close and save. 
        try:
            while True:
                frame = self.get_frame(frame_type=frame_type)
                if (frame_type == KFrame.RAW_COLOR):
                    frame = frame[:,:,:3]                
                cv2.imshow("KinectVideo", frame)
                out.write(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            out.release()
            cv2.destroyAllWindows()
        except KeyboardInterrupt:
            pass
        finally:
            out.release()
            cv2.destroyAllWindows()

    @staticmethod   
    def _new_frame(shape):
        '''
            _new_frame: Return a pylibfreenect2 frame with the dimensions
                specified. Note that Frames are pointers, and as such returning 
                using numpy images created from them directly results in some
                strange behavior. After using the frame, it is highly 
                recommended to copy the resulting image using np.copy() rather 
                than returning the Frame referencing array directly. 
        '''
        return Frame(shape[0], shape[1], shape[2])

    @staticmethod
    def _get_raw_color_frame(frames):
        ''' _get_raw_color_frame: Return the current rgb image as a cv2 image. '''
        return cv2.resize(frames["color"].asarray(), COLOR_SHAPE)

    @staticmethod
    def _get_raw_depth_frame(frames):
        ''' _get_raw_depth_frame: Return the current depth image as a cv2 image. '''
        return cv2.resize(frames["depth"].asarray(), DEPTH_SHAPE[:2])

    @staticmethod
    def _get_ir_frame(frames):
        ''' get_ir_depth_frame: Return the current ir image as a cv2 image. '''
        return cv2.resize(frames["ir"].asarray(), DEPTH_SHAPE[:2])
    
    def _get_depth_frame(self, frames):
        ''' _get_depth_frame: Return the current undistorted depth image. '''
        undistorted = self._new_frame(DEPTH_SHAPE)
        self.registration.undistortDepth(frames["depth"], undistorted)
        undistorted = np.copy(undistorted.asarray(np.float32))
        return undistorted

    def _get_registered_frame(self, frames):
        ''' get_registered: Return registered color and undistorted depth image. '''
        registered = self._new_frame(DEPTH_SHAPE)
        undistorted = self._new_frame(DEPTH_SHAPE)
        self.registration.undistortDepth(frames["depth"], undistorted)
        self.registration.apply(frames["color"], frames["depth"], undistorted, registered)
        undistorted = np.copy(undistorted.asarray(np.float32))
        registered = np.copy(registered.asarray(np.uint8))[...,:3]
        return registered, undistorted

    @staticmethod
    def _depthMatrixToPointCloudPos(z, camera_params, scale=1):
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

    @staticmethod
    def _applyCameraMatrixOrientation(pt, camera_position=None):
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

        if (camera_position is not None):
            rotatePoints(1, 2, camera_position['roll']) #rotate on the Y&Z plane # Usually disabled because most tripods don't roll. If an Inertial Nav Unit is available this could be used)
            rotatePoints(0, 2, camera_position['elevation']) #rotate on the X&Z plane
            rotatePoints(0, 1, camera_position['azimuth']) #rotate on the X&Y

            # Apply offsets for height and linear position of the sensor (from viewport's center)
            pt[:] += np.float_([camera_position['x'], camera_position['y'], camera_position['z']])

        return pt