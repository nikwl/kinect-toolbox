'''
kinect toolbox: A more intuitive interface for the kinectv2.
==================================================

Copyright (c) 2020 by Nikolas Lamb.
'''

from .kinect import Kinect
from .constants import *

# To use, need to export freenect install location
# export LIBFREENECT2_INSTALL_PREFIX=~/freenect2
# export LD_LIBRARY_PATH=$HOME/freenect2/lib:$LD_LIBRARY_PATH
import os
if (os.getenv('LIBFREENECT2_INSTALL_PREFIX') is None or 
        (os.getenv('LD_LIBRARY_PATH') is not None and 
        'freenect2' not in os.getenv('LD_LIBRARY_PATH'))):
    import warnings
    warnings.warn("LIBFREENECT2_INSTALL_PREFIX environment variable not set", ImportWarning)