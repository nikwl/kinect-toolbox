'''
    kinecttb: An intuitive interface for the kinectv2.
    ==================================================

    Copyright (c) 2020 by Nikolas Lamb.
'''

# To use, need to export freenect install location
# export LIBFREENECT2_INSTALL_PREFIX=~/freenect2
# export LD_LIBRARY_PATH=$HOME/freenect2/lib:$LD_LIBRARY_PATH

from .kinecttb import *
from .constants import *

if (os.getenv('LIBFREENECT2_INSTALL_PREFIX') is None):
    import warnings
    warnings.warn("LIBFREENECT2_INSTALL_PREFIX environment variable not set", 
        ImportWarning)